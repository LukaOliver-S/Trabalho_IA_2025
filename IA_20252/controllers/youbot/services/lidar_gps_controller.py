# controllers/youbot/youbot_auxiliar/LidarGpsController.py
from collections import deque
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn

class LidarPoseCNN(nn.Module):
    def __init__(self, T, n_rays, n_out=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,5), padding=(1,2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3,5), padding=(1,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,4)),
            nn.Conv2d(64, 64, kernel_size=(3,5), padding=(1,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,4)),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, T, n_rays)
            out = self.features(dummy)
            flat_dim = out.view(1, -1).size(1)
        self.head = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_out),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


class LidarPoseLight(nn.Module):
    """Lightweight model that accepts yaw (cos, sin) as extra inputs to the head."""
    def __init__(self, T, n_rays, n_out=2):
        super().__init__()
        # Canais reduzidos e pooling mais agressivo no eixo angular (W = n_rays)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,5), padding=(1,2)),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=(3,5), padding=(1,2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,4)),

            nn.Conv2d(32, 32, kernel_size=(3,5), padding=(1,2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,4)),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, T, n_rays)
            out = self.features(dummy)
            flat_dim = out.view(1, -1).size(1)

        # head receives extra 2 dims for yaw (cos, sin)
        self.head = nn.Sequential(
            nn.Linear(flat_dim + 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_out),
        )

    def forward(self, x, yaw):
        """
        x: [B,1,T,n_rays]
        yaw: [B,2] (cos, sin)
        """
        x = self.features(x)          # [B, C, T', W']
        x = x.view(x.size(0), -1)     # [B, flat_dim]
        x = torch.cat([x, yaw], dim=1)
        return self.head(x)


class LidarGpsController:
    def __init__(
        self,
        lidar_device,
        model_path=None,
        T=2,
        max_range=5.5,
        device=None,
        debug=False,
        use_half=False,
        predict_every=15,   # roda a rede a cada N steps
        yaw_provider=None,  # callable -> returns (cos, sin) or None
        model_cls=None,     # class to instantiate (LidarPoseCNN or LidarPoseLight)
    ):
        self.lidar = lidar_device
        self.model_path = Path(model_path) if model_path else None
        self.T = T
        self.max_range = max_range
        self.buffer = deque(maxlen=self.T)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half = use_half and ("cuda" in self.device)

        self.model = None
        self.n_rays = None
        self.last_pose = None
        self.debug = debug

        self.predict_every = max(1, int(predict_every))
        self._step_counter = 0

        self.yaw_provider = yaw_provider
        # last_yaw as default (cos=1,sin=0) -> yaw=0
        self.last_yaw = np.array([1.0, 0.0], dtype=np.float32)

        # allow user to pick model class (compatibility/test)
        self.model_cls = model_cls or LidarPoseLight
        
        # ===== MÉDIA MÓVEL =====
        self.ma_window = 3
        self.pose_buffer = deque(maxlen=self.ma_window)

    # =====================================================
    # MÉDIA MÓVEL
    # =====================================================

    def _moving_average_pose(self, pose):
        self.pose_buffer.append(pose)
        arr = np.array(self.pose_buffer, dtype=np.float32)
        mean = arr.mean(axis=0)
        return float(mean[0]), float(mean[1])

    # =====================================================
    
    def init_after_first_step(self):
        if self.lidar is None:
            return False

        ranges = self.lidar.getRangeImage()
        if ranges is None:
            return False

        self.n_rays = len(ranges)
        if self.debug:
            print(f"[LidarGpsController] n_rays = {self.n_rays}")

        # instantiate the requested model class
        self.model = self.model_cls(self.T, self.n_rays).to(self.device)

        # load weights (best-effort with some compatibility handling)
        if self.model_path and self.model_path.exists():
            state_dict = torch.load(self.model_path, map_location=self.device)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # try direct load first
            try:
                self.model.load_state_dict(state_dict)
                if self.debug:
                    print("[LidarGpsController] checkpoint loaded (direct).")
            except RuntimeError as e:
                # attempt partial load: load matching keys only (useful when head changed)
                if self.debug:
                    print("[LidarGpsController] checkpoint mismatch, attempting partial load:", e)
                model_state = self.model.state_dict()
                filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
                model_state.update(filtered)
                self.model.load_state_dict(model_state)
                if self.debug:
                    print(f"[LidarGpsController] partial weights loaded ({len(filtered)} tensors matched).")
        else:
            if self.debug:
                print("[LidarGpsController] no checkpoint provided / not found.")

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        if self.use_half:
            self.model.half()

        return True

    def _get_current_yaw(self):
        """Return numpy array (2,) with cos,sin. Uses yaw_provider if available or last_yaw fallback."""
        if callable(self.yaw_provider):
            try:
                y = self.yaw_provider()
                if y is not None:
                    y = np.array(y, dtype=np.float32)
                    if y.shape == (2,):
                        self.last_yaw = y
                        return y
            except Exception as e:
                if self.debug:
                    print("[LidarGpsController] yaw_provider error:", e)
        return self.last_yaw

    def _preprocess_scan(self, scan):
        arr = np.nan_to_num(
            scan,
            nan=self.max_range,
            posinf=self.max_range,
            neginf=self.max_range,
        )
        arr = np.clip(arr, 0.05, self.max_range) / self.max_range
        return arr.astype(np.float32)

    def step(self):
        if self.lidar is None or self.model is None:
            return None

        scan = self.lidar.getRangeImage()
        if scan is None:
            return None

        # atualiza buffer temporal
        self.buffer.append(self._preprocess_scan(scan))
        if len(self.buffer) < self.T:
            return None

        self._step_counter += 1
        # só roda a rede a cada N steps; nos outros, devolve última pose
        if (self._step_counter % self.predict_every) != 0:
            return self.last_pose

        buf = np.stack(self.buffer, dtype=np.float32)      # [T, n_rays]
        X = torch.from_numpy(buf).unsqueeze(0).unsqueeze(0)  # [1,1,T,n_rays]

        # prepare yaw tensor if model expects it
        yaw_np = self._get_current_yaw().astype(np.float32)  # (2,)
        yaw_t = torch.from_numpy(yaw_np).unsqueeze(0)        # [1,2]

        if self.device != "cpu":
            X = X.to(self.device, non_blocking=True)
            yaw_t = yaw_t.to(self.device, non_blocking=True)
        if self.use_half:
            X = X.half()
            yaw_t = yaw_t.half()

        t0 = time.time()
        with torch.no_grad():
            # if model supports yaw (signature), call model(X, yaw_t)
            try:
                out = self.model(X, yaw_t)[0].cpu().numpy()
            except TypeError:
                # model doesn't accept yaw param (legacy): call without yaw
                out = self.model(X)[0].cpu().numpy()
        t1 = time.time()

        if self.debug:
            print(f"[LidarGpsController] forward = {(t1 - t0)*1000:.2f} ms")
            
        raw_pose = (float(out[0]), float(out[1]))

        # ===== APLICA MÉDIA MÓVEL =====
        smoothed_pose = self._moving_average_pose(raw_pose)

        self.last_pose = smoothed_pose
        return self.last_pose

    def get_pose(self):
        return self.last_pose