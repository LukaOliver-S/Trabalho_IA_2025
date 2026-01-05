# controllers/youbot/youbot_auxiliar/LidarGpsController.py
from collections import deque
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import csv
from datetime import datetime

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
        enable_logging=False,  # ← NOVA FLAG
        log_dir=None,          
        ground_truth_provider=None,
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

        # ===== LOGGING =====
        self.enable_logging = enable_logging
        self.log_dir = Path(log_dir) if log_dir else Path(__file__).parent.parent / "lidar_gps_data"
        self.csv_file = None
        self.csv_writer = None
        self._tick = 0
        self._session_start_time = None
        self.ground_truth_provider = ground_truth_provider
        if self.enable_logging:
            self._start_logging_session()

    # =====================================================
    # LOGGING SETUP
    # =====================================================
    def _start_logging_session(self):
        """Cria diretório e arquivo CSV para logging"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.log_dir / f"lidar_gps_{timestamp}.csv"
        
        self.csv_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Header com colunas raw e smoothed
        self.csv_writer.writerow([
            'tick',
            'timestamp',
            'x_pred',           # ← Predição smoothed (usada no controle)
            'y_pred',           # ← Predição smoothed (usada no controle)
            'x_raw',            # ← Predição bruta da CNN
            'y_raw',            # ← Predição bruta da CNN
            'x_real',           # ← GPS ground truth
            'y_real',           # ← GPS ground truth
            'error_x',          # ← x_pred - x_real (smoothed)
            'error_y',          # ← y_pred - y_real (smoothed)
            'error_euclidean',  # ← distância euclidiana (smoothed)
            'error_x_raw',      # ← erro da predição bruta
            'error_y_raw',      # ← erro da predição bruta
            'error_euclidean_raw',  # ← distância euclidiana (raw)
            'yaw_cos',
            'yaw_sin',
            'inference_time_ms',
            'smoothed'
        ])
        
        self._session_start_time = time.time()
        self._tick = 0
        
        if self.debug:
            print(f"[LidarGpsController] Logging iniciado: {csv_path}")

    def _log_prediction(self, raw_pose, smoothed_pose, x_real, y_real, error_x, error_y, error_eucl, yaw, inference_time_ms, is_smoothed):
        """Salva predição no CSV com versões raw e smoothed"""
        if not self.enable_logging or self.csv_writer is None:
            return
        
        elapsed = time.time() - self._session_start_time
        
        # Calcular erros da predição RAW também
        error_x_raw = None
        error_y_raw = None
        error_eucl_raw = None
        if x_real is not None and y_real is not None:
            error_x_raw = raw_pose[0] - x_real
            error_y_raw = raw_pose[1] - y_real
            error_eucl_raw = np.sqrt(error_x_raw**2 + error_y_raw**2)
        
        self.csv_writer.writerow([
            self._tick,
            f"{elapsed:.3f}",
            f"{smoothed_pose[0]:.6f}",  # x_pred (smoothed)
            f"{smoothed_pose[1]:.6f}",  # y_pred (smoothed)
            f"{raw_pose[0]:.6f}",       # x_raw
            f"{raw_pose[1]:.6f}",       # y_raw
            f"{x_real:.6f}" if x_real is not None else "",
            f"{y_real:.6f}" if y_real is not None else "",
            f"{error_x:.6f}" if error_x is not None else "",
            f"{error_y:.6f}" if error_y is not None else "",
            f"{error_eucl:.6f}" if error_eucl is not None else "",
            f"{error_x_raw:.6f}" if error_x_raw is not None else "",
            f"{error_y_raw:.6f}" if error_y_raw is not None else "",
            f"{error_eucl_raw:.6f}" if error_eucl_raw is not None else "",
            f"{yaw[0]:.6f}",
            f"{yaw[1]:.6f}",
            f"{inference_time_ms:.2f}",
            int(is_smoothed)
        ])
        
        self._tick += 1

    def stop_logging(self):
        """Fecha arquivo CSV"""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            if self.debug:
                print("[LidarGpsController] Logging finalizado")

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
            # ===== LOG (sem inferência, usa última pose) =====
            if self.enable_logging and self.last_pose:
                yaw_np = self._get_current_yaw()
                gt = self.ground_truth_provider() if self.ground_truth_provider else (None, None)
                self._log_prediction(
                    self.last_pose, 
                    self.last_pose,
                    gt[0], gt[1],
                    None, None, None,  # can't calculate errors without new inference
                    yaw_np, 
                    0.0,  # sem inferência
                    False
                )
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
        
        inference_time_ms = (t1 - t0) * 1000

        if self.debug:
            print(f"[LidarGpsController] forward = {inference_time_ms:.2f} ms")
            
        raw_pose = (float(out[0]), float(out[1]))

        # ===== APLICA MÉDIA MÓVEL =====
        smoothed_pose = self._moving_average_pose(raw_pose)

        # ===== LOG =====
        if self.enable_logging:
            if self.enable_logging:
                gt = self.ground_truth_provider() if self.ground_truth_provider else (None, None)
                
                error_x = smoothed_pose[0] - gt[0] if gt[0] is not None else None
                error_y = smoothed_pose[1] - gt[1] if gt[1] is not None else None
                error_eucl = np.sqrt(error_x**2 + error_y**2) if error_x is not None else None
                
                self._log_prediction(
                    raw_pose,
                    smoothed_pose,
                    gt[0], gt[1],  # ground truth
                    error_x, error_y, error_eucl,
                    yaw_np,
                    inference_time_ms,
                    True
                )

        self.last_pose = smoothed_pose
        return self.last_pose

    def get_pose(self):
        return self.last_pose
    
    def __del__(self):
        """Garante que CSV é fechado"""
        self.stop_logging()