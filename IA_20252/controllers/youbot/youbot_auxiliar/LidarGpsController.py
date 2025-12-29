# controllers/youbot/youbot_auxiliar/LidarGpsController.py
from collections import deque
from pathlib import Path
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


class LidarGpsController:
    def __init__(self, lidar_device=None, model_path=None, T=3, max_range=5.5, device=None, smoothing=5, debug=False):
        self.lidar = lidar_device
        self.model_path = Path(model_path) if model_path else None
        self.T = T
        self.MAX_RANGE = max_range
        self.buffer = deque(maxlen=self.T)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.n_rays = None
        self.smoothing = smoothing
        self.debug = debug
        self.recent_poses = deque(maxlen=self.smoothing)

    def init_after_first_step(self):
        if self.lidar is None:
            if self.debug: print("⚠️ Lidar missing for LidarGpsController")
            return False
        try:
            ranges = self.lidar.getRangeImage()
            self.n_rays = len(ranges)
        except Exception:
            self.n_rays = 360
            if self.debug: print("⚠️ could not read lidar ranges; using N=360 fallback")

        self.model = LidarPoseCNN(self.T, self.n_rays).to(self.device)
        if self.model_path and self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
                model_state = self.model.state_dict()
                for k, v in state_dict.items():
                    if k in model_state and v.size() == model_state[k].size():
                        model_state[k] = v
                self.model.load_state_dict(model_state)
                if self.debug: print("✅ Lidar GPS model loaded")
            except Exception as e:
                if self.debug: print("❌ Error loading model:", e)
        else:
            if self.debug: print("⚠️ Model checkpoint not found; using random weights")
        self.model.eval()
        return True

    def _preprocess_scan(self, scan):
        arr = np.array(scan, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=self.MAX_RANGE, posinf=self.MAX_RANGE, neginf=self.MAX_RANGE)
        arr = np.clip(arr, 0.05, self.MAX_RANGE) / self.MAX_RANGE
        return arr

    def step(self):
        if self.lidar is None or self.model is None:
            return None
        scan = self.lidar.getRangeImage()
        if scan is None:
            return None
        pre = self._preprocess_scan(scan)
        self.buffer.append(pre)
        if len(self.buffer) < self.T:
            return None
        try:
            X = torch.tensor(np.stack(list(self.buffer)), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(X)[0].cpu().numpy()
            self.recent_poses.append(out)
            avg = np.mean(np.stack(list(self.recent_poses)), axis=0)
            if self.debug: print(f"📍 LidarGPS pose: x={avg[0]:.3f}, y={avg[1]:.3f}")
            return tuple(avg.tolist())
        except Exception as e:
            if self.debug: print("❌ Inference error:", e)
            return None

    def get_pose(self):
        if not self.recent_poses:
            return None
        avg = np.mean(np.stack(list(self.recent_poses)), axis=0)
        return tuple(avg.tolist())