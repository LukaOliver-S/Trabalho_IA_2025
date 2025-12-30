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
    def __init__(self, lidar_device, model_path=None, T=3, max_range=5.5, device=None, debug=False):
        self.lidar = lidar_device
        self.model_path = Path(model_path) if model_path else None
        self.T = T
        self.max_range = max_range
        self.buffer = deque(maxlen=self.T)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.n_rays = None
        self.last_pose = None
        
    def init_after_first_step(self):
        if self.lidar is None:
            return False
        ranges = self.lidar.getRangeImage()
        self.n_rays = len(ranges)
        self.model = LidarPoseCNN(self.T, self.n_rays).to(self.device)
        if self.model_path and self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self.model.load_state_dict(state, strict=False)
        self.model.eval()
        return True

    def _preprocess_scan(self, scan):
        arr = np.array(scan, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=self.max_range, posinf=self.max_range, neginf=self.max_range)
        arr = np.clip(arr, 0.05, self.max_range) / self.max_range
        return arr

    def step(self):
        if self.lidar is None or self.model is None:
            return None
        scan = self.lidar.getRangeImage()
        if scan is None:
            return None
        self.buffer.append(self._preprocess_scan(scan))
        if len(self.buffer) < self.T:
            return None
        X = torch.tensor(np.stack(list(self.buffer)), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(X)[0].cpu().numpy()
        self.last_pose = tuple(out.tolist())
        return self.last_pose

    def get_pose(self):
        return self.last_pose