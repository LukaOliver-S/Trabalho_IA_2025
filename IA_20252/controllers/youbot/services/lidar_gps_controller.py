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


class LidarPoseLight(nn.Module):
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

        self.head = nn.Sequential(
            nn.Linear(flat_dim, 128),  
            nn.ReLU(inplace=True),
            nn.Linear(128, n_out),
        )

    def forward(self, x):
        x = self.features(x)          # [B, C, T', W']
        x = x.view(x.size(0), -1)     # [B, flat_dim]
        return self.head(x)


class LidarGpsController:
    def __init__(self, lidar_device, model_path=None, T=3, max_range=5.5, device=None, debug=False, use_half=False):
        self.lidar = lidar_device
        self.model_path = Path(model_path) if model_path else None
        self.T = T
        self.max_range = max_range
        self.buffer = deque(maxlen=self.T)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half = use_half and "cuda" in self.device  # float16 apenas se GPU
        self.model = None
        self.n_rays = None
        self.last_pose = None
        self.debug = debug
        
    def init_after_first_step(self):
        if self.lidar is None:
            return False

        ranges = self.lidar.getRangeImage()
        self.n_rays = len(ranges)
        print(self.n_rays)

        self.model = LidarPoseLight(self.T, self.n_rays).to(self.device)

        # Carrega apenas os pesos do modelo (state_dict)
        if self.model_path and self.model_path.exists():
            state_dict = torch.load(self.model_path, map_location=self.device)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.model.load_state_dict(state_dict)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        if self.use_half:
            self.model.half()  # otimiza memória GPU

        return True

    def _preprocess_scan(self, scan):
        arr = np.nan_to_num(scan, nan=self.max_range, posinf=self.max_range, neginf=self.max_range)
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

        X = torch.from_numpy(np.stack(self.buffer)).unsqueeze(0).unsqueeze(0).float().to(self.device)
        if self.use_half:
            X = X.half()

        with torch.no_grad():
            out = self.model(X)[0].cpu().numpy()

        self.last_pose = tuple(out.tolist())
        return self.last_pose


    def get_pose(self):
        return self.last_pose
