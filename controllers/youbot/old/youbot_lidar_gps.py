#!/usr/bin/env python
from pathlib import Path
from controller import Robot, Keyboard
from base import Base
from arm import Arm
from gripper import Gripper
import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque

# ------------------------------
# CNN Model (Lidar → Pose)
# ------------------------------
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

        # calcula dimensão para a cabeça da rede
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

# ------------------------------
# YouBot Controller
# ------------------------------
class YouBotController:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())

        # Robot components
        self.base = Base(self.robot)
        self.arm = Arm(self.robot)
        self.gripper = Gripper(self.robot)

        # Keyboard
        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.time_step)

        # LiDAR
        self.lidar = self.robot.getDevice("lidar")
        if self.lidar:
            self.lidar.enable(self.time_step)
        else:
            print("⚠️ LiDAR not found. CNN pose will be disabled.")

        # Movement
        self.forward_speed = 0.3
        self.strafe_speed = 0.3
        self.movement_duration = 10
        self.move_forward_counter = 0
        self.move_backward_counter = 0
        self.strafe_left_counter = 0
        self.strafe_right_counter = 0

        # CNN placeholders
        self.T = 3 # For the kaggle CNN
        self.N = None
        self.buffer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_path = Path(r"C:\Users\User\Documents\IA_PROJECT\Archives\files_webot\IA_20252\controllers\model_processing_lidar\lidar_pose_cnn_temporal_1_kaggle.pth")
       # self.model_path = Path(r"C:\Users\User\Documents\IA_PROJECT\Archives\files_webot\IA_20252\controllers\model_processing_lidar\lidar_pose_cnn_temporal_1.pth")
        self.MAX_RANGE = 5.5

        # Debug
        self._step_counter = 0
        self._debug_every = 1

        print("YouBotController initialized.")

    # Inicializa CNN depois do primeiro step
    def init_model_after_first_step(self):
        try:
            ranges = self.lidar.getRangeImage()
            self.N = len(ranges)
        except Exception:
            self.N = 360
            print("⚠️ Could not read LiDAR ranges; fallback to N=360")

        print(f"LiDAR rays: N={self.N} | temporal T={self.T}")

        self.buffer = deque(maxlen=self.T)
        self.model = LidarPoseCNN(self.T, self.N).to(self.device)

        # Carregar checkpoint
        if os.path.isfile(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
                model_state = self.model.state_dict()
                for k, v in state_dict.items():
                    if k in model_state and v.size() == model_state[k].size():
                        model_state[k] = v
                self.model.load_state_dict(model_state)
                print("✅ CNN checkpoint loaded.")
            except Exception as e:
                print("❌ Error loading CNN checkpoint:", e)
        else:
            print("⚠️ CNN checkpoint not found. Using random weights.")

        self.model.eval()

    # Handle keyboard
    def handle_keyboard_input(self):
        key = self.keyboard.getKey()
        while key >= 0:
            if key in [ord('W'), ord('w')]: self.move_forward_counter = self.movement_duration
            elif key in [ord('S'), ord('s')]: self.move_backward_counter = self.movement_duration
            elif key in [ord('A'), ord('a')]: self.strafe_left_counter = self.movement_duration
            elif key in [ord('D'), ord('d')]: self.strafe_right_counter = self.movement_duration
            elif key in [ord('I'), ord('i')]: self.arm.increase_height()
            elif key in [ord('K'), ord('k')]: self.arm.decrease_height()
            elif key in [ord('J'), ord('j')]: self.arm.increase_orientation()
            elif key in [ord('L'), ord('l')]: self.arm.decrease_orientation()
            elif key in [ord('O'), ord('o')]: self.gripper.release()
            elif key in [ord('P'), ord('p')]: self.gripper.grip()
            elif key == ord(' '):
                self.move_forward_counter = self.move_backward_counter = 0
                self.strafe_left_counter = self.strafe_right_counter = 0
                self.base.move(0,0,0)
            elif key in [ord('Q'), ord('q')]:
                return False
            key = self.keyboard.getKey()
        return True

    # Atualiza movimento
    def update_movement(self):
        vx = vy = omega = 0.0
        if self.move_forward_counter > 0: vx = self.forward_speed; self.move_forward_counter -= 1
        elif self.move_backward_counter > 0: vx = -self.forward_speed; self.move_backward_counter -= 1
        if self.strafe_left_counter > 0: vy = self.strafe_speed; self.strafe_left_counter -= 1
        elif self.strafe_right_counter > 0: vy = -self.strafe_speed; self.strafe_right_counter -= 1
        self.base.move(vx, vy, omega)

    # Atualiza pose da CNN
    def update_cnn_pose(self, lidar_scan):
        if self.model is None or self.buffer is None or lidar_scan is None:
            return

        scan_raw = np.array(lidar_scan, dtype=np.float32)
        scan = np.nan_to_num(scan_raw, nan=self.MAX_RANGE, posinf=self.MAX_RANGE, neginf=self.MAX_RANGE)
        scan = np.clip(scan, 0.05, self.MAX_RANGE) / self.MAX_RANGE

        self._step_counter += 1
       
        self.buffer.append(scan)
        if len(self.buffer) == self.T:
            try:
                X = torch.tensor(np.stack(self.buffer), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    pose_out = self.model(X)[0].cpu().numpy()
                if (self._step_counter % self._debug_every) == 0:
                 print(f"📍 CNN Pose: x={pose_out[0]:.3f}, y={pose_out[1]:.3f}")
            except Exception as e:
                print("❌ CNN inference error:", e)

    # Loop principal
    def run(self):
        print("YouBot Controller - Keyboard + CNN")
        print("W/S/A/D: move, Space: stop, Q: quit")
        self.robot.step(self.time_step)

        if self.lidar: 
            self.init_model_after_first_step()
        else:
            print("⚠️ LiDAR not found: CNN pose disabled.")

        try:
            while self.robot.step(self.time_step) != -1:
                if not self.handle_keyboard_input(): break
                self.update_movement()
                lidar_scan = self.lidar.getRangeImage() if self.lidar else None
                self.update_cnn_pose(lidar_scan)
        except KeyboardInterrupt:
            pass
        finally:
            self.base.move(0,0,0)
            print("Stopped")

# ------------------------------
# Entrypoint
# ------------------------------
if __name__ == "__main__":
    controller = YouBotController()
    controller.run()
