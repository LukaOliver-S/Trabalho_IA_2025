from controller import Robot, Keyboard
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'youbot'))
from base import Base
from arm import Arm
from gripper import Gripper
import numpy as np
import cv2
import joblib
from pathlib import Path


class YouBotController:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())

        # ================= ROBOT =================
        self.base = Base(self.robot)
        self.arm = Arm(self.robot)
        self.gripper = Gripper(self.robot)

        # ================= KEYBOARD =================
        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.time_step)

        # ================= CAMERAS =================
        self.gripper_camera = self.robot.getDevice("gripper_camera")
        self.front_camera = self.robot.getDevice("front camera")

        if self.gripper_camera:
            self.gripper_camera.enable(self.time_step)
            print("Gripper camera enabled")

        if self.front_camera:
            self.front_camera.enable(self.time_step)
            print("Front camera enabled")

        self.active_camera = self.gripper_camera
        self.camera_mode = "gripper"

        # ================= MOVEMENT =================
        self.forward_speed = 0.3
        self.strafe_speed = 0.3
        self.movement_duration = 10

        self.move_forward_counter = 0
        self.move_backward_counter = 0
        self.strafe_left_counter = 0
        self.strafe_right_counter = 0

   

        # ================= AUTO GRAB =================
        self.auto_grab_mode = False
        self.grab_sequence_step = 0

        # ================= LIDAR & GPS =================
        self.lidar = self.robot.getDevice("lidar")  # ajuste o nome se precisar
        self.gps = self.robot.getDevice("gps")      # ajuste o nome se precisar

        if self.lidar:
            self.lidar.enable(self.time_step)
            print("LiDAR enabled")
        else:
            print("LiDAR not found!")

        if self.gps:
            self.gps.enable(self.time_step)
            print("GPS enabled")
        else:
            print("GPS not found!")

        # ================= DATASET FOLDER =================
        self.dataset_dir = os.path.join(os.path.dirname(__file__), "dataset_lidar_pose")
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.sample_count = 0

    # ==========================================================
    # CAMERA SWITCH
    # ==========================================================
    def switch_camera(self):
        cams = []
        if self.gripper_camera:
            cams.append(("gripper", self.gripper_camera))
        if self.front_camera:
            cams.append(("front", self.front_camera))
        if hasattr(self, 'detection_camera') and self.detection_camera:
            cams.append(("detection", self.detection_camera))

        idx = [c[0] for c in cams].index(self.camera_mode)
        self.camera_mode, self.active_camera = cams[(idx + 1) % len(cams)]
        print(f"📷 Switched to {self.camera_mode} camera")

    # ==========================================================
    # POSITION FOR TEST
    # ==========================================================
    def position_for_gripper_camera_test(self):
        self.active_camera = self.gripper_camera
        self.camera_mode = "gripper"

        self.arm.set_height(self.arm.RESET)
        self.arm.set_orientation(self.arm.FRONT)
        self.gripper.grip()

        print("🎯 Ready for gripper HSV test")

    # ==========================================================
    # AUTO GRAB
    # ==========================================================
    def grab_cube_sequence(self):
        if self.grab_sequence_step == 0:
            self.arm.set_height(self.arm.FRONT_FLOOR)
            self.arm.set_orientation(self.arm.FRONT)
            self.grab_sequence_step = 1
        elif self.grab_sequence_step == 1:
            self.gripper.release()
            self.grab_sequence_step = 2
        elif self.grab_sequence_step == 2:
            self.move_forward_counter = 15
            self.grab_sequence_step = 3
        elif self.grab_sequence_step == 3:
            self.gripper.grip()
            self.grab_sequence_step = 4
        elif self.grab_sequence_step == 4:
            self.arm.set_height(self.arm.FRONT_CARDBOARD_BOX)
            self.auto_grab_mode = False
            self.grab_sequence_step = 0

    # ==========================================================
    # KEYBOARD
    # ==========================================================
    def handle_keyboard_input(self):
        key = self.keyboard.getKey()
        while key >= 0:
            if key in (ord('W'), ord('w')):
                self.move_forward_counter = self.movement_duration
            elif key in (ord('S'), ord('s')):
                self.move_backward_counter = self.movement_duration
            elif key in (ord('A'), ord('a')):
                self.strafe_left_counter = self.movement_duration
            elif key in (ord('D'), ord('d')):
                self.strafe_right_counter = self.movement_duration

            elif key in (ord('I'), ord('i')):
                self.arm.increase_height()
            elif key in (ord('K'), ord('k')):
                self.arm.decrease_height()
            elif key in (ord('J'), ord('j')):
                self.arm.increase_orientation()
            elif key in (ord('L'), ord('l')):
                self.arm.decrease_orientation()

            elif key in (ord('O'), ord('o')):
                self.gripper.release()
            elif key in (ord('P'), ord('p')):
                self.gripper.grip()
            elif key in (ord('G'), ord('g')):
                self.auto_grab_mode = True
                self.grab_sequence_step = 0
            elif key in (ord('R'), ord('r')):
                self.arm.set_height(self.arm.RESET)
                self.arm.set_orientation(self.arm.FRONT)
                self.gripper.release()

            elif key in (ord('V'), ord('v')):
                self.switch_camera()
            elif key in (ord('T'), ord('t')):
                self.position_for_gripper_camera_test()
            elif key in (ord('H'), ord('h')):
                self.test_hsv()

            elif key in (ord('Q'), ord('q')):
                return False

            key = self.keyboard.getKey()
        return True

    # ==========================================================
    # MOVEMENT LOOP
    # ==========================================================
    def update_movement(self):
        vx = vy = 0.0

        if self.move_forward_counter > 0:
            vx = self.forward_speed
            self.move_forward_counter -= 1
        elif self.move_backward_counter > 0:
            vx = -self.forward_speed
            self.move_backward_counter -= 1

        if self.strafe_left_counter > 0:
            vy = self.strafe_speed
            self.strafe_left_counter -= 1
        elif self.strafe_right_counter > 0:
            vy = -self.strafe_speed
            self.strafe_right_counter -= 1

        self.base.move(vx, vy, 0)

    def update_auto_grab(self):
        if self.auto_grab_mode:
            self.grab_cube_sequence()

    # ==========================================================
    # LIDAR+GPS DATASET SAVE (LiDAR -> X, GPS -> label)
    # ==========================================================
    def save_sample(self, lidar_data, gps_data):
        sample = {
            "lidar": lidar_data.astype(np.float32),  # entrada
            "pose": gps_data.astype(np.float32),     # label (x, y, z) do robô
        }
        filename = f"sample_{self.sample_count:05d}.npz"
        filepath = os.path.join(self.dataset_dir, filename)
        np.savez_compressed(filepath, **sample)
        print(f"Saved {filepath}")
        self.sample_count += 1

    # ==========================================================
    # MAIN LOOP
    # ==========================================================
    def run(self):
        print("🎯 HSV + MLP READY — ALL KEYS ACTIVE")
        print("LiDAR→Pose dataset collection started. Press Q to stop.")

        while self.robot.step(self.time_step) != -1:
            if not self.handle_keyboard_input():
                break
            self.update_movement()
            self.update_auto_grab()

            # ====== LIDAR+GPS DATA COLLECTION ======
            if self.lidar and self.gps:
                self.lidar.enablePointCloud() 
                lidar_values = np.array(self.lidar.getRangeImage(), dtype=np.float32)

                MAX_RANGE = 5.5 #baseado no mapa

                # Substituir inf e valores inválidos
                lidar_values[np.isinf(lidar_values)] = MAX_RANGE
                lidar_values[np.isnan(lidar_values)] = MAX_RANGE
                lidar_values = np.clip(lidar_values, 0.05, MAX_RANGE)

                # Normalizar para [0,1]
                lidar_values = lidar_values / MAX_RANGE

                gps_values = np.array(self.gps.getValues())  # [x, y, z] = ground-truth
                self.save_sample(lidar_values, gps_values)

        self.base.move(0, 0, 0)
        print("Stopped")


if __name__ == "__main__":
    YouBotController().run()
