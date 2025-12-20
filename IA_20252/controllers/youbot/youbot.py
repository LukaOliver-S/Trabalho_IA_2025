from controller import Robot, Keyboard
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

        # ================= HSV MLP =================
        self.hsv_available = self.try_load_hsv_model()

        # ================= AUTO GRAB =================
        self.auto_grab_mode = False
        self.grab_sequence_step = 0

    # ==========================================================
    # LOAD MLP
    # ==========================================================
    def try_load_hsv_model(self):
        model_path = Path(r"C:\Users\User\Documents\IA_PROJECT\Archives\files_webot\IA_20252\controllers\Model_Processing\hsv_mlp_model.pkl")


        if not model_path.exists():
            print("❌ hsv_mlp_model.pkl not found")
            return False

        data = joblib.load(model_path)
        self.hsv_model = data["model"]
        self.hsv_scaler = data["scaler"]
        self.hsv_label_encoder = data["label_encoder"]

        print("✅ HSV MLP model loaded")
        return True

    # ==========================================================
    # CAMERA SWITCH
    # ==========================================================
    def switch_camera(self):
        cams = []
        if self.gripper_camera:
            cams.append(("gripper", self.gripper_camera))
        if self.front_camera:
            cams.append(("front", self.front_camera))
        if self.detection_camera:
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
    # HSV FEATURES (IGUAL AO TREINO)
    # ==========================================================
    def extract_hsv_features_from_bgr(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        h = hsv[:, :, 0].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)

        mask = (s > 60) & (v > 40)
        if np.count_nonzero(mask) < 50:
            return None

        h, s, v = h[mask], s[mask], v[mask]

        s_mean, s_std = np.mean(s) / 255, np.std(s) / 255
        v_mean, v_std = np.mean(v) / 255, np.std(v) / 255

        r = np.mean((h <= 8) | (h >= 172))
        g = np.mean((h >= 40) & (h <= 80))
        b = np.mean((h >= 100) & (h <= 130))

        ratios = np.array([r, g, b])
        ratios /= (np.sum(ratios) + 1e-6)

        return np.array([*ratios, s_mean, s_std, v_mean, v_std], dtype=np.float32)

    def capture_hsv_features(self):
        img = self.active_camera.getImage()
        if img is None:
            return None

        w, h = self.active_camera.getWidth(), self.active_camera.getHeight()
        frame = np.frombuffer(img, np.uint8).reshape((h, w, 4))
        return self.extract_hsv_features_from_bgr(frame[:, :, :3])

    # ==========================================================
    # MLP DECISION
    # ==========================================================
    def decide_color(self, features):
        X = self.hsv_scaler.transform(features.reshape(1, -1))
        pred = self.hsv_model.predict(X)[0]
        return self.hsv_label_encoder.inverse_transform([pred])[0]

    def test_hsv(self):
        if not self.hsv_available:
            print("❌ MLP unavailable")
            return

        feat = self.capture_hsv_features()
        if feat is None:
            print("❌ No valid HSV data")
            return

        color = self.decide_color(feat)
        print(f"\n🤖 HSV + MLP → {color.upper()}")

    # ==========================================================
    # KEYBOARD (TODOS MANTIDOS)
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
    # MAIN LOOP
    # ==========================================================
    def run(self):
        print("🎯 HSV + MLP READY — ALL KEYS ACTIVE")

        while self.robot.step(self.time_step) != -1:
            if not self.handle_keyboard_input():
                break
            self.update_movement()
            self.update_auto_grab()

        self.base.move(0, 0, 0)
        print("Stopped")


if __name__ == "__main__":
    YouBotController().run()
