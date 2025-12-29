from controller import Robot, Keyboard
from base import Base
from arm import Arm
from gripper import Gripper
import numpy as np
import cv2
import joblib
from scipy.stats import circmean

            
class YouBotController:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        
        # Initialize robot components
        self.base = Base(self.robot)
        self.arm = Arm(self.robot)
        self.gripper = Gripper(self.robot)

        # Initialize keyboard
        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.time_step)

        # Initialize cameras - prioritize gripper camera
        self.gripper_camera = self.robot.getDevice("gripper_camera")
        self.front_camera = self.robot.getDevice("front camera")
        self.detection_camera = self.robot.getDevice("detection_camera")

        # Enable cameras
        if self.gripper_camera:
            self.gripper_camera.enable(self.time_step)
            print("Gripper camera enabled (512x512)")
            
        if self.front_camera:
            self.front_camera.enable(self.time_step)
            print("Front camera enabled")
            
        if self.detection_camera:
            self.detection_camera.enable(self.time_step)
            print("Detection camera enabled")

        # Start with gripper camera for close-up testing
        self.active_camera = self.gripper_camera if self.gripper_camera else self.front_camera
        self.camera_mode = "gripper"  # "gripper", "front", "detection"

        # Movement parameters
        self.forward_speed = 0.3
        self.strafe_speed = 0.3
        self.movement_duration = 10

        # Movement counters
        self.move_forward_counter = 0
        self.move_backward_counter = 0
        self.strafe_left_counter = 0
        self.strafe_right_counter = 0

        # HSV Model status
        self.hsv_attempted = False
        self.hsv_available = False
        
        # Gripping sequence state
        self.auto_grab_mode = False
        self.grab_sequence_step = 0

    def switch_camera(self):
        """Cycle through available cameras"""
        cameras = []
        if self.gripper_camera:
            cameras.append(("gripper", self.gripper_camera))
        if self.front_camera:
            cameras.append(("front", self.front_camera))
        if self.detection_camera:
            cameras.append(("detection", self.detection_camera))
            
        if len(cameras) == 0:
            print("No cameras available")
            return
            
        # Find current camera index
        current_idx = 0
        for i, (name, cam) in enumerate(cameras):
            if name == self.camera_mode:
                current_idx = i
                break
                
        # Switch to next camera
        next_idx = (current_idx + 1) % len(cameras)
        self.camera_mode, self.active_camera = cameras[next_idx]
        print(f"Switched to {self.camera_mode} camera")

    def position_for_gripper_camera_test(self):
        """Position robot optimally for gripper camera testing"""
        print("Positioning for gripper camera close-up test...")
        
        # Switch to gripper camera
        if self.gripper_camera:
            self.active_camera = self.gripper_camera
            self.camera_mode = "gripper"
            print("Using gripper camera for close-up detection")
        
        # Reset arm position
        self.arm.set_height(self.arm.RESET)
        self.arm.set_orientation(self.arm.FRONT)
        self.gripper.grip()
        
        print("Robot positioned for gripper camera test")
        print("The gripper_test_cube should be visible to gripper camera")
        print("Press 'H' to test HSV color detection")

    def grab_cube_sequence(self):
        """Automated cube grabbing sequence"""
        if self.grab_sequence_step == 0:
            print("Step 1: Setting arm to front floor position...")
            self.arm.set_height(self.arm.FRONT_FLOOR)
            self.arm.set_orientation(self.arm.FRONT)
            self.grab_sequence_step = 1
            return 30
            
        elif self.grab_sequence_step == 1:
            print("Step 2: Opening gripper...")
            self.gripper.release()
            self.grab_sequence_step = 2
            return 20
            
        elif self.grab_sequence_step == 2:
            print("Step 3: Moving forward to cube...")
            self.move_forward_counter = 15
            self.grab_sequence_step = 3
            return 20
            
        elif self.grab_sequence_step == 3:
            print("Step 4: Closing gripper to grab cube...")
            self.gripper.grip()
            self.grab_sequence_step = 4
            return 30
            
        elif self.grab_sequence_step == 4:
            print("Step 5: Lifting cube...")
            self.arm.set_height(self.arm.FRONT_CARDBOARD_BOX)
            self.grab_sequence_step = 5
            return 30
            
        else:
            print("Grab sequence complete!")
            self.auto_grab_mode = False
            self.grab_sequence_step = 0
            return 0
  
    def extract_hsv_features_from_bgr(self, bgr_image):
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].astype(np.float32)

        red_ratio   = np.mean((h <= 10) | (h >= 170))
        green_ratio = np.mean((h >= 35) & (h <= 85))
        blue_ratio  = np.mean((h >= 90) & (h <= 130))

        return red_ratio, green_ratio, blue_ratio

   

    def capture_hsv_features(self):
        if not self.active_camera:
            return None

        image_data = self.active_camera.getImage()
        if image_data is None:
            return None

        width = self.active_camera.getWidth()
        height = self.active_camera.getHeight()

        img = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))
        bgr = img[:, :, :3]

        return self.extract_hsv_features_from_bgr(bgr)



        
    def classify_color(self, r, g, b):
        if r > g and r > b:
            return "red"
        elif g > r and g > b:
            return "green"
        elif b > r and b > g:
            return "blue"
        return "unknown"

    def test_hsv(self):
        ratios = self.capture_hsv_features()
        if ratios is None:
            print("❌ No image")
            return

        r, g, b = ratios
        color = self.classify_color(r, g, b)

        print("\n🔎 HSV DOMINANCE")
        print(f"Red:   {r:.3f}")
        print(f"Green: {g:.3f}")
        print(f"Blue:  {b:.3f}")
        print(f"✅ COLOR = {color.upper()}")

    def handle_keyboard_input(self):
        """Handle keyboard input"""
        key = self.keyboard.getKey()
        
        while key >= 0:
            # Movement controls
            if key == ord('W') or key == ord('w'):
                self.move_forward_counter = self.movement_duration
            elif key == ord('S') or key == ord('s'):
                self.move_backward_counter = self.movement_duration
            elif key == ord('A') or key == ord('a'):
                self.strafe_left_counter = self.movement_duration
            elif key == ord('D') or key == ord('d'):
                self.strafe_right_counter = self.movement_duration
            
            # Arm height controls
            elif key == ord('I') or key == ord('i'):
                print("Arm up")
                self.arm.increase_height()
            elif key == ord('K') or key == ord('k'):
                print("Arm down")
                self.arm.decrease_height()
                
            # Arm orientation controls
            elif key == ord('J') or key == ord('j'):
                print("Arm rotate left")
                self.arm.increase_orientation()
            elif key == ord('L') or key == ord('l'):
                print("Arm rotate right")
                self.arm.decrease_orientation()
            
            # Gripper controls
            elif key == ord('O') or key == ord('o'):
                print("Gripper OPEN (release)")
                self.gripper.release()
            elif key == ord('P') or key == ord('p'):
                print("Gripper CLOSE (grip)")
                self.gripper.grip()
            elif key == ord('G') or key == ord('g'):
                print("Starting AUTO GRAB sequence...")
                self.auto_grab_mode = True
                self.grab_sequence_step = 0
            elif key == ord('R') or key == ord('r'):
                print("RESET arm to safe position")
                self.arm.set_height(self.arm.RESET)
                self.arm.set_orientation(self.arm.FRONT)
                self.gripper.release()
            
            # Camera and HSV controls
            elif key == ord('V') or key == ord('v'):
                self.switch_camera()
            elif key == ord('T') or key == ord('t'):
                print("Setting up GRIPPER CAMERA test...")
                self.position_for_gripper_camera_test()
            elif key == ord('H') or key == ord('h'):
                print(f"Testing HSV with {self.camera_mode} camera...")
                self.test_hsv()
              
            
            # General controls
            elif key == ord('Q') or key == ord('q'):
                return False
            elif key == ord(' '):
                self.move_forward_counter = 0
                self.move_backward_counter = 0
                self.strafe_left_counter = 0
                self.strafe_right_counter = 0
                print("Movement stopped")
                
            key = self.keyboard.getKey()
        
        return True

    def update_movement(self):
        """Update robot movement"""
        vx = vy = omega = 0.0
        
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
        
        self.base.move(vx, vy, omega)

    def take_test_photo(self):
        """Take a simple photo without HSV model"""
        print(f"Taking photo with {self.camera_mode} camera...")
        
        if self.active_camera:
            print("Photo captured (check Webots camera overlay)")
            if self.camera_mode == "gripper":
                print("Gripper camera should see close-up objects clearly!")
        else:
            print("No active camera")

  

    def update_auto_grab(self):
        """Update automatic grabbing sequence"""
        if self.auto_grab_mode and self.grab_sequence_step >= 0:
            wait_time = self.grab_cube_sequence()

    def run(self):
        """Main control loop"""
        print("YouBot HSV Color Detection Controller")
        print("Movement: W/S/A/D=Move, Space=Stop")
        print("Arm Height: I/K=Up/Down")
        print("Arm Rotation: J/L=Rotate Left/Right")
        print("Gripper: O=Open, P=Close, G=Auto-Grab, R=Reset")
        print("Camera: V=Switch, T=Setup Gripper Cam Test, H=Test HSV Color")
        print("Q=Quit")
        print()
        print("🎯 HSV COLOR DETECTION TEST:")
        print("1. Press 'T' to setup gripper camera test")
        print("2. Press 'H' to test HSV color detection")
        print(f"📷 Current camera: {self.camera_mode}")
        
        try:
            while self.robot.step(self.time_step) != -1:
                if not self.handle_keyboard_input():
                    break
                self.update_movement()
                self.update_auto_grab()
                        
        except KeyboardInterrupt:
            pass
        finally:
            self.base.move(0, 0, 0)
            print("Stopped")

if __name__ == "__main__":
    controller = YouBotController()
    controller.run()