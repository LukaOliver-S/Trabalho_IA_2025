from controller import Robot, Keyboard
from base import Base
from arm import Arm
from gripper import Gripper

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

        # CNN status
        self.cnn_attempted = False
        self.cnn_available = False
        
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
        print("Press 'C' to test CNN detection")

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

    def try_load_cnn_libraries(self):
        """Try to load CNN libraries only when requested"""
        if self.cnn_attempted:
            return self.cnn_available
            
        self.cnn_attempted = True
        print("Loading CNN libraries...")
        
        try:
            import numpy as np
            import torch
            from PIL import Image
            from transformers import MobileNetV2ForImageClassification
            from torchvision import transforms
            
            self.cnn_available = True
            self.setup_model(torch, transforms, MobileNetV2ForImageClassification)
            print("CNN loaded successfully")
            return True
            
        except Exception as e:
            print(f"CNN loading failed: {e}")
            self.cnn_available = False
            return False

    def setup_model(self, torch, transforms, MobileNetV2):
        """Setup the CNN model"""
        try:
            import os
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            model_path = os.path.join(os.path.dirname(__file__), "../Model_Processing/results/best_model_fold_0.pth")
            
            self.model = MobileNetV2.from_pretrained("google/mobilenet_v2_1.0_224", num_labels=3, ignore_mismatched_sizes=True)
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                self.model.to(self.device)
                self.model.eval()
                print("Model loaded successfully")
            else:
                print("Model file not found")
                
        except Exception as e:
            print(f"Model setup failed: {e}")

    def capture_image(self):
        """Capture image from active camera"""
        if not self.active_camera:
            return None
            
        try:
            import numpy as np
            from PIL import Image
            
            image_data = self.active_camera.getImage()
            if image_data is None:
                return None
                
            width = self.active_camera.getWidth()
            height = self.active_camera.getHeight()
            
            # Convert BGRA to RGB
            image_array = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))
            rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
            rgb_array[:, :, 0] = image_array[:, :, 2]
            rgb_array[:, :, 1] = image_array[:, :, 1]
            rgb_array[:, :, 2] = image_array[:, :, 0]
            
            return Image.fromarray(rgb_array)
            
        except Exception as e:
            print(f"Image capture failed: {e}")
            return None

    def predict_color(self, image):
        """Predict color using CNN"""
        if not self.cnn_available or not hasattr(self, 'model') or image is None:
            return None, 0.0
            
        try:
            import torch
            
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
            classes = ['red', 'green', 'blue']
            return classes[predicted_class], confidence
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            return None, 0.0

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
            
            # Camera and CNN controls
            elif key == ord('V') or key == ord('v'):
                self.switch_camera()
            elif key == ord('T') or key == ord('t'):
                print("Setting up GRIPPER CAMERA test...")
                self.position_for_gripper_camera_test()
            elif key == ord('C') or key == ord('c'):
                print(f"Testing CNN with {self.camera_mode} camera...")
                if self.try_load_cnn_libraries():
                    self.test_cnn()
                else:
                    print("CNN not available - taking simple photo")
                    self.take_test_photo()
            
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
        """Take a simple photo without CNN"""
        print(f"Taking photo with {self.camera_mode} camera...")
        
        if self.active_camera:
            print("Photo captured (check Webots camera overlay)")
            if self.camera_mode == "gripper":
                print("Gripper camera should see close-up objects clearly!")
        else:
            print("No active camera")

    def test_cnn(self):
        """Test CNN on current view"""
        if not self.cnn_available:
            print("CNN not available")
            return
            
        print(f"Capturing from {self.camera_mode} camera...")
        image = self.capture_image()
        
        if image:
            predicted_color, confidence = self.predict_color(image)
            if predicted_color:
                print(f"✅ Detected: {predicted_color}, Confidence: {confidence:.2f}")
                print(f"📷 Camera: {self.camera_mode} (close-up detection)")
                
                if confidence > 0.8:
                    try:
                        import os
                        save_path = os.path.join(os.path.dirname(__file__), f"{self.camera_mode}_{predicted_color}_{confidence:.2f}.png")
                        image.save(save_path)
                        print(f"💾 Saved: {save_path}")
                    except Exception as e:
                        print(f"Save failed: {e}")
            else:
                print("❌ Could not predict color")
        else:
            print("❌ Failed to capture image")

    def update_auto_grab(self):
        """Update automatic grabbing sequence"""
        if self.auto_grab_mode and self.grab_sequence_step >= 0:
            wait_time = self.grab_cube_sequence()

    def run(self):
        """Main control loop"""
        print("YouBot Gripper Camera Controller")
        print("Movement: W/S/A/D=Move, Space=Stop")
        print("Arm Height: I/K=Up/Down")
        print("Arm Rotation: J/L=Rotate Left/Right")
        print("Gripper: O=Open, P=Close, G=Auto-Grab, R=Reset")
        print("Camera: V=Switch, T=Setup Gripper Cam Test, C=Test CNN")
        print("Q=Quit")
        print()
        print("🎯 GRIPPER CAMERA TEST:")
        print("1. Press 'T' to setup gripper camera test")
        print("2. Press 'C' to test close-up color detection")
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