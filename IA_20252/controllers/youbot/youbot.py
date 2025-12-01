from controller import Robot
from base import Base
from arm import Arm
from gripper import Gripper
import random as rand
import math
import os

# ---- Tunable parameters ----
SAFE_DISTANCE = 0.40     # meters: minimum allowed distance in front sector
FRONT_SECTOR_RAYS = 40    # number of LIDAR rays on each side of front to check
TURN_STEPS = 70          # controller steps to spend rotating in AVOID state
FORWARD_STEPS = 130       # steps to keep chosen random direction
MAX_LINEAR_SPEED = 0.5  # vx/vy magnitude limit passed to base.move (m/s)
MAX_ANGULAR_SPEED = 0.4   # omega (rad/s) when rotating to avoid
CAMERA_SAVE_INTERVAL = 10  # steps between saving front-camera images
IMAGE_SAVE_QUALITY = 100
MIN_OBSTACLE_DISTANCE = 0.45  # meters: ignore readings closer than this (wheels/robot body)
SAVE_ALL_CAMERAS = True     
MAX_IMAGES_PER_SESSION = 20000  
# WE NEED TO DEFINE THE LABELS!
class YouBotController:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        
        self.base = Base(self.robot)
        self.arm = Arm(self.robot)
        self.gripper = Gripper(self.robot)

        # Cameras
        self.front_camera = self.robot.getDevice("front camera")
        self.front_camera.enable(self.time_step)

        self.left_camera = self.robot.getDevice("left camera")
        self.left_camera.enable(self.time_step)
        
        self.right_camera = self.robot.getDevice("right camera")
        self.right_camera.enable(self.time_step)

        self.back_camera = self.robot.getDevice("back camera")
        self.back_camera.enable(self.time_step)
        
        
        
        # bookkeeping for camera saves
        self.step_count = 0
        self.images_saved = 0
        os.makedirs("training_data/front", exist_ok=True)
        os.makedirs("training_data/left", exist_ok=True)
        os.makedirs("training_data/right", exist_ok=True)
        os.makedirs("training_data/back", exist_ok=True)
        
        # LIDAR
        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.time_step)
        self.lidar.enablePointCloud()

        # FSM state
        self.state = "RANDOM_MOVE"   # RANDOM_MOVE, AVOID
        self.state_timer = 0

        # movement command
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
        
        
    def _save_training_data(self):
            """Save comprehensive training data with labels"""
            if self.images_saved >= MAX_IMAGES_PER_SESSION:
                return
                
            timestamp = f"{self.step_count:06d}"
            
            # Save all camera views if enabled
            cameras = {
                "front": self.front_camera,
                "left": self.left_camera, 
                "right": self.right_camera,
                "back": self.back_camera
            }
            
            saved_any = False
            for cam_name, camera in cameras.items():
                image_data = camera.getImage()
                if image_data:
                    filename = f"training_data/{cam_name}/img_{timestamp}.png"
                    if camera.saveImage(filename, IMAGE_SAVE_QUALITY):
                        saved_any = True
            
            self.images_saved += 1
        
    def lidar_sectors(self, ranges):
        n = len(ranges)
        if n == 0:
            return float('inf'), float('inf'), float('inf')

        # YouBot LIDAR: index 0 = front, last index = also front

        # Sector widths
        SIDE = 50
        CENTER = 30

        left = ranges[:SIDE]
        right = ranges[-SIDE:]
        center_left = ranges[SIDE:SIDE + CENTER]
        center_right = ranges[n - SIDE - CENTER:n - SIDE]

        # combine center zone symmetrically
        center = list(center_left) + list(center_right)

        filter_valid = lambda arr: [d for d in arr if d > 0]

        left_min = min(filter_valid(left)) if filter_valid(left) else float('inf')
        center_min = min(filter_valid(center)) if filter_valid(center) else float('inf')
        right_min = min(filter_valid(right)) if filter_valid(right) else float('inf')

        return left_min, center_min, right_min

    def _pick_random_direction(self, max_speed=MAX_LINEAR_SPEED):
        """Return (vx, vy) random vector with magnitude in [0.1*max, max]"""
        angle = rand.uniform(0, 2 * math.pi)
        mag = rand.uniform(0.1 * max_speed, max_speed)
        return mag * math.cos(angle), mag * math.sin(angle)

    def _front_min_distance(self, ranges):
        """Compute minimum distance in the front wedge from LIDAR ranges"""
        # Webots LIDAR: index 0 ~ front, indices increase clockwise; use left and right slices
        n = len(ranges)
        if n == 0:
            return float('inf')
        left = ranges[:FRONT_SECTOR_RAYS]
        right = ranges[-FRONT_SECTOR_RAYS:]
        front = list(left) + list(right)
        # filter out invalid non-positive readings
        valid = [d for d in front if d is not None and d > 0.0]
        return min(valid) if valid else float('inf')

    def run(self):
        # initial random direction
        self.vx, self.vy = self._pick_random_direction()
        self.state_timer = FORWARD_STEPS

        while self.robot.step(self.time_step) != -1:
            self.step_count += 1

           
            raw_ranges = self.lidar.getRangeImage()
            # Filter out wheel detections
            ranges = [r if r > MIN_OBSTACLE_DISTANCE else float('inf') for r in raw_ranges]
            
            # Debug: show actual obstacle detections (not wheels)
            obstacle_ranges = [i for i in ranges if i != float('inf')]
            print(f"Real obstacles detected: {len(obstacle_ranges)} readings")
            if obstacle_ranges:
                print(f"Closest obstacle: {min(obstacle_ranges):.2f}m")

            #wheels_range = [i for i in ranges if i!=float('inf')] #approx 0.4m
            dmin_front = self._front_min_distance(ranges)

            # Save front-camera image periodically
            if self.step_count % CAMERA_SAVE_INTERVAL == 0:
                image_data = self.front_camera.getImage()
                if image_data:
                    filename = f"camera_outputs/front_{self.step_count}.png"
                    # saveImage returns True/False depending on success
                    self.front_camera.saveImage(filename, IMAGE_SAVE_QUALITY)

            left_d, center_d, right_d = self.lidar_sectors(ranges)

        # Decide navigation
            if self.state == "RANDOM_MOVE":

                # If the CENTER is blocked → avoid
                if center_d < SAFE_DISTANCE:
                    self.state = "AVOID"
                    self.state_timer = TURN_STEPS

                    # Choose turn direction based on which side is more free
                    if left_d > right_d:
                        # turn left
                        self.omega = MAX_ANGULAR_SPEED
                    else:
                        # turn right
                        self.omega = -MAX_ANGULAR_SPEED

                    self.base.move(0.0, 0.0, self.omega)
                    continue

                # If only side sectors see boxes → go straight → pass between them
                if self.state_timer <= 0:
                    self.vx, self.vy = self._pick_random_direction()
                    self.state_timer = FORWARD_STEPS

                self.base.move(self.vx, self.vy, 0.0)
                self.state_timer -= 1

            elif self.state == "AVOID":
                self.base.move(0, 0, self.omega)
                self.state_timer -= 1

                if self.state_timer <= 0:
                    self.vx, self.vy = self._pick_random_direction()
                    self.state = "RANDOM_MOVE"
                    self.state_timer = FORWARD_STEPS
                    self.base.move(self.vx, self.vy, 0)


            # Loop end: continue stepping
            # Save training data
            if self.step_count % CAMERA_SAVE_INTERVAL == 0:
                self._save_training_data()
if __name__ == "__main__":
    controller = YouBotController()
    controller.run()
