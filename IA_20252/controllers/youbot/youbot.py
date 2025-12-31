#!/usr/bin/env python
import math
from controller import Robot, Keyboard
from base import Base
from arm import Arm
from gripper import Gripper
import numpy as np
from pathlib import Path

import sys
from services import (
    BlockHandler,
    SensorSuite,
    MovementController,
    AngleController,
    ObstacleAvoider,
    AlignmentController,
    ObjectDetector,
    LidarGpsController,
    ColorClassifier,
    FuzzySimple,
)
from dataclasses import dataclass

@dataclass
class YouBotConfig:
    pick_distance: float = 0.170
    pick_tolerance: float = 0.005
    min_approach_speed: float = 0.03
    align_deadzone: float = 0.005
    obstacle_min_dist: float = 0.25
    cube_height: float = 0.03
    distance_diff_thresh: float = 0.01
    min_approach_distance:float = 0.30
    v_max:float = 0.15

class YouBotController:
    def __init__(self):
        # core environment
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.dt = self.time_step / 1000.0
        self.DEBUG = False

        # central config and minimal runtime flags
        self.config = YouBotConfig()
        self.is_picking = False
        self.gripper_descended = False
        self.initial_angle = 0.0
        self.rotating = False
        self.target_angle = None

        # grouped initialization
        self._init_devices()
        self._init_components()
        self._init_controllers()

    # --- compatibility properties (no duplicated state) ---
    @property
    def PICK_DISTANCE(self):
        return self.config.pick_distance

    @property
    def PICK_TOL(self):
        return self.config.pick_tolerance

    @property
    def MIN_APPROACH_SPEED(self):
        return self.config.min_approach_speed

    @property
    def ALIGN_DEADZONE(self):
        return self.config.align_deadzone

    @property
    def OBSTACLE_MIN_DIST(self):
        return self.config.obstacle_min_dist

    # --- helpers used in __init__ ---
    def _init_devices(self):
        # camera & keyboard
        self.camera = self.robot.getDevice("camera")
        if self.camera: self.camera.enable(self.time_step)
        self.keyboard = self.robot.getKeyboard()
        if self.keyboard: self.keyboard.enable(self.time_step)

        # lidars & compass
        self.lidar_low = self.robot.getDevice("lidar_horizontal")
        self.lidar_high = self.robot.getDevice("lidar_horizontal_2")
        self.lidar_global = self.robot.getDevice("lidar")
        self.compass = self.robot.getDevice("compass")
         
        self.lidar_right = self.robot.getDevice("lidar right")
        self.lidar_left = self.robot.getDevice("lidar left")
        missing = [n for n, d in (
            ("lidar_horizontal", self.lidar_low),
            ("lidar_horizontal_2", self.lidar_high),
            ("lidar", self.lidar_global),
            ("compass", self.compass),
            ("lidar left", self.lidar_left),
            ("lidar right", self.lidar_right)
        ) if d is None]
        if missing:
            print(f"ERRO: dispositivos faltando: {', '.join(missing)}", file=sys.stderr)
            sys.exit(1)

        # enable sensors
        self.lidar_low.enable(self.time_step)
        self.lidar_high.enable(self.time_step)
        self.lidar_global.enable(self.time_step)
        self.compass.enable(self.time_step)
        self.lidar_left.enable(self.time_step)   
        self.lidar_right.enable(self.time_step)
    def _init_components(self):
        # hardware wrappers and logic components
        self.fuzzy = FuzzySimple(v_max = self.config.v_max)

        self.base = Base(self.robot)
        self.arm = Arm(self.robot)
        self.gripper = Gripper(self.robot)
        
        self.sensors = SensorSuite(
            self.lidar_low,
            self.lidar_high,
            self.compass,
            step_wait=self._step_wait,
            lidar_high_left=self.lidar_right,  
            lidar_high_right=self.lidar_left   
        )
        self.movement = MovementController(
            self.base,
            step_wait=self._step_wait,
            forward_speed=0.1,
            strafe_speed=0.1,
            movement_duration=10,
            fuzzy=self.fuzzy,
            front_dist_fn=self.sensors.read_low_filtered,
            side_dist_fn=self.sensors.read_side_distances,  
        )
        self.lidar_gps = LidarGpsController(
            self.lidar_global,
            model_path=Path("./models/lidar_gps_best_4.pth"),
            T=2,
            max_range=5.5,
            debug=self.DEBUG,
        )

    def _init_controllers(self):

        self.aligner = AlignmentController(
            self.base, self.sensors,
            kp=1.5, max_vy=0.06, deadzone=self.config.align_deadzone,
            forward_speed=self.movement.forward_speed,
            pick_distance=self.config.pick_distance,
            pick_tol=self.config.pick_tolerance,
            min_approach_speed=self.config.min_approach_speed,
            debug=self.DEBUG,
            
        )
        self.avoider = ObstacleAvoider(self.base, self.lidar_low, self.lidar_high, self.sensors, step_wait=self._step_wait, dt=self.dt, obstacle_min_dist=self.config.obstacle_min_dist, debug=self.DEBUG)
        self.detector = ObjectDetector(self.sensors,
                                       distance_diff_thresh=self.config.distance_diff_thresh,
                                       obstacle_min_dist=self.config.obstacle_min_dist,
                                       step_wait=self._step_wait,
                                       debug=self.DEBUG)
        self.block_handler = BlockHandler(self)
        self.color_classifier = ColorClassifier(self, "./models/mlp_ab_model.joblib")
    # ================= KEYBOARD =================
    def handle_keyboard_input(self):
        key = self.keyboard.getKey()
        while key >= 0:
            if key in (ord('W'), ord('w')):
                self.movement.forward()
            elif key in (ord('S'), ord('s')):
                self.movement.backward()
            elif key in (ord('A'), ord('a')):
                self.movement.strafe_left()
            elif key in (ord('D'), ord('d')):
                self.movement.strafe_right()
            elif key == ord(' '):
                self.movement.stop_all()
            elif key in (ord('Q'), ord('q')):
                return False
            elif key == ord('J'):  # Gira 90° à esquerda
                self.rotate_right_90()
            elif key == ord('L'):  # Gira 90° à direita
                self.rotate_left_90()
            elif key == ord('K'):  # Volta à posição original
                self.rotate_to_initial()
            key = self.keyboard.getKey()
        return True

    def wait(self, steps=30):
        for _ in range(steps):
            self.robot.step(self.time_step)
            
    # ================= MOVIMENTO =================
    def update_movement(self):
        # proxy picking/block state into movement controller and delegate
        self.movement.is_picking = getattr(self, "picker", None) and self.picker.is_picking
        self.movement.block_forward = getattr(self, "block_forward", False)
        self.movement.update()

    # ================= LIDAR READ =================
    def read_lidars(self):
        return self.sensors.read_lidars()
    # ================= COMPASS / ROTAÇÃO =================
    def get_current_angle(self):
        north = self.compass.getValues()
        angle_rad = math.atan2(north[0], north[1])
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        return angle_deg

    def angle_diff(self, target, current):
        """Diferença mínima entre dois ângulos (±180°)."""
        d = (target - current + 180) % 360 - 180
        return d

    def start_rotation(self, target_angle):
        """Inicia a rotação não-bloqueante."""
        self.target_angle = target_angle % 360
        self.rotating = True


    def rotate_to_initial(self):
        """Gira até o ângulo inicial definido."""
        if self.rotating:
            return
        self.rotate_to_angle(self.initial_angle)

    def rotate_to_angle(self, target_angle):
        """Inicia a rotação em direção a target_angle (não-bloqueante)."""
        self.target_angle = target_angle % 360
        self.rotating = True
        self.rotation_started = False  

    def rotate_left_90(self):
        """Gira 90° para a esquerda a partir do ângulo atual."""
        if self.rotating:
            return
        current = self.get_current_angle()
        self.rotate_to_angle((current - 90) % 360)

    def rotate_right_90(self):
        """Gira 90° para a direita a partir do ângulo atual."""
        if self.rotating:
            return
        current = self.get_current_angle()
        self.rotate_to_angle((current + 90) % 360)

    def update_rotation(self):
        if not self.rotating:
            return

        current_angle = self.get_current_angle()
        diff = self.angle_diff(self.target_angle, current_angle)

        # params
        deadzone = 0.5      
        max_speed = 1.0     
        min_speed = 0.08   
        slow_zone = 15.0    
        #print(current_angle)
        # stop if within deadzone
        if abs(diff) <= deadzone:
            self.base.move(0, 0, 0)
            self.rotating = False
            return

        # speed scaled with error magnitude (smooth, proportional-ish)
        if abs(diff) >= slow_zone:
            speed = max_speed
        else:
            frac = abs(diff) / slow_zone
            speed = min_speed + (max_speed - min_speed) * frac
            speed = max(min_speed, min(speed, max_speed))

        # choose rotation sign based on error each tick (avoid stale direction)
        direction = -1 if diff > 0 else 1
        angular = direction * speed

        self.base.move(0, 0, angular)





    # ================= DETECT (INTACTA) =================
    def detect_objects(self):
        # Reset flags (mantém comportamento anterior)
        if not self.is_picking:
            if self.gripper_descended:
                if self.DEBUG: print("resetando gripper_descended")
            self.gripper_descended = False
            if hasattr(self, "picker"):
                self.picker.gripper_descended = False

        self.cube_detected_a_frente = False
        self.obstacle_detected = False
        self.obstacle_blocking_cube = False

        res = self.detector.detect()
        if not res:
            return

        self.cube_detected_a_frente = bool(res["cube_detected_a_frente"])
        self.obstacle_detected = bool(res["obstacle_detected"])
        self.obstacle_blocking_cube = bool(res["obstacle_blocking_cube"])

    # ================= AVOID OBJECT ================================
    def avoid_obstacle(self):
            self.move_forward_counter = 0
            self.move_backward_counter = 0
            self.block_forward = True
            self.base.move(0, 0, 0)
            self._step_wait(0.03)
            freed = self.avoider.start_avoid()
            if freed:
                self.obstacle_detected = False
         
            self.block_forward = False
            
    # ================= DESCEND GRIPPER (REESCRITA) =================
    def descend_gripper_if_target_distance(self):
        if self.is_picking or not self.cube_detected_a_frente:
            self.gripper_descended = False
            return

        min_low, _ = self.read_lidars()
        if not np.isfinite(min_low):
            self.gripper_descended = False
            return

        if abs(min_low - self.PICK_DISTANCE) <= self.PICK_TOL:
            self.is_picking = True
            try:
                self.picker.do_pick_blocking()
                self.gripper_descended = True
            finally:
                self.is_picking = False
        else:
            self.gripper_descended = False
            
   
    # ================= UTILS =================
    def _step_wait(self, seconds):
        steps = max(1, int(seconds / self.dt))
        for _ in range(steps):
            if self.robot.step(self.time_step) == -1:
                break

    # ================= LOOP =================
    def run(self):
        print("=== YouBot | Garra desce a 0.104 m | Com alinhamento lateral automático ===")
        self.robot.step(self.time_step)
        self.lidar_gps.init_after_first_step()
        
        while self.robot.step(self.time_step) != -1:
            # inside main loop
            pose = self.lidar_gps.step()
            
            if pose is not None:
                self.lidar_pose = pose
                if not  self.DEBUG:
                    print(f"LidarGPS pose: x={pose[0]:.3f}, y={pose[1]:.3f}")
            if not self.handle_keyboard_input():
                break
            self.detect_objects()
            # Prioriza pegar se houver cubo acessível e NÃO estiver bloqueado
            dist = self.sensors.read_low_filtered()
            if self.cube_detected_a_frente and  dist <= self.config.min_approach_distance:
                aligned = self.aligner.align_with_cube()
                if aligned:
                    reached = self.aligner.auto_approach_cube()
                    if reached:
                        label = self.color_classifier.capture_and_classify()
                        self.block_handler.pick(label) 
                        
            # Senão, se houver obstáculo detectado, trata de evitar
            elif self.obstacle_detected:
             
                self.avoid_obstacle()
            else:
              
                self.update_movement()
                if not self.DEBUG:
                    print("SIDE:", self.sensors.read_side_distances())
            self.update_rotation()

        self.base.move(0, 0, 0)
        print("Controller finalizado")

if __name__ == "__main__":
    YouBotController().run()
