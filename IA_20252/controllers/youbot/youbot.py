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
    NavigationController,
    MissionController,
    AngleController,
    ObstacleAvoider,
    AlignmentController,
    ObjectDetector,
    LidarGpsController,
    ColorClassifier,
    FuzzySimple
)

class YouBotController:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.dt = self.time_step / 1000.0
        self.DEBUG = True
        # ================= COMPONENTES =================
        self.base = Base(self.robot)
        self.arm = Arm(self.robot)
        self.gripper = Gripper(self.robot)
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.time_step)

        # ================= KEYBOARD =================
        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.time_step)

        # ================= LiDARs =================
        self.lidar_low = self.robot.getDevice("lidar_horizontal")
        self.lidar_high = self.robot.getDevice("lidar_horizontal_2")
        self.lidar_global = self.robot.getDevice("lidar")
        if self.lidar_low is None or self.lidar_high is None or self.lidar_global is None:
            print("❌ ERRO: LiDARs não encontrados", file=sys.stderr)
            sys.exit(1)

        self.lidar_low.enable(self.time_step)
        self.lidar_high.enable(self.time_step)
        self.lidar_global.enable(self.time_step)   
        # ================= LIDAR_GPS ===============

        self.lidar_gps = LidarGpsController(
            self.lidar_global,
            model_path=Path("./models/lidar_gps_yaw_2.pth"),
            T=2,
            max_range=5.5,
            debug=self.DEBUG,
            yaw_provider=self._compass_yaw_vector,  # <--- provider conectado à bússola
        )

        self.lidar_pose = None
        # ================= COMPASS =================
        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.time_step)
        # ==================SENSORS====================
        self.sensors = SensorSuite(self.lidar_low, self.lidar_high, self.compass, step_wait=self._step_wait)
        # ================= MOVIMENTO =================
        self.movement = MovementController(self.base, step_wait=self._step_wait)
        self.forward_speed = 0.10
        self.strafe_speed = 0.15
        self.movement_duration = 10
        self.move_forward_counter = 0
        self.move_backward_counter = 0
        self.strafe_left_counter = 0
        self.strafe_right_counter = 0
        # ================= angulação =================
        self.angle_controller = AngleController(self.compass, self.base, self.dt) # Por algum motivo não funciona
        
        
        # fuzzy velocity selector 
        self.fuzzy = FuzzySimple(v_max=0.10)
        # ================= PARÂMETROS =================
        self.CUBE_HEIGHT = 0.03
        self.DISTANCE_DIFF_THRESH = 0.01
        # ================= PICK =================
        self.PICK_DISTANCE = 0.170
        self.PICK_TOL = 0.005
        self.MIN_APPROACH_SPEED = 0.03  # velocidade mínima para aproximação
        self.ALIGN_DEADZONE = 0.005     # tolerância lateral para considerar "centralizado"
        #================== GRIPPER CONTROLLER ========
        self.gripper_descended = False
        self.is_picking = False   
        self.initial_angle = 0.0
        self.rotating = False
        self.target_angle = None
        self.Cube_min_dist = 0.25
        #================= Align Controller ==========
        self.aligner = AlignmentController(
            self.base, self.sensors,
            kp=0.3, max_vy=0.1, deadzone=self.ALIGN_DEADZONE,
            forward_speed=self.forward_speed,
            pick_distance=self.PICK_DISTANCE,
            pick_tol=self.PICK_TOL,
            min_approach_speed=self.MIN_APPROACH_SPEED,
            debug=self.DEBUG,
            fuzzy=self.fuzzy, 
        )
   
    # ================= AVOID ====================
        self.OBSTACLE_MIN_DIST = 0.30   # espaço mínimo atrás do cubo para considerar "acessível"
        self.obstacle_detected = False
        self.obstacle_blocking_cube = False
        self.avoiding = False
        self.avoider = ObstacleAvoider(self.base, self.lidar_low, self.lidar_high, self.sensors, step_wait=self._step_wait, dt=self.dt, debug=self.DEBUG,    obstacle_min_dist=self.OBSTACLE_MIN_DIST,  # <= pass min-dist
)

     # ================== OBJECT DETECTOR==========
        self.detector = ObjectDetector(self.sensors,
                               distance_diff_thresh=self.DISTANCE_DIFF_THRESH,
                               obstacle_min_dist=self.OBSTACLE_MIN_DIST,
                               step_wait=self._step_wait,
                               debug=self.DEBUG)
    
        # ================= BLOCK HANDLER & COLOR CLASSIFIER ==========
        self.block_handler = BlockHandler(self)
        self.color_classifier = ColorClassifier(self, "./models/mlp_ab_model.joblib")
        
        # ================= NAVIGATION =================
        self.navigator = NavigationController(
            self,
          
        )
        self.mission_controller = MissionController(self.navigator, self)
        
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
        self.rotation_started = False  # reinicia a inicialização da direção

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

        deadzone = 0.2
        max_speed = 1.0
        slow_zone = 15.0

        # Inicializa a direção apenas no início
        if not self.rotation_started:
            if diff == 0:
                self.rotation_direction = 0
            else:
               
                self.rotation_direction = -1 if diff > 0 else 1
            self.rotation_started = True
        print(current_angle)
        # Se dentro do deadzone, parar de girar
        if abs(diff) <= deadzone:
            self.base.move(0, 0, 0)
            self.rotating = False
            self.rotation_started = False
            return

        # Velocidade proporcional
        speed = max_speed
        if abs(diff) < slow_zone:
            speed *= 0.3

        # Gira na direção 
        self.base.move(0, 0, speed * self.rotation_direction)
    def _compass_yaw_vector(self):
        """
        Retorna (cos, sin) a partir da bússola.
        Usa as duas primeiras componentes (x,y) do vetor norte da bússola.
        """
        north = self.compass.getValues()
        x, y = float(north[0]), float(north[1])
        norm = math.hypot(x, y)
        if norm == 0 or not np.isfinite(norm):
            return (1.0, 0.0)  # fallback yaw=0
        cos_yaw = y / norm
        sin_yaw = x / norm
        return (cos_yaw, sin_yaw)




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
            # copia estados úteis do avoider para o controlador (opcional)
            self.avoid_cooldown = self.avoider.avoid_cooldown
            self.avoid_attempts = self.avoider.avoid_attempts
            self.last_avoid_side = self.avoider.last_avoid_side
            self.recently_freed = self.avoider.recently_freed
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
            
    # def run(self):
    #     print("=== YouBot | Missões com Coleta e Retorno ===")

    #     # ===== STEP INICIAL =====
    #     self.robot.step(self.time_step)
    #     self.lidar_gps.init_after_first_step()

    #     # ===== ESTADOS =====
    #     navigating = True
    #     collecting_cube = False
    #     returning = False

    #     saved_pose = None
    #     saved_mission_step = None
    #     saved_nav_state = None

    #     # ===== INICIA MISSÃO =====
    #     self.mission_controller.start()

    #     # ===== LOOP PRINCIPAL =====
    #     while self.robot.step(self.time_step) != -1:

    #         # ===== POSE =====
    #         pose = self.lidar_gps.step()
    #         if pose is not None:
    #             self.lidar_pose = pose
    #             if self.DEBUG:
    #                 print(f"📍 Pose: x={pose[0]:.2f}, y={pose[1]:.2f}")

    #         # ===== TECLADO =====
    #         if not self.handle_keyboard_input():
    #             break

    #         # ===== PERCEPÇÃO =====
    #         self.detect_objects()

    #         # =====================================================
    #         # 🚨 INTERRUPÇÃO — CUBO DETECTADO
    #         # =====================================================
    #         if navigating and self.cube_detected_a_frente:
    #             navigating = False
    #             collecting_cube = True

    #             # salva ponto exato
    #             saved_pose = self.lidar_pose

    #             # salva estado da missão
    #             saved_mission_step = self.mission_controller.current_step
    #             saved_nav_state = self.navigator.save_state()

    #             # pausa missão e navegação
    #             self.mission_controller.active = False
    #             self.navigator.stop()

    #             if self.DEBUG:
    #                 print("🧊 Cubo detectado — missão pausada")
    #             continue

    #         # =====================================================
    #         # 🧊 COLETA DO CUBO
    #         # =====================================================
    #         if collecting_cube:
    #             aligned = self.aligner.align_with_cube()

    #             if aligned:
    #                 reached = self.aligner.auto_approach_cube()

    #                 if reached:
    #                     label = self.color_classifier.capture_and_classify()
    #                     self.block_handler.pick(label)

    #                     collecting_cube = False
                        
    #                     # VERIFICA SE ESTAVA EM MODO FORWARD
    #                     if saved_nav_state and saved_nav_state["mode"] == "forward":
    #                         # No forward, não precisa voltar, retoma direto
    #                         navigating = True
    #                         self.mission_controller.current_step = saved_mission_step
    #                         self.navigator.restore_state(saved_nav_state)
    #                         self.mission_controller.active = True
                            
    #                         if self.DEBUG:
    #                             print("📦 Cubo coletado — retomando forward")
    #                     else:
    #                         # Em path/single, volta para o ponto salvo
    #                         returning = True
    #                         self.navigator.go_to(saved_pose[0], saved_pose[1])
                            
    #                         if self.DEBUG:
    #                             print("📦 Cubo coletado — retornando ao ponto salvo")
    #             continue

    #         # =====================================================
    #         # 🔄 RETORNO AO PONTO DA INTERRUPÇÃO
    #         # =====================================================
    #         if returning:
    #             arrived = self.navigator.update()

    #             if arrived:
    #                 returning = False
    #                 navigating = True

    #                 # restaura estado da missão
    #                 self.mission_controller.current_step = saved_mission_step
    #                 self.navigator.restore_state(saved_nav_state)
    #                 self.mission_controller.active = True

    #                 if self.DEBUG:
    #                     print("🔄 Missão retomada do ponto exato")
    #             continue

    #         # =====================================================
    #         # 🧭 EXECUÇÃO NORMAL DA MISSÃO
    #         # =====================================================
    #         if navigating and self.mission_controller.active:
    #             self.mission_controller.update()

    #         # ===== CONTROLE FINO =====
    #         self.update_rotation()

    #     print("🛑 Controller finalizado")

    # def run(self):
    #     print("=== YouBot | Missões com Coleta e Retorno ===")

    #     # ===== STEP INICIAL =====
    #     self.robot.step(self.time_step)
    #     self.lidar_gps.init_after_first_step()

    #     # ===== ESTADOS =====
    #     navigating = True
    #     collecting_cube = False
    #     returning = False

    #     saved_pose = None
    #     saved_mission_step = None
    #     saved_nav_state = None

    #     # ===== INICIA MISSÃO =====
    #     self.mission_controller.start()

    #     # ===== LOOP PRINCIPAL =====
    #     while self.robot.step(self.time_step) != -1:
            
    #         # ===== POSE =====
    #         pose = self.lidar_gps.step()
    #         if pose is not None:
    #             self.lidar_pose = pose
    #             if not self.DEBUG:
    #                 print(f"📍 Pose: x={pose[0]:.2f}, y={pose[1]:.2f}")

    #         # ===== TECLADO =====
    #         if not self.handle_keyboard_input():
    #             break

    #         # ===== PERCEPÇÃO =====
    #         self.detect_objects()

    #         # =====================================================
    #         # 🚨 INTERRUPÇÃO — CUBO DETECTADO
    #         # =====================================================
    #         if navigating and self.cube_detected_a_frente:
                
    #             navigating = False
    #             collecting_cube = True

    #             # salva ponto exato
    #             # salva ponto exato (poderá aguardar primeiros frames se necessário)
    #             saved_pose = self.lidar_pose
    #             if saved_pose is None:
    #                 if self.DEBUG: print("⚠️ Pose ainda indisponível — aguardando primeiro pose...")
    #                 wait_steps = 50  # ajuste se quiser (50 passos de simulação)
    #                 for _ in range(wait_steps):
    #                     if self.robot.step(self.time_step) == -1:
    #                         break
    #                     pose = self.lidar_gps.step()
                      
    #                     if pose is not None:
    #                         self.lidar_pose = pose
    #                         saved_pose = pose
    #                         break
    #                 if saved_pose is None and self.DEBUG:
    #                     print("⚠️ Não foi possível obter pose antes da pausa; prosseguindo sem saved_pose.")
               
    #             # salva estado da missão
    #             saved_mission_step = self.mission_controller.current_step
    #             saved_nav_state = self.navigator.save_state()

    #             # pausa missão e navegação
    #             self.mission_controller.active = False
    #             self.navigator.stop()

    #             if self.DEBUG:
    #                 print("🧊 Cubo detectado — missão pausada")
    #             continue

    #         # =====================================================
    #         # 🧊 COLETA DO CUBO
    #         # =====================================================
    #         if collecting_cube :
    #             aligned = self.aligner.align_with_cube()

    #             if aligned:
    #                 reached = self.aligner.auto_approach_cube()

    #                 if reached:
    #                     label = self.color_classifier.capture_and_classify()
    #                     self.block_handler.pick(label)
                        
    #                     collecting_cube = False
    #                     returning = True

    #                     # volta exatamente para onde estava
    #                     self.navigator.go_to(saved_pose[0], saved_pose[1])

    #                     if self.DEBUG:
    #                         print("📦 Cubo coletado — retornando ao ponto salvo")
    #             continue

    #         # =====================================================
    #         # 🔄 RETORNO AO PONTO DA INTERRUPÇÃO
    #         # =====================================================
    #         if returning:
    #             arrived = self.navigator.update()

    #             if arrived:
    #                 returning = False
    #                 navigating = True

    #                 # restaura estado da missão
    #                 self.mission_controller.current_step = saved_mission_step
    #                 self.navigator.restore_state(saved_nav_state)
    #                 self.mission_controller.active = True

    #                 if self.DEBUG:
    #                     print("🔄 Missão retomada do ponto exato")
    #             continue

    #         # =====================================================
    #         # 🧭 EXECUÇÃO NORMAL DA MISSÃO
    #         # =====================================================
    #         if navigating and self.mission_controller.active:
    #             self.mission_controller.update()

    #         # ===== CONTROLE FINO =====
    #         self.update_rotation()

    #     print("🛑 Controller finalizado")

    def run(self):
        print("=== YouBot | Garra desce a 0.104 m | Com alinhamento lateral automático ===")
        self.robot.step(self.time_step)
        self.lidar_gps.init_after_first_step()
        
        while self.robot.step(self.time_step) != -1:
            # inside main loop
            pose = self.lidar_gps.step()
            
            if pose is not None:
                self.lidar_pose = pose
                if self.DEBUG:
                    print(f"📍 LidarGPS pose: x={pose[0]:.3f}, y={pose[1]:.3f}")
            if not self.handle_keyboard_input():
                break
            self.detect_objects()
            # Prioriza pegar se houver cubo acessível e NÃO estiver bloqueado
            if self.cube_detected_a_frente:
                aligned = self.aligner.align_with_cube()
                if aligned:
                    reached = self.aligner.auto_approach_cube()
                    if reached:
                        label = self.color_classifier.capture_and_classify()
                        self.block_handler.pick(label) 
                        
            # Senão, se houver obstáculo detectado, trata de evitar
            # elif self.obstacle_detected:

            #     self.avoid_obstacle()
            else:

                self.update_movement()
            self.update_rotation()

        self.base.move(0, 0, 0)
        print("🛑 Controller finalizado")


if __name__ == "__main__":
    YouBotController().run()