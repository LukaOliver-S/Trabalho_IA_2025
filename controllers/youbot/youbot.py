import sys

import math
from controller import Robot
from base import Base
from arm import Arm
from gripper import Gripper
import numpy as np
from pathlib import Path

from services import (
    BlockHandler,
    SensorSuite,
    NavigationController,
    MissionController,
    AlignmentController,
    LidarGpsController,
    ColorClassifier,
    FuzzySimple,
    ObjectDetector
)

import warnings
warnings.filterwarnings("ignore")

class YouBotController:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.dt = self.time_step / 1000.0
        self.DEBUG = False
        
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
        self.lidar_global.enablePointCloud()
        
        # ================= LIDAR_GPS ===============

        self.lidar_gps = LidarGpsController(
            self.lidar_global,
            model_path=Path("./models/lidar_gps_yaw_4.pth"),
            T=3,
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
        self.forward_speed = 0.09
        self.strafe_speed = 0.09
        self.movement_duration = 10
        self.move_forward_counter = 0
        self.move_backward_counter = 0
        self.strafe_left_counter = 0
        self.strafe_right_counter = 0
                
        # fuzzy velocity selector 
        self.fuzzy = FuzzySimple(v_max=0.10)
        
        # ================= PARÂMETROS =================
        self.CUBE_HEIGHT = 0.03
        self.DISTANCE_DIFF_THRESH = 0.01
        
        # ================= PICK =================
        self.PICK_DISTANCE = 0.169
        self.PICK_TOL = 0.001
        self.MIN_APPROACH_SPEED = 0.03  # velocidade mínima para aproximação
        self.ALIGN_DEADZONE = 0.005     # tolerância lateral para considerar "centralizado"
        
        #================== GRIPPER CONTROLLER ========
        self.gripper_descended = False
        self.is_picking = False   
        self.initial_angle = 0.0
        self.rotating = False
        self.rotation_started = False      
        self.rotation_direction = 0      
        self.target_angle = None
        self.Cube_min_dist = 0.25
        self. failed_alignment_attempts = 0
        self.max_alignment_attempts = 2
        
        # ================= Align Controller ==========
        self.max_align_ticks = 6000
        self.aligner = AlignmentController(
            self.base, self.sensors,
            kp=0.3, max_vy=0.1, deadzone=self.ALIGN_DEADZONE,
            forward_speed=self.forward_speed,
            pick_distance=self.PICK_DISTANCE,
            pick_tol=self.PICK_TOL,
            min_approach_speed=self.MIN_APPROACH_SPEED,
            debug=self.DEBUG,
            fuzzy=self.fuzzy,
            max_align_ticks= self.max_align_ticks, 
            max_capacity=15,
        )
    
        # ================= BLOCK HANDLER & COLOR CLASSIFIER ==========
        self.block_handler = BlockHandler(self)
        self.color_classifier = ColorClassifier(self, "./models/mlp_ab_model.joblib")
        
        # ================= NAVIGATION =================
        self.navigator = NavigationController(
            self,
        )
        self.mission_controller = MissionController(self.navigator, self)

     # ================== OBJECT DETECTOR==========
        self.OBSTACLE_MIN_DIST = 0.30
        self.obstacle_detected = False
        self.obstacle_blocking_cube = False
        
        self.detector = ObjectDetector(self.sensors,
                                distance_diff_thresh=self.DISTANCE_DIFF_THRESH,
                                obstacle_min_dist=self.OBSTACLE_MIN_DIST,
                                step_wait=self._step_wait,
                                debug=self.DEBUG)

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

    def rotate_to_angle(self, target_angle):
        """Inicia a rotação em direção a target_angle (não-bloqueante)."""
        self.target_angle = target_angle % 360
        self.rotating = True
        self.rotation_started = False  # reinicia a inicialização da direção

    def update_rotation(self):
        if not self.rotating:
            return False  # <<< NÃO está girando

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

        # Se dentro do deadzone, parar de girar
        if abs(diff) <= deadzone:
            self.base.move(0, 0, 0)
            self.rotating = False
            self.rotation_started = False
            return False  # <<< terminou de girar

        # Velocidade proporcional
        speed = max_speed
        if abs(diff) < slow_zone:
            speed *= 0.3

        # Gira na direção correta
        self.base.move(0, 0, speed * self.rotation_direction)

        return True  # <<< ESTÁ girando

        
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

    # ================= DETECT =================
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
            
    # ================= UTILS =================
    def _step_wait(self, seconds):
        steps = max(1, int(seconds / self.dt))
        for _ in range(steps):
            if self.robot.step(self.time_step) == -1:
                break

    def wait(self, steps=30):
        for _ in range(steps):
            self.robot.step(self.time_step)
            

    # ================= RUN =================
    def run(self):
        print("=== YouBot | Missões com Coleta e Retorno ===")

        # ===== STEP INICIAL =====
        self.robot.step(self.time_step)
        self.lidar_gps.init_after_first_step()

        # ===== ESTADOS =====
        navigating = True
        collecting_cube = False
        returning = False

        saved_pose = None
        saved_mission_step = None
        saved_nav_state = None

        # ===== INICIA MISSÃO =====
        self.mission_controller.start()

        # ===== LOOP PRINCIPAL =====
        while self.robot.step(self.time_step) != -1:
            
            # ===== POSE =====
            pose = self.lidar_gps.step()
            if pose is not None:
                self.lidar_pose = pose
                if  self.DEBUG:
                    print(f"📍 Pose: x={pose[0]:.2f}, y={pose[1]:.2f}")

            # =====================================================
            # 🔄 CONTROLE DE ROTAÇÃO
            # =====================================================
            is_rotating = self.update_rotation()  # <-- deve retornar True/False

            # =====================================================
            # 👁️ PERCEPÇÃO (somente se NÃO estiver rotacionando)
            # =====================================================
            if not is_rotating:
                self.detect_objects()
            else:
                self.cube_detected_a_frente = False  # segurança extra

            # =====================================================
            # 🚨 INTERRUPÇÃO — CUBO DETECTADO
            # =====================================================
            if navigating and self.cube_detected_a_frente and not is_rotating:
                
                # ===== VERIFICA CAPACIDADE ANTES DE INTERROMPER =====
                if self.block_handler.counter >= 15:  # max_capacity
                    if self.DEBUG:
                        print(f"Base cheia ({self.block_handler.counter}/15) — ignorando cubo detectado")
                    # Não pausa a missão, continua navegando
                    continue
                
                navigating = False
                collecting_cube = True

                # salva pose exata
                saved_pose = self.lidar_pose
                if saved_pose is None:
                    if self.DEBUG:
                        print("Pose ainda indisponível — aguardando...")
                    for _ in range(50):
                        if self.robot.step(self.time_step) == -1:
                            break
                        pose = self.lidar_gps.step()
                        if pose is not None:
                            self.lidar_pose = pose
                            saved_pose = pose
                            break

                # salva estado da missão
                saved_mission_step = self.mission_controller.current_step
                saved_nav_state = self.navigator.save_state()

                # pausa missão
                self.mission_controller.active = False
                self.navigator.stop()

                if self.DEBUG:
                    print("🧊 Cubo detectado — missão pausada")
                continue

            # =====================================================
            # 🧊 COLETA DO CUBO
            # =====================================================
            if collecting_cube:
                # ===== VERIFICA TIMEOUT DE ALINHAMENTO =====
                if self.aligner.alignment_failed:
                    self.failed_alignment_attempts += 1  # ← ADICIONAR: incrementa tentativas
                    
                    # ← ADICIONAR: verifica se excedeu limite
                    if self.failed_alignment_attempts >= self.max_alignment_attempts:
                        if self.DEBUG:
                            print(f"⛔ Desistindo do cubo após {self.failed_alignment_attempts} tentativas falhadas")
                        
                        collecting_cube = False
                        navigating = True
                        self.failed_alignment_attempts = 0  # ← reset contador
                        
                        # Restaura missão
                        self.mission_controller.current_step = saved_mission_step
                        if saved_nav_state:
                            self.navigator.restore_state(saved_nav_state)
                        self.mission_controller.active = True
                        
                        self.aligner.align_ticks = 0
                        self.aligner.alignment_failed = False
                        continue
                    
                    # ← ADICIONAR: se não excedeu, tenta novamente
                    collecting_cube = False
                    navigating = True

                    # Restaura missão
                    self.mission_controller.current_step = saved_mission_step
                    if saved_nav_state:
                        self.navigator.restore_state(saved_nav_state)
                    self.mission_controller.active = True

                    # Reset do aligner para próxima tentativa
                    self.aligner.align_ticks = 0
                    self.aligner.alignment_failed = False

                    if self.DEBUG:
                        print(f"⛔ Falha no alinhamento (tentativa {self.failed_alignment_attempts}/{self.max_alignment_attempts}) — retentando")
                    continue

                # ===== TENTATIVA DE ALINHAMENTO =====
                aligned = self.aligner.align_with_cube()

                if aligned:
                    reached = self.aligner.auto_approach_cube()

                    if reached:
                        # ===== VERIFICA CAPACIDADE ANTES DE PEGAR =====
                        if self.block_handler.counter >= 15:  # max_capacity
                            if self.DEBUG:
                                print(f"⛔ Base cheia ({self.block_handler.counter}/15) — NÃO pegando cubo")
                            
                            collecting_cube = False
                            returning = True
                            self.navigator.go_to(saved_pose[0], saved_pose[1])
                        else:
                            label = self.color_classifier.capture_and_classify()
                            self.block_handler.pick(label)
                            
                            self.failed_alignment_attempts = 0  # ← ADICIONAR: reset ao pegar com sucesso

                            collecting_cube = False
                            returning = True

                            self.navigator.go_to(saved_pose[0], saved_pose[1])

                            if self.DEBUG:
                                print("📦 Cubo coletado — retornando ao ponto salvo")
                continue


            # =====================================================
            # 🔄 RETORNO AO PONTO DA INTERRUPÇÃO
            # =====================================================
            if returning:
                arrived = self.navigator.update()

                if arrived:
                    returning = False
                    navigating = True

                    self.mission_controller.current_step = saved_mission_step
                    self.navigator.restore_state(saved_nav_state)
                    self.mission_controller.active = True

                    if self.DEBUG:
                        print("🔄 Missão retomada do ponto exato")
                continue

            # =====================================================
            # 🧭 EXECUÇÃO NORMAL DA MISSÃO
            # =====================================================
            if navigating and self.mission_controller.active:
                self.mission_controller.update()

        print("🛑 Controller finalizado")

    
if __name__ == "__main__":
    YouBotController().run()