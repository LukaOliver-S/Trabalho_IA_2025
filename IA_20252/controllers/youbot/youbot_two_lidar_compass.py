#!/usr/bin/env python
import math
from controller import Robot, Keyboard
from base import Base
from arm import Arm
from gripper import Gripper
import numpy as np
import sys

class YouBotController:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.dt = self.time_step / 1000.0

        # ================= COMPONENTES =================
        self.base = Base(self.robot)
        self.arm = Arm(self.robot)
        self.gripper = Gripper(self.robot)

        # ================= KEYBOARD =================
        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.time_step)

        # ================= LiDARs =================
        self.lidar_low = self.robot.getDevice("lidar_horizontal")
        self.lidar_high = self.robot.getDevice("lidar_horizontal_2")

        if self.lidar_low is None or self.lidar_high is None:
            print("❌ ERRO: LiDARs não encontrados", file=sys.stderr)
            sys.exit(1)

        self.lidar_low.enable(self.time_step)
        self.lidar_high.enable(self.time_step)

        # ================= COMPASS =================
        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.time_step)
        # ================= MOVIMENTO =================
        self.forward_speed = 0.3
        self.strafe_speed = 0.3
        self.movement_duration = 10

        self.move_forward_counter = 0
        self.move_backward_counter = 0
        self.strafe_left_counter = 0
        self.strafe_right_counter = 0

        # ================= PARÂMETROS =================
        self.CUBE_HEIGHT = 0.03
        self.DISTANCE_DIFF_THRESH = 0.01
        self.DEBUG = True

        # ================= PICK =================
        self.PICK_DISTANCE = 0.104
        self.PICK_TOL = 0.003

        self.gripper_descended = False
        self.is_picking = False   # <<< flag CRÍTICA
        self.initial_angle = 0.0
        self.rotating = False
        self.target_angle = None
    # ================= KEYBOARD =================
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
            elif key == ord(' '):
                self.base.move(0, 0, 0)
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

    # ================= MOVIMENTO =================
    def update_movement(self):
        # 🚨 durante o pick o robô NÃO SE MOVE
        if self.is_picking:
            self.base.move(0, 0, 0)
            return

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

    # ================= LIDAR READ =================
    def read_lidars(self):
        low = np.array(self.lidar_low.getRangeImage())
        high = np.array(self.lidar_high.getRangeImage())

        low = low[np.isfinite(low) & (low > 0)]
        high = high[np.isfinite(high) & (high > 0)]

        min_low = np.min(low) if low.size else float('inf')
        min_high = np.min(high) if high.size else float('inf')

        return min_low, min_high
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

        deadzone = 0.1
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





    # ================= DETECT (INTACTA) =================
    def detect_objects(self):
        self.cube_detected_a_frente = False  # Reset a cada ciclo

        try:
            low_ranges = np.array(self.lidar_low.getRangeImage(), dtype=np.float32)
        except Exception:
            low_ranges = np.array([], dtype=np.float32)
        try:
            high_ranges = np.array(self.lidar_high.getRangeImage(), dtype=np.float32)
        except Exception:
            high_ranges = np.array([], dtype=np.float32)

        if low_ranges.size == 0:
            if self.DEBUG:
                print("Nenhuma leitura do lidar_horizontal (baixo)")
            return
        if high_ranges.size == 0:
            if self.DEBUG:
                print("Nenhuma leitura do lidar_horizontal_2 (alto)")
            return
        self.lidar_low.enablePointCloud()
        self.lidar_high.enablePointCloud()
        
        valid_low = low_ranges[np.isfinite(low_ranges) & (low_ranges > 0)]
        valid_high = high_ranges[np.isfinite(high_ranges) & (high_ranges > 0)]

        min_low = np.min(valid_low) if valid_low.size > 0 else float('inf')
        min_high = np.min(valid_high) if valid_high.size > 0 else float('inf')

        if min_low == float('inf') and min_high == float('inf'):
          #  if self.DEBUG:
             #   print("· Nada detectado pelos LiDARs")
            return

        if min_low < float('inf') and min_high == float('inf'):
            print(f"🟦 Cubinho detectado (apenas baixo) | baixo={min_low:.3f} m")
            self.cube_detected_a_frente = True
            return

        if min_low == float('inf') and min_high < float('inf'):
            print(f"🟨 Objeto alto detectado (apenas alto) | alto={min_high:.3f} m")
            return

        diff = min_high - min_low
        if diff > self.DISTANCE_DIFF_THRESH:
            if min_low < min_high:
                print(f"🔹 Cubo na frente de obstáculo | baixo={min_low:.3f} m, alto={min_high:.3f} m, Δ={diff:.3f} m")
                self.cube_detected_a_frente = True
            else:
                print(f"⚠️ Inconsistência: alto mais próximo que baixo | baixo={min_low:.3f} m, alto={min_high:.3f} m, Δ={diff:.3f} m")
        else:
            print(f"🟥 Obstáculo grande detectado | baixo={min_low:.3f} m, alto={min_high:.3f} m, Δ={diff:.3f} m")

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
            if not self.gripper_descended:
                print("⬇️ Cubo a 0.104 m → descendo garra")
                self.is_picking = True

                # Para tudo
                self.move_forward_counter = 0
                self.move_backward_counter = 0
                self.strafe_left_counter = 0
                self.strafe_right_counter = 0
                self.base.move(0, 0, 0)

                # 1. Abre a garra
                print("Abrindo garra...")
                self.gripper.release()
                self._step_wait(1.0)  # Espera garantir abertura

                # 2. Desce o braço até o cubo
                print("Descendo braço...")
                self.arm.set_orientation(Arm.FRONT)
                self._step_wait(0.05)
                self.arm.set_height(Arm.FRONT_FLOOR)
                self._step_wait(3.5)
                self.arm.motors[1].setPosition(-1.10)
                self.arm.motors[2].setPosition(-1.70)
                self.arm.motors[3].setPosition(-0.70)
                self._step_wait(1.5)  # Espera garantir braço no chão

                # 3. Fecha a garra
                print("Fechando garra...")
                self.gripper.grip()
                self._step_wait(1.2)  # Espera garantir fechamento

                # 4. Sobe o braço
                print("Subindo braço...")
                self.arm.motors[1].setPosition(-1.0)
                self.arm.motors[2].setPosition(-1.5)
                self.arm.motors[3].setPosition(-0.6)
                self._step_wait(1.5)

                print("Levando para trás (braço esticado para trás, base reta)...")
                self.arm.motors[0].setPosition(0.0)      # base reta (NÃO gira)
                self.arm.motors[1].setPosition(0.5)      # levanta o braço
                self.arm.motors[2].setPosition(1.0)      # estica para trás
                self.arm.motors[3].setPosition(0.9)     # joga o braço para trás
                self.arm.motors[4].setPosition(-0.5)      # gripper para baixo
                self._step_wait(6.0)

                print("Liberando cubo atrás do robô...")
                self.gripper.release()
                self._step_wait(0.7)

                print("Retornando braço e garra para posição inicial...")
                self.arm.set_height(Arm.RESET)
                self.arm.set_orientation(Arm.FRONT)
                self._step_wait(3.0)
                self.gripper.release()
                self._step_wait(0.5)

                print("Pegada finalizada.")
                self.gripper_descended = True
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
        print("=== YouBot | Garra desce a 0.104 m | Sem avanço automático ===")

        while self.robot.step(self.time_step) != -1:
            if not self.handle_keyboard_input():
                break
           # current_angle = self.get_current_angle()
            #print(f"current_angle: {current_angle}")
            self.update_movement()
            self.detect_objects()
            self.descend_gripper_if_target_distance()
            self.update_rotation()

        self.base.move(0, 0, 0)
        print("🛑 Controller finalizado")


if __name__ == "__main__":
    YouBotController().run()
