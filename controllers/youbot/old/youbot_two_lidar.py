#!/usr/bin/env python
from controller import Robot, Keyboard
from base import Base
from arm import Arm
from gripper import Gripper
import numpy as np
import math
import sys

class YouBotController:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())

        # componentes
        self.base = Base(self.robot)
        self.arm = Arm(self.robot)
        self.gripper = Gripper(self.robot)

        # teclado
        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.time_step)

        # LiDARs (nomes informados)
        self.lidar_low = self.robot.getDevice("lidar_horizontal")     # LiDAR embaixo
        self.lidar_high = self.robot.getDevice("lidar_horizontal_2")  # LiDAR em cima

        if self.lidar_low is None:
            print("❌ lidar_horizontal NÃO encontrado", file=sys.stderr)
            self.lidar_low = None
            return
        if self.lidar_high is None:
            print("❌ lidar_horizontal_2 NÃO encontrado", file=sys.stderr)
            self.lidar_high = None
            return

        # habilita com timestep do controlador
        try:
            self.lidar_low.enable(self.time_step)
        except Exception:
            self.lidar_low.enable(int(self.lidar_low.getBasicTimeStep()))
        try:
            self.lidar_high.enable(self.time_step)
        except Exception:
            self.lidar_high.enable(int(self.lidar_high.getBasicTimeStep()))

        # optional: try enablePointCloud safely (não obrigatório)
        try:
            if hasattr(self.lidar_low, "enablePointCloud"):
                self.lidar_low.enablePointCloud()
        except Exception:
            pass
        try:
            if hasattr(self.lidar_high, "enablePointCloud"):
                self.lidar_high.enablePointCloud()
        except Exception:
            pass

        print("✅ lidar_horizontal e lidar_horizontal_2 ativados")

        # movimento
        self.forward_speed = 0.3
        self.strafe_speed = 0.3
        self.movement_duration = 10
        self.move_forward_counter = 0
        self.move_backward_counter = 0
        self.strafe_left_counter = 0
        self.strafe_right_counter = 0
        self.step_count = 0

        # limiares / parâmetros
        self.CUBE_HEIGHT = 0.03         # 3 cm
        self.DISTANCE_DIFF_THRESH = 0.001  # se diferença > 2 cm, consideramos diferente (ajuste)
        self.DEBUG = True

    # teclado
    def handle_keyboard_input(self):
        key = self.keyboard.getKey()
        while key >= 0:
            if key in [ord('W'), ord('w')]:
                self.move_forward_counter = self.movement_duration
            elif key in [ord('S'), ord('s')]:
                self.move_backward_counter = self.movement_duration
            elif key in [ord('A'), ord('a')]:
                self.strafe_left_counter = self.movement_duration
            elif key in [ord('D'), ord('d')]:
                self.strafe_right_counter = self.movement_duration
            elif key == ord(' '):
                self.base.move(0, 0, 0)
            elif key in [ord('Q'), ord('q')]:
                return False
            key = self.keyboard.getKey()
        return True

    # movimento
    def update_movement(self):
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

    # detecção principal usando os dois LiDARs
    def detect_objects(self):
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

        # pegar apenas valores válidos
        valid_low = low_ranges[np.isfinite(low_ranges) & (low_ranges > 0)]
        valid_high = high_ranges[np.isfinite(high_ranges) & (high_ranges > 0)]

        min_low = np.min(valid_low) if valid_low.size > 0 else float('inf')
        min_high = np.min(valid_high) if valid_high.size > 0 else float('inf')

        # caso 1: nenhum detecta nada
        if min_low == float('inf') and min_high == float('inf'):
            if self.DEBUG:
                print("· Nada detectado pelos LiDARs")
            return

        # caso 2: apenas baixo detecta algo
        if min_low < float('inf') and min_high == float('inf'):
            print(f"🟦 Cubinho detectado (apenas baixo) | baixo={min_low:.3f} m")
            return

        # caso 3: apenas alto detecta algo
        if min_low == float('inf') and min_high < float('inf'):
            print(f"🟨 Objeto alto detectado (apenas alto) | alto={min_high:.3f} m")
            return

        # caso 4: ambos detectam
        diff = min_high - min_low
        if diff > self.DISTANCE_DIFF_THRESH:
            if min_low < min_high:
                print(f"🔹 Cubo na frente de obstáculo | baixo={min_low:.3f} m, alto={min_high:.3f} m, Δ={diff:.3f} m")
            else:
                print(f"⚠️ Inconsistência: alto mais próximo que baixo | baixo={min_low:.3f} m, alto={min_high:.3f} m, Δ={diff:.3f} m")
        else:
            print(f"🟥 Obstáculo grande detectado | baixo={min_low:.3f} m, alto={min_high:.3f} m, Δ={diff:.3f} m")


    # loop principal
    def run(self):
        print("=== YouBot | Deteção com lidar_horizontal (baixo) + lidar_horizontal_2 (alto) ===")
        print("WASD move | Espaço para parar | Q sai")
        while self.robot.step(self.time_step) != -1:
            self.step_count += 1
            if not self.handle_keyboard_input():
                break
            self.update_movement()
            # opcional: processar menos frequentemente p/ performance (ex: every 2 steps)
            self.detect_objects()
        # parar
        self.base.move(0, 0, 0)
        print("🛑 Controller finalizado")


if __name__ == "__main__":
    controller = YouBotController()
    if getattr(controller, "lidar_low", None) is not None and getattr(controller, "lidar_high", None) is not None:
        controller.run()
