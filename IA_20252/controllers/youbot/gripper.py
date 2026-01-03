"""
Copyright 1996-2024 Cyberbotics Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Description: Python wrapper for YouBot gripper control
"""


MIN_POS = 0.0      
MAX_POS = 0.025   

GAP_MIN = 0.0     
GAP_MAX = 0.05     

# Tamanho do cubo (aresta)
CUBE_SIZE = 0.00015 

def bound(value, min_val, max_val):
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))

class Gripper:
    """Controls the YouBot parallel gripper"""

    def __init__(self, robot):
        """Initialize gripper motors and sensors

        Args:
            robot: Webots Robot instance
        """
        self.robot = robot
        self.time_step = int(robot.getBasicTimeStep())

        # Motor do dedo (controla os dois dedos)
        self.finger = robot.getDevice("finger::left")

        # Encoder linear do dedo
        self.finger_sensor = robot.getDevice("finger::leftsensor")
        if self.finger_sensor:
            self.finger_sensor.enable(self.time_step)

        # Configura velocidade
        if self.finger:
            self.finger.setVelocity(0.01)
        else:
            print("Warning: Could not find gripper motor 'finger::left'")

        self.is_gripping = False   
        self.has_cube = False      

    # ---------- comandos básicos ----------
    def grip(self):
        """Fecha totalmente a garra (tenta agarrar)"""
        if self.finger:
            self.finger.setPosition(MIN_POS)
        self.is_gripping = True
        self.has_cube = False  

    def release(self):
        """Abre totalmente a garra (solta qualquer coisa)"""
        if self.finger:
            self.finger.setPosition(MAX_POS)
        self.is_gripping = False
        self.has_cube = False

    def set_gap(self, gap):
        """Set gripper to a specific gap width between fingers

        Args:
            gap: desired gap between fingers in meters
        """
        gap = bound(gap, GAP_MIN, GAP_MAX)

        # mapeia gap -> posição do atuador linear (assumindo relação linear)
        alpha = (gap - GAP_MIN) / (GAP_MAX - GAP_MIN)
        pos = MIN_POS + alpha * (MAX_POS - MIN_POS)
        pos = bound(pos, MIN_POS, MAX_POS)

        if self.finger:
            self.finger.setPosition(pos)

        self.is_gripping = (gap <= CUBE_SIZE)  
       

    # ---------- leitura ----------
    def current_pos(self):
        """Posição atual do atuador linear (m)"""
        if not self.finger_sensor:
            return MIN_POS
        return float(self.finger_sensor.getValue())

    def current_gap(self):
        """Gap estimado entre os dedos (m)"""
        pos = self.current_pos()
        alpha = (pos - MIN_POS) / (MAX_POS - MIN_POS)
        gap = GAP_MIN + alpha * (GAP_MAX - GAP_MIN)
        return bound(gap, GAP_MIN, GAP_MAX)

    def check_cube_grasped(self):
        """Atualiza e retorna se o cubo de 3 cm foi realmente pego.

        Lógica:
          - se depois de fechar o gap final for < CUBE_SIZE => dedos passaram do cubo => não pegou
          - se gap final >= CUBE_SIZE - tol => dedos travaram no cubo => pegou
        """
        gap = self.current_gap()
        print(CUBE_SIZE)
        print(gap)
        if gap < CUBE_SIZE:
            # conseguiu fechar demais: não travou no cubo
            self.has_cube = False
        else:
            # dedos não conseguiram fechar além do tamanho do cubo
            self.has_cube = True

        return self.has_cube

    # ---------- estados ----------
    def is_closed(self):
        return self.is_gripping

    def has_object(self):
        return self.has_cube