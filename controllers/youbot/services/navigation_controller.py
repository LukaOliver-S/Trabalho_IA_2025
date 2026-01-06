import math

class NavigationController:
    def __init__(self, robot):
        self.robot = robot
        self.base = robot.base

        # ===== PARÂMETROS =====
        self.MAX_V = 0.09
        self.MIN_V = 0.03
        self.TOL_X = 0.07
        self.TOL_Y = 0.07
        self.KP = 0.7

        # ===== MAPA (TRECHOS) =====
        self.paths = {
            "busca_principal": [
                (-3.4,   1.0),
                (-3.3,  -1.0),
                (-3.2,   1.0),
                (-3.1,  -1.0),
                (-3.0,   1.0),
                (-2.9,  -1.0),
                (-2.8,   1.0),
                (-2.7,  -1.0),
                (-2.6,   1.0),
                (-2.5,  -1.0),
                (-2.4,   1.0),
                (-2.3,  -1.0),
                (-2.2,   1.0),
                (-2.1,  -1.0),
                (-2.0,   1.0),
                (-1.9,  -1.0),
                (-1.8,   1.0),
                (-1.7,  -1.0),
                (-1.66,  1.0),
                (-1.7,  -0.25),
                (-1.6,  -0.25),
                (-1.5,   0.20),
                (-1.4,  -0.20),
                (-1.3,   0.20),
                (-1.2,  -0.20),
                (-1.1,   0.20),
                (-1.0,  -0.20),
                (-0.9,   0.20),
                (-0.8,  -0.20),
                (-0.7,   0.20),
                (-0.6,  -0.20),
                (-0.5,   0.25),
                (-0.4,  -0.25),
                (-0.3,  0.25),
                (-0.2,  -0.25),
                (-0.1,  -0.25),
                (-0.04,  -0.25),
                (-0.1,  -0.25),
                (-0.1,  0.20),
                (-0.05,  0.0),
            ],

            "busca_lateral_esquerda1": [
                (0.1,  0.1),
                (-0.7, 0.2),
                (0.1,  0.3),
                (-0.25, 0.4),
                (0.15, 0.5),
                (-0.25, 0.6),
                (0.15, 0.7),
                (0.15, 0.7),
                (-0.3, 0.2),
            ],

            "busca_lateral_esquerda2": [
                (-0.2,  0.25),
                (-0.2, 0.95),
                (-0.3,  0.25),
                (-0.4, 0.95),
                (-0.45,  0.25),
                (-0.8, 0.25),
                (-0.5, 0.25),
            ],

            "busca_lateral_esquerda3": [
                (-0.30,  0.5),
                (-0.25,  0.92),
                (-0.15, 0.5),
                (-0.05,  0.2),
                (0.10, 0.45),
                (0.20,  0.92),
                (0.30, 0.45),
                (0.40,  0.92),
                (0.50, 0.55),
                (0.60,  0.92),
                (0.70, 0.55),
                (0.80,  0.92),
                (0.90, 0.55),
                (1.0,  0.92),
                (1.1, 0.55),
                (1.2,  0.92),
                (1.3, 0.55),
                (1.3,  0.92),
            ],

            "busca_fundo_direita": [
                (0.5,  0.8),
                (1.60, 0.7),
                (0.55,  0.6),
                (1.60, 0.5),
                (1.1,  0.4),
                (1.60, 0.3),
                (1.1,  0.2),
                (1.60, 0.1),
                (1.1, 0.0),
                (1.60, -0.1),
                (1.2, -0.2),
                (1.60, -0.3),
                (1.2, -0.4),
                (1.60, -0.5),
                (1.2, -0.6),
                (1.63, -0.63),
                (1.4, 0)
            ],

            "busca_lateral_direita": [
                (1.5, -0.95),
                (1.4, -0.2),
                (1.3, -0.95),
                (1.2, -0.2),
                (1.1, -0.95),
                (1.0, -0.45),
                (1.1, -0.95),
                (1.0, -0.45),
                (0.9, -0.95),
                (0.8, -0.45),
                (0.7, -0.95),
                (0.6, -0.45),
                (0.5, -0.95),
                (0.4, -0.45),
                (0.3, -0.95),
                (0.2, -0.45),
                (0.1, -0.95),
                (0.0, -0.45),
                (-0.1, -0.95),
                (-0.2, -0.45),
                (-0.33, -0.95),
                (-0.3, -0.30),
                (-0.4, -0.30),
                (-0.5, -0.30),
                (-0.1, -0.30),
            ],
            "caixa_blue": [
                (0.35, -1.0),
            ],
            "caixa_blue_re": [
                (0.35, -0.6),
            ],
            "caixa_red": [
                (1.2, -0.7),
                (1.53, 0.0),
            ],
            "caixa_red_re": [
                (1.4, 0),
            ],
            "caixa_green": [
                (1.4, 0.6),
                (0.5, 0.92),
            ],
             "fall_back_1":[
                (-0.2, 0.80),
                (-0.2, 0.03),
                (-1.7,  0.03),
                (-1.80,0.003)
            ],
            "fall_back_2":[
                (-2.3,0.003),
                (-2.3,1.135),
                (-2.6,1.135),
                (-3.4,1.135),
                (-3.4,1.0) 
            ]
        }

        # ===== ESTADO =====
        self.active = False
        self.mode = None              # "single" | "path"
        self.current_path = None
        self.current_wp = 0
        self.tx = None
        self.ty = None

    # =====================================================
    # API PÚBLICA
    # =====================================================

    def go_to(self, x, y):
        """Vai para um único ponto"""
        self.tx = x
        self.ty = y
        self.mode = "single"
        self.active = True

    def start(self, path_name):
        """Inicia um caminho pré-definido"""
        if path_name not in self.paths:
            print(f"❌ Caminho '{path_name}' não existe")
            return

        self.current_path = self.paths[path_name]
        self.current_wp = 0
        self.mode = "path"
        self.active = True
        self._load_current_wp()

    def stop(self):
        self.base.move(0, 0, 0)
        self.active = False
        self.mode = None

    def is_active(self):
        return self.active

    # =====================================================
    # SALVAR / RESTAURAR ESTADO (para coleta de cubos)
    # =====================================================

    def save_state(self):
        return {
            "mode": self.mode,
            "current_path": self.current_path,
            "current_wp": self.current_wp,
            "tx": self.tx,
            "ty": self.ty,
        }

    def restore_state(self, state):
        self.mode = state["mode"]
        self.current_path = state["current_path"]
        self.current_wp = state["current_wp"]
        self.tx = state["tx"]
        self.ty = state["ty"]
        self.active = True

    # =====================================================
    # LOOP PRINCIPAL
    # =====================================================

    def update(self):
        if not self.active:
            return True

            # Se o LidarGPS ainda não forneceu um pose válido, não avançamos.
        if not hasattr(self.robot, "lidar_pose") or self.robot.lidar_pose is None:
            if getattr(self.robot, "DEBUG", False):
                print("⚠️ NavigationController.update: lidar_pose indisponível (aguardando).")
            return False


        x, y = self.robot.lidar_pose[:2]

        dx = self.tx - x
        dy = self.ty - y

        arrived_x = abs(dx) < self.TOL_X
        arrived_y = abs(dy) < self.TOL_Y

        # ===== CHEGOU =====
        if arrived_x and arrived_y:

            if self.mode == "single":
                self.stop()
                return True

            self.current_wp += 1
            if self.current_wp >= len(self.current_path):
                self.stop()
                return True

            self._load_current_wp()
            return False

        # =================================================
        # CONTROLE P (NO MUNDO)
        # =================================================
        ux = 0.0
        uy = 0.0

        if not arrived_x:
            ux = self.KP * dx
            if abs(ux) < self.MIN_V:
                ux = self.MIN_V * math.copysign(1, dx)

        if not arrived_y:
            uy = self.KP * dy
            if abs(uy) < self.MIN_V:
                uy = self.MIN_V * math.copysign(1, dy)

        # 🔒 SATURAÇÃO (RESPEITA MAX_V)
        ux = max(-self.MAX_V, min(self.MAX_V, ux))
        uy = max(-self.MAX_V, min(self.MAX_V, uy))

        # =================================================
        # TRANSFORMAÇÃO MUNDO → ROBÔ (BÚSSOLA)
        # =================================================
        theta = self._get_heading_rad()

        vx =  math.cos(theta) * ux + math.sin(theta) * uy
        vy = -math.sin(theta) * ux + math.cos(theta) * uy

        self.base.move(vx, -vy, 0)
        return False

    # =====================================================
    # INTERNO
    # =====================================================

    def _load_current_wp(self):
        self.tx, self.ty = self.current_path[self.current_wp]

    def _get_heading_rad(self):
        """Heading atual do robô em radianos (bússola)."""
        north = self.robot.compass.getValues()
        return math.atan2(north[0], north[1])