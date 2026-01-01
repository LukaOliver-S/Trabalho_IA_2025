import math
import numpy as np

class AlignmentController:
    def __init__(self, base, sensors, kp=1.5, max_vy=0.2,
                 deadzone=0.005, forward_speed=0.3,
                 pick_distance=0.104, pick_tol=0.005,
                 min_approach_speed=0.03, debug=False,fuzzy = None):
        self.base = base
        self.sensors = sensors
        self.kp = kp
        self.max_vy = max_vy
        self.deadzone = deadzone
        self.forward_speed = forward_speed
        self.PICK_DISTANCE = pick_distance
        self.PICK_TOL = pick_tol
        self.MIN_APPROACH_SPEED = min_approach_speed
        self.DEBUG = debug
        self.fuzzy = fuzzy
        

        
    def align_with_cube(self):
            lateral = self.sensors.get_cube_lateral_offset()
            if lateral is None:
                if self.DEBUG:
                    print("Nenhum cubo detectado para alinhar.")
                self.base.move(0, 0, 0)
                return False

            if abs(lateral) <= self.deadzone:
                self.base.move(0, 0, 0)
                if self.DEBUG:
                    print(f"Alinhado: lateral={lateral:.4f}")
                return True

            # módulo da velocidade lateral (igual para esquerda/direita) – P saturado
            speed_p = self.kp * abs(lateral)

            # tenta limitar usando fuzzy de distância frontal
            v_front = None
            speed = speed_p
            if self.fuzzy is not None:
                try:
                    dist = self.sensors.read_low_filtered()
                    if np.isfinite(dist):
                        v_front, _ = self.fuzzy.compute_velocity(dist)
                        speed = min(speed_p, v_front)
                except Exception:
                    speed = speed_p

            # aplica limite físico da base
            speed = min(self.max_vy, speed)

            # mesmo módulo para +lateral e -lateral; sinal só indica lado
            vy = math.copysign(speed, lateral)

            self.base.move(0, vy, 0)
            if not self.DEBUG:
                print(
                    f"[AlignFuzzyCap] lateral={lateral:.4f} "
                    f"speed_p={speed_p:.3f} v_front={v_front} "
                    f"speed={speed:.3f} → vy={vy:.3f}"
                )
            return False



    def auto_approach_cube(self):
            dist = self.sensors.read_low_filtered()
            if not np.isfinite(dist):
                if self.DEBUG:
                    print("Leitura inválida do lidar para abordagem.")
                self.base.move(0, 0, 0)
                return False

            err = dist - self.PICK_DISTANCE

            if abs(err) > self.PICK_TOL:
                lateral = self.sensors.get_cube_lateral_offset()
                if lateral is not None and abs(lateral) > self.deadzone:
                    if self.DEBUG:
                        print("Desalinhado: alinhar primeiro.")
                    self.base.move(0, 0, 0)
                    return False

                # P puro de distância
                vx_p = 0.9 * err

                # teto por forward_speed
                v_cap_const = self.forward_speed

                # teto fuzzy
                v_front = None
                v_cap_fuzzy = v_cap_const
                if self.fuzzy is not None:
                    try:
                        v_front, _ = self.fuzzy.compute_velocity(dist)
                        v_cap_fuzzy = min(v_cap_const, v_front)
                    except Exception:
                        v_cap_fuzzy = v_cap_const

                # aplica teto (constante + fuzzy)
                vx = max(-v_cap_fuzzy, min(v_cap_fuzzy, vx_p))

                # aplica velocidade mínima
                if abs(vx) < self.MIN_APPROACH_SPEED:
                    vx = math.copysign(self.MIN_APPROACH_SPEED, vx)

                self.base.move(vx, 0, 0)

                if not self.DEBUG:
                    print(
                        f"[ApproachFuzzyCap] dist={dist:.3f} "
                        f"err={err:.3f} vx_p={vx_p:.3f} "
                        f"v_front={v_front} cap_const={v_cap_const:.3f} "
                        f"cap_fuzzy={v_cap_fuzzy:.3f} vx={vx:.3f}"
                    )

                return False

            self.base.move(0, 0, 0)
            if self.DEBUG:
                print(f"Abordagem completa: distancia={dist:.4f}")
            return True
