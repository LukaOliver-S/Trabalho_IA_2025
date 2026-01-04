import math
import numpy as np

class AlignmentController:
    def __init__(self, base, sensors, kp=1.5, max_vy=0.2,
                 deadzone=0.005, forward_speed=0.3,
                 pick_distance=0.104, pick_tol=0.005,
                 min_approach_speed=0.03, debug=False, fuzzy=None,
                 max_align_ticks=300, block_handler=None, max_capacity=5):
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
        
        # ===== CONTROLE DE CAPACIDADE =====
        self.block_handler = block_handler
        self.MAX_CAPACITY = max_capacity  # número máximo de poses na base
        
        # ===== TIMEOUT GERAL DE ALINHAMENTO =====
        self.max_align_ticks = max_align_ticks
        self.align_ticks = 0
        self.alignment_failed = False
        self.lost_cube_ticks = 0
        self.max_lost_cube_ticks = 50

    def reset_alignment(self):
        """Reseta contador e flag"""
        self.align_ticks = 0
        self.alignment_failed = False
        self.lost_cube_ticks = 0

    def _check_capacity(self):
        """Verifica se ainda há espaço na base do robô"""
        if self.block_handler is None:
            return True  # Se não tiver handler, assume que tem espaço
        
        if self.block_handler.counter >= self.MAX_CAPACITY:
            self.alignment_failed = True
            self.base.move(0, 0, 0)
            if self.DEBUG:
                print(f"⛔ Base cheia: {self.block_handler.counter}/{self.MAX_CAPACITY} — desistindo de alinhar")
            return False
        return True

    def align_with_cube(self):
        # ===== VERIFICA CAPACIDADE =====
        if not self._check_capacity():
            return False
        
        lateral = self.sensors.get_cube_lateral_offset()
        
        # ===== SEM CUBO DETECTADO =====
        if lateral is None:
            if self.align_ticks > 0:
                self.align_ticks += 1
                self.lost_cube_ticks += 1
                
                if self.lost_cube_ticks > self.max_lost_cube_ticks:
                    self.alignment_failed = True
                    self.base.move(0, 0, 0)
                    if self.DEBUG:
                        print(f"⛔ Cubo perdido por {self.lost_cube_ticks} ticks — desistindo")
                    return False
                
                if self.align_ticks > self.max_align_ticks:
                    self.alignment_failed = True
                    self.base.move(0, 0, 0)
                    if self.DEBUG:
                        print(f"⛔ Timeout geral: {self.align_ticks} ticks")
                    return False
                
                if self.DEBUG:
                    print(f"⚠️ Cubo não visível (lost={self.lost_cube_ticks}, total={self.align_ticks})")
            else:
                if self.DEBUG:
                    print("Nenhum cubo detectado para alinhar.")
            
            self.base.move(0, 0, 0)
            return False

        self.lost_cube_ticks = 0
        self.align_ticks += 1

        if self.align_ticks > self.max_align_ticks:
            self.alignment_failed = True
            self.base.move(0, 0, 0)
            if self.DEBUG:
                print(f"⛔ Timeout geral: {self.align_ticks} ticks")
            return False

        if abs(lateral) <= self.deadzone:
            self.base.move(0, 0, 0)
            if self.DEBUG:
                print(f"✅ Alinhado lateralmente: lateral={lateral:.4f}")
            return True

        speed_p = self.kp * abs(lateral)

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

        speed = min(self.max_vy, speed)
        vy = math.copysign(speed, lateral)

        self.base.move(0, vy, 0)
        
        if  self.DEBUG:
            print(
                f"[Align] tick={self.align_ticks}/{self.max_align_ticks} "
                f"lateral={lateral:.4f} → vy={vy:.3f}"
            )
        return False

    def auto_approach_cube(self):
        # ===== VERIFICA CAPACIDADE =====
        if not self._check_capacity():
            return False
        
        dist = self.sensors.read_low_filtered()
        
        if not np.isfinite(dist):
            if self.DEBUG:
                print("Leitura inválida do lidar para abordagem.")
            self.base.move(0, 0, 0)
            return False

        self.align_ticks += 1

        if self.align_ticks > self.max_align_ticks:
            self.alignment_failed = True
            self.base.move(0, 0, 0)
            if self.DEBUG:
                print(f"⛔ Timeout geral: {self.align_ticks} ticks")
            return False

        err = dist - self.PICK_DISTANCE

        if abs(err) <= self.PICK_TOL:
            self.base.move(0, 0, 0)
            if self.DEBUG:
                print(f"✅ Approach completo: distância={dist:.4f}")
            self.reset_alignment()
            return True

        lateral = self.sensors.get_cube_lateral_offset()
        if lateral is not None and abs(lateral) > self.deadzone:
            if self.DEBUG:
                print("⚠️ Desalinhado — voltar para align")
            self.base.move(0, 0, 0)
            return False

        vx_p = 0.9 * err
        v_cap_const = self.forward_speed

        v_front = None
        v_cap_fuzzy = v_cap_const
        if self.fuzzy is not None:
            try:
                v_front, _ = self.fuzzy.compute_velocity(dist)
                v_cap_fuzzy = min(v_cap_const, v_front)
            except Exception:
                v_cap_fuzzy = v_cap_const

        vx = max(-v_cap_fuzzy, min(v_cap_fuzzy, vx_p))

        if abs(vx) < self.MIN_APPROACH_SPEED:
            vx = math.copysign(self.MIN_APPROACH_SPEED, vx)

        self.base.move(vx, 0, 0)

        if  self.DEBUG:
            print(
                f"[Approach] tick={self.align_ticks}/{self.max_align_ticks} "
                f"dist={dist:.3f} → vx={vx:.3f}"
            )

        return False