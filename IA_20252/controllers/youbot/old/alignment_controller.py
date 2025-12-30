import math
import numpy as np

class AlignmentController:
    """
    Alignment and approach helper for the cube:
    - align_with_cube(): lateral alignment (vy)
    - auto_approach_cube(): controlled forward approach (vx) with smooth slow-down
    """

    def __init__(self, base, sensors, kp=1.5, max_vy=0.06,
                 deadzone=0.005, forward_speed=0.3,
                 pick_distance=0.104, pick_tol=0.005,
                 min_approach_speed=0.03, slow_range=0.12,
                 debug=False, fuzzy=None):
        self.base = base
        self.sensors = sensors
        self.kp = kp
        self.max_vy = max_vy
        self.deadzone = deadzone
        self.forward_speed = float(forward_speed)
        self.PICK_DISTANCE = float(pick_distance)
        self.PICK_TOL = float(pick_tol)
        self.MIN_APPROACH_SPEED = float(min_approach_speed)
        self.SLOW_RANGE = float(slow_range)
        self.DEBUG = debug
        self.fuzzy = fuzzy

    def align_with_cube(self):
        lateral = self.sensors.get_cube_lateral_offset()
        if lateral is None:
            if self.DEBUG: print("Nenhum cubo detectado para alinhar.")
            self.base.move(0, 0, 0)
            return False

        if abs(lateral) <= self.deadzone:
            self.base.move(0, 0, 0)
            if self.DEBUG: print(f"Alinhado: lateral={lateral:.4f}")
            return True

        vy = max(-self.max_vy, min(self.max_vy, self.kp * lateral))
        self.base.move(0, vy, 0)
        if self.DEBUG: print(f"Alinhando: lateral={lateral:.4f} → vy={vy:.3f}")
        return False

    def auto_approach_cube(self):
        dist = self.sensors.read_low_filtered()
        if not np.isfinite(dist):
            if self.DEBUG: print("Leitura inválida do lidar para abordagem.")
            self.base.move(0, 0, 0)
            return False

        err = dist - self.PICK_DISTANCE

        # Already at target?
        if abs(err) <= self.PICK_TOL:
            self.base.move(0, 0, 0)
            if self.DEBUG: print(f"Abordagem completa: distancia={dist:.4f}")
            return True

        # If misaligned, require alignment first
        lateral = self.sensors.get_cube_lateral_offset()
        if lateral is not None and abs(lateral) > self.deadzone:
            if self.DEBUG: print("Desalinhado: alinhar primeiro.")
            self.base.move(0, 0, 0)
            return False

        # Smooth slow-down when close to the object
        abs_err = abs(err)
        if abs_err < self.SLOW_RANGE and self.SLOW_RANGE > 0:
            # Linear scaling from forward_speed down to MIN_APPROACH_SPEED
            frac = abs_err / self.SLOW_RANGE
            target_speed = max(self.MIN_APPROACH_SPEED, self.forward_speed * frac)
            vx = math.copysign(target_speed, err)
            if self.DEBUG: print(f"Slow-down (near): dist={dist:.4f} frac={frac:.3f} vx={vx:.3f}")
        else:
            # Far: proportional command, clipped
            vx = 0.9 * err
            vx = max(-self.forward_speed, min(self.forward_speed, vx))
            if abs(vx) < self.MIN_APPROACH_SPEED:
                vx = math.copysign(self.MIN_APPROACH_SPEED, vx)
            if self.DEBUG: print(f"Approach (far): dist={dist:.4f} err={err:.4f} vx={vx:.3f}")

        # Move forward/back as computed
        self.base.move(vx, 0, 0)
        if self.DEBUG: print(f"Abordando: distancia={dist:.4f} err={err:.4f} vx={vx:.3f}")
        return False