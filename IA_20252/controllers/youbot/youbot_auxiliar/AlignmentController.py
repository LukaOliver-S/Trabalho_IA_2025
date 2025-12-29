# controllers/youbot/youbot_auxiliar/AlignmentController.py
import math
import numpy as np

class AlignmentController:
    def __init__(self, base, sensors, step_wait, kp=1.5, max_vy=0.06,
                 deadzone=0.005, forward_speed=0.3, pick_distance=0.104,
                 pick_tol=0.005, min_approach_speed=0.03, debug=False):
        self.base = base
        self.sensors = sensors
        self._step_wait = step_wait
        self.kp = kp
        self.max_vy = max_vy
        self.deadzone = deadzone
        self.forward_speed = forward_speed
        self.PICK_DISTANCE = pick_distance
        self.PICK_TOL = pick_tol
        self.MIN_APPROACH_SPEED = min_approach_speed
        self.DEBUG = debug

        # tuning
        self.PICK_STRICT_TOL = 0.001
        self.NUDGE_SPEED = max(self.MIN_APPROACH_SPEED, 0.03)
        self.NUDGE_DUR = 0.06

    def get_cube_lateral_offset(self):
        return self.sensors.get_cube_lateral_offset()

    def read_low_filtered(self, samples=3, settle=0.03):
        return self.sensors.read_low_filtered(samples, settle)

    def align_with_cube(self):
        lateral = self.get_cube_lateral_offset()
        if lateral is None:
            if self.DEBUG: print("Nenhum cubo detectado para alinhar.")
            self.base.move(0, 0, 0)
            return False

        if abs(lateral) <= self.deadzone:
            self.base.move(0, 0, 0)
            if self.DEBUG: print(f"[ALINHADO] lateral={lateral:.4f} m")
            return True

        vy = self.kp * lateral
        vy = max(-self.max_vy, min(self.max_vy, vy))
        self.base.move(0, vy, 0)
        if self.DEBUG: print(f"[ALINHANDO] lateral={lateral:.4f} → vy={vy:.3f}")
        return False

    def auto_approach_cube(self):
        dist = self.read_low_filtered()
        if not np.isfinite(dist):
            if self.DEBUG: print("Leitura inválida do lidar para abordagem.")
            self.base.move(0, 0, 0)
            return False

        err = dist - self.PICK_DISTANCE

        if abs(err) > self.PICK_TOL:
            lateral = self.get_cube_lateral_offset()
            if lateral is not None and abs(lateral) > (self.deadzone + 0.004):
                if self.DEBUG: print(f"[ABORDAGEM] desalinhado lateralmente ({lateral:.4f}) → primeiro alinhar")
                self.base.move(0, 0, 0)
                return False

            kp = 0.9
            vx = kp * err
            vx = max(-self.forward_speed, min(self.forward_speed, vx))
            if abs(vx) < self.MIN_APPROACH_SPEED:
                vx = math.copysign(self.MIN_APPROACH_SPEED, vx)

            self.base.move(vx, 0, 0)
            if self.DEBUG: print(f"[ABORDAGEM] distancia={dist:.4f} err={err:.4f} vx={vx:.3f}")
            return False

        if abs(err) <= self.PICK_STRICT_TOL:
            self.base.move(0, 0, 0)
            if self.DEBUG: print(f"[ABORDAGEM] preciso: {dist:.4f} → pronto")
            return True

        vx = math.copysign(self.NUDGE_SPEED, err)
        if self.DEBUG: print(f"[FINALIZANDO] nudge vx={vx:.3f} dur={self.NUDGE_DUR}s (err={err:.4f})")
        self.base.move(vx, 0, 0)
        if self._step_wait: self._step_wait(self.NUDGE_DUR)
        self.base.move(0, 0, 0)
        if self._step_wait: self._step_wait(0.03)

        new_dist = self.read_low_filtered()
        if np.isfinite(new_dist) and abs(new_dist - self.PICK_DISTANCE) <= self.PICK_STRICT_TOL:
            if self.DEBUG: print(f"[ABORDAGEM] pós-nudge distância={new_dist:.4f} → pronto para descida")
            return True

        if self.DEBUG: print(f"[ABORDAGEM] pós-nudge distância={new_dist:.4f} → ainda não pronto")
        return False