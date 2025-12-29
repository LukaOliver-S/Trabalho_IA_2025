# controllers/youbot/youbot_auxiliar/ObstacleAvoider.py
import numpy as np

class ObstacleAvoider:
    """
    Encapsula a lógica de avoid_obstacle() original.
    Uso:
      avoider = ObstacleAvoider(base, lidar_low, lidar_high, sensors, step_wait, dt, debug=True)
      freed = avoider.start_avoid()
      # depois pode inspecionar avoider.avoid_cooldown, avoider.avoid_attempts, etc.
    """
    def __init__(self, base, lidar_low, lidar_high, sensors, step_wait, dt, debug=False):
        self.base = base
        self.lidar_low = lidar_low
        self.lidar_high = lidar_high
        self.sensors = sensors
        self._step_wait = step_wait
        self.dt = dt
        self.DEBUG = debug

        # state (persistent between chamadas)
        self.avoid_cooldown = 0
        self.avoid_attempts = 0
        self.last_avoid_side = None
        self.recently_freed = 0
        self.RECENTLY_FREED_TICKS = max(1, int(0.6 / (dt if dt > 0 else 0.1)))

    def can_attempt(self):
        """Se estiver em cooldown e não houver override, devolve False."""
        return self.avoid_cooldown == 0 and self.recently_freed == 0

    def start_avoid(self):
        """
        Blocking routine: tenta liberar a frente com strafes e nudges.
        Retorna True se liberado (freed), False caso contrário.
        Também atualiza self.avoid_cooldown, self.avoid_attempts, self.last_avoid_side, self.recently_freed.
        """
        # Config local (copiado da implementação original para manter comportamento)
        STRAFE_DUR = 0.45
        MAX_STRAFES = 3
        SIDE_MIN = getattr(self, "SIDE_MIN", 0.12)
        MIN_CLEAR_FOR_FRONT = getattr(self, "OBSTACLE_MIN_DIST", 0.30) + 0.01
        CENTER_ANGLE_TH = getattr(self, "CENTER_ANGLE_TH", 0.06)
        HYST = 0.06
        NUDGE_DUR = 0.10
        EPS = 0.005

        # Cancel any forward command immediately (controller should also stop moving)
        self.base.move(0, 0, 0)
        if self._step_wait: self._step_wait(0.03)

        # Cooldown handling (decrement, but allow forced entry if obstacle still present)
        if self.avoid_cooldown > 0:
            self.avoid_cooldown = max(0, self.avoid_cooldown - 1)
            if self.DEBUG:
                print(f"⏳ avoid cooldown ({self.avoid_cooldown})")

        if self.recently_freed > 0:
            self.recently_freed = max(0, self.recently_freed - 1)
            if self.DEBUG:
                print(f"⏳ recently_freed ({self.recently_freed})")

        # Leitura inicial
        min_low, min_high = self.sensors.read_lidars()
        if self.DEBUG:
            print(f"🔍 avoid_obstacle start | min_low={min_low:.3f} min_high={min_high:.3f}")

        # tenta obter ângulo do mínimo do lidar alto (preferencial)
        angle_high = None
        try:
            ranges_high = np.array(self.lidar_high.getRangeImage(), dtype=np.float32)
            if ranges_high.size and np.isfinite(ranges_high).any():
                valid_h = (np.isfinite(ranges_high) & (ranges_high > 0))
                if valid_h.any():
                    # reutiliza helper do sensors (se disponível)
                    angle_high = self.sensors.get_high_min_angle()
        except Exception:
            angle_high = None

        # laterais
        ranges_low = np.array(self.lidar_low.getRangeImage(), dtype=np.float32)
        left_min, right_min = self.sensors.compute_side_mins(ranges_low)
        if self.DEBUG:
            print(f"📐 angle_high={angle_high} left={left_min:.3f} right={right_min:.3f}")

        # Escolhe o lado a tentar
        if angle_high is not None and abs(angle_high) < CENTER_ANGLE_TH:
            chosen_side = "right" if angle_high < 0 else "left"
        else:
            chosen_side = "left" if left_min < right_min else "right"
            if self.last_avoid_side and abs(left_min - right_min) <= HYST:
                chosen_side = self.last_avoid_side

        if self.DEBUG:
            print(f"⚠️ tentando apenas o lado: {chosen_side}")

        side_before = left_min if chosen_side == "left" else right_min
        freed = False

        for s in range(MAX_STRAFES):
            if chosen_side == "left":
                if self.DEBUG: print(f"↪️ Strafe {s+1}/{MAX_STRAFES} → left")
                # usa os comandos do base
                self.base.strafe_left()
            else:
                if self.DEBUG: print(f"↪️ Strafe {s+1}/{MAX_STRAFES} → right")
                self.base.strafe_right()

            # espera a strafing
            if self._step_wait: self._step_wait(STRAFE_DUR)
            if self._step_wait: self._step_wait(0.05)

            ranges = np.array(self.lidar_low.getRangeImage(), dtype=np.float32)
            cur_left, cur_right = self.sensors.compute_side_mins(ranges)
            new_front, _ = self.sensors.read_lidars()
            cur_side = cur_left if chosen_side == "left" else cur_right

            # frente piorou -> abort
            if np.isfinite(new_front) and new_front < MIN_CLEAR_FOR_FRONT - 0.01:
                if self.DEBUG: print(f"🚫 Frente piorou durante strafe (front {new_front:.3f}) → abortando {chosen_side} + recuando")
                if self._step_wait: self._step_wait(0.30)
                self.avoid_attempts += 1
                self.avoid_cooldown = max(1, int(0.5 / (self.dt if self.dt>0 else 0.05)))
                # sinaliza que ainda existe obstáculo
                freed = False
                break

            # lateral piorou -> abort
            if np.isfinite(side_before) and np.isfinite(cur_side) and (cur_side + EPS) < side_before:
                if self.DEBUG: print(f"🚫 Lateral piorou (antes {side_before:.3f} agora {cur_side:.3f}) → abortando {chosen_side} + recuando")
                # recua um pouco
                self.base.move(-0.03, 0, 0)
                if self._step_wait: self._step_wait(0.30)
                self.avoid_attempts += 1
                self.avoid_cooldown = max(1, int(0.5 / (self.dt if self.dt>0 else 0.05)))
                freed = False
                break

            # frente abriu -> sucesso
            if np.isfinite(new_front) and new_front >= MIN_CLEAR_FOR_FRONT:
                freed = True
                if self.DEBUG: print("✅ Liberado (frente) após strafe")
                break

            # lateral melhorou -> nudge frontal para testar
            lateral_ok = (chosen_side == "left" and cur_left >= SIDE_MIN) or (chosen_side == "right" and cur_right >= SIDE_MIN)
            if lateral_ok:
                if self.DEBUG: print("ℹ️ lateral melhorou → nudge frontal para testar")
                # pequeno nudge frontal
                self.base.move(0.03, 0, 0)
                if self._step_wait: self._step_wait(NUDGE_DUR)
                if self._step_wait: self._step_wait(0.03)
                after_nudge_low, _ = self.sensors.read_lidars()
                if np.isfinite(after_nudge_low) and after_nudge_low >= MIN_CLEAR_FOR_FRONT:
                    freed = True
                    if self.DEBUG: print("✅ Liberado após nudge frontal")
                    break

            side_before = cur_side

        # Resultado: atualiza estados persistentes
        if freed:
            self.avoid_cooldown = max(1, int(0.6 / (self.dt if self.dt>0 else 0.05)))
            self.recently_freed = self.RECENTLY_FREED_TICKS
            self.last_avoid_side = chosen_side
            self.avoid_attempts = 0
            if self.DEBUG: print("🔓 avoid result: freed")
        else:
            if self.avoid_attempts >= 2:
                self.avoid_cooldown = max(1, int(0.8 / (self.dt if self.dt>0 else 0.05)))
                self.avoid_attempts = 0
                if self.DEBUG: print("⏳ cooldown maior após tentativas fracassadas")

        # garante parada no final
        self.base.move(0, 0, 0)
        return freed