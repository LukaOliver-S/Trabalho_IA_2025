import numpy as np

class ObstacleAvoider:
    """
    Minimal avoider:
    - tries up to MAX_STRAFES strafes on a selected side
    - small forward nudge if lateral space opens
    - keeps minimal persistent state (cooldown, last side)
    """

    def __init__(self, base, lidar_low, lidar_high, sensors, step_wait, dt, debug=False,fuzzy = None):
        self.base = base
        self.lidar_low = lidar_low
        self.lidar_high = lidar_high
        self.sensors = sensors
        self._step_wait = step_wait
        self.dt = dt
        self.DEBUG = debug
        self.fuzzy = fuzzy
        # minimal persistent state
        self.avoid_cooldown = 0
        self.last_avoid_side = None

    def _dbg(self, msg):
        if self.DEBUG: print(msg)

    def _side_min_fallback(self, ranges, side, min_angle=0.05, max_angle=1.5):
        if ranges.size == 0 or not np.isfinite(ranges).any():
            return float("inf")
        num = len(ranges)
        fov = self.lidar_low.getFov()
        angles = (np.arange(num) - num // 2) * (fov / num)
        if side == "left":
            mask = (angles > min_angle) & (angles < max_angle)
        else:
            mask = (angles < -min_angle) & (angles > -max_angle)
        vals = ranges[mask & np.isfinite(ranges) & (ranges > 0)]
        return float(np.min(vals)) if vals.size else float("inf")

    def _get_side_mins(self, ranges):
        left, right = self.sensors.compute_side_mins(ranges)
        if not np.isfinite(left):
            left = self._side_min_fallback(ranges, "left")
        if not np.isfinite(right):
            right = self._side_min_fallback(ranges, "right")
        return left, right

    def _choose_side(self, angle_high, left_min, right_min):
        # center threshold hard-coded (simple)
        CENTER_ANGLE_TH = getattr(self, "CENTER_ANGLE_TH", 0.06)
        if angle_high is not None and abs(angle_high) < CENTER_ANGLE_TH:
            return "right" if angle_high < 0 else "left"
        if not np.isfinite(left_min) and not np.isfinite(right_min):
            return None
        if not np.isfinite(left_min): return "right"
        if not np.isfinite(right_min): return "left"
        return "left" if left_min < right_min else "right"

    def _strafe_once(self, side, dur):
        if side == "left":
            self.base.strafe_left()
        else:
            self.base.strafe_right()
        if self._step_wait:
            self._step_wait(dur)
            self._step_wait(0.05)

    def start_avoid(self):
        # only the constants we actually use
        STRAFE_DUR = 0.35
        MAX_STRAFES = 3
        SIDE_MIN = 0.12
        NUDGE_DUR = 0.10
        MIN_CLEAR = getattr(self, "OBSTACLE_MIN_DIST", 0.30)
        EPS_FRONT = 0.005

        # stop and settle
        self.base.move(0, 0, 0)
        if self._step_wait: self._step_wait(0.03)

        # cooldown
        if self.avoid_cooldown > 0:
            self.avoid_cooldown = max(0, self.avoid_cooldown - 1)
            self._dbg(f"avoid cooldown {self.avoid_cooldown}")

        # single snapshot and initial readings
        low_snapshot, high_snapshot = self.sensors.snapshot()
        min_low, min_high = self.sensors.read_lidars()
        self._dbg(f"avoid start | low={min_low:.3f} high={min_high:.3f}")

        angle_high = self.sensors.get_high_min_angle() if (high_snapshot.size and np.isfinite(high_snapshot).any()) else None
        left_min, right_min = self._get_side_mins(low_snapshot)
        self._dbg(f"left={left_min:.3f} right={right_min:.3f}")
          
        side = self._choose_side(angle_high, left_min, right_min)
        if side is None:
            self.avoid_cooldown = 2
            return False
        self._dbg(f"trying side: {side}")

        freed = False
        for i in range(MAX_STRAFES):
            self._dbg(f"strafe {side} {i+1}/{MAX_STRAFES}")
            self._strafe_once(side, STRAFE_DUR)

            # fresh snapshot and front reading
            low_snapshot, _ = self.sensors.snapshot()
            new_front, _ = self.sensors.read_lidars()

            # abort if front got worse
            if np.isfinite(new_front) and new_front < MIN_CLEAR - EPS_FRONT:
                self._dbg(f"front worse {new_front:.3f} -> abort")
                self.base.move(-0.03, 0, 0)
                if self._step_wait: self._step_wait(0.30)
                break

            # success if cleared
            if np.isfinite(new_front) and new_front >= MIN_CLEAR - EPS_FRONT:
                freed = True
                self._dbg("freed after strafe")
                break

            # if side improved, try small forward nudge
            cur_left, cur_right = self._get_side_mins(low_snapshot)
            lateral_ok = (side == "left" and cur_left >= SIDE_MIN) or (side == "right" and cur_right >= SIDE_MIN)
            if lateral_ok:
                self._dbg("lateral ok -> forward nudge")
                self.base.move(0.03, 0, 0)
                if self._step_wait: self._step_wait(NUDGE_DUR)
                if self._step_wait: self._step_wait(0.03)
                after_nudge, _ = self.sensors.read_lidars()
                if np.isfinite(after_nudge) and after_nudge >= MIN_CLEAR - EPS_FRONT:
                    freed = True
                    self._dbg("freed after nudge")
                    break

        # final state updates
        if freed:
            self.avoid_cooldown = max(1, int(0.6 / (self.dt if self.dt > 0 else 0.05)))
            self.last_avoid_side = side
            self._dbg("avoid succeeded")
        else:
            self.avoid_cooldown = 2
            self._dbg("avoid failed - cooldown set")

        self.base.move(0, 0, 0)
        return freed