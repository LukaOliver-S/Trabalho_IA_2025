import numpy as np
from typing import Callable, Optional


class MovementController:
    """Controls robot movement with fuzzy speed adaptation."""

    def __init__(self, base, step_wait: Optional[Callable[[float], None]] = None,
             compass=None, forward_speed: float = 0.1, strafe_speed: float = 0.1,
             movement_duration: int = 10, fuzzy=None, front_dist_fn: Optional[Callable[[], float]] = None,
             side_dist_fn: Optional[Callable[[], tuple]] = None,
             debug: bool = False):
        self.base = base
        self._step_wait = step_wait
        self.compass = compass
        
        # Nominal speeds (fallback values)
        self.forward_speed = forward_speed
        self.strafe_speed = strafe_speed
        self.movement_duration = movement_duration
        
        # Fuzzy integration
        self.fuzzy = fuzzy
        self.front_dist_fn = front_dist_fn
        self.side_dist_fn = side_dist_fn
            
        # State flags
        self.block_forward = False  # Set by obstacle avoidance
        self.is_picking = False     # Set during pick action
        
        # Movement counters
        self.counters = {
            "forward": 0,
            "back": 0,
            "left": 0,
            "right": 0
        }
        
        self.debug = debug

    def set_front_dist_fn(self, fn: Callable[[], float]):
        """Update front distance sensor function."""
        self.front_dist_fn = fn
        
    def set_side_dist_fn(self, fn: Callable[[], tuple]):
        self.side_dist_fn = fn
        
    # ========================================================================
    # MOVEMENT COMMANDS
    # ========================================================================

    def forward(self, ticks: Optional[int] = None):
        """Move forward for specified ticks (or default duration)."""
        self.counters["forward"] = ticks if ticks is not None else self.movement_duration

    def backward(self, ticks: Optional[int] = None):
        """Move backward for specified ticks."""
        self.counters["back"] = ticks if ticks is not None else self.movement_duration

    def strafe_left(self, ticks: Optional[int] = None):
        """Strafe left for specified ticks."""
        self.counters["left"] = ticks if ticks is not None else self.movement_duration

    def strafe_right(self, ticks: Optional[int] = None):
        """Strafe right for specified ticks."""
        self.counters["right"] = ticks if ticks is not None else self.movement_duration

    def stop_all(self):
        """Clear all counters and stop the base."""
        for key in self.counters:
            self.counters[key] = 0
        self.base.move(0, 0, 0)

    def immediate_stop(self, wait: float = 0.03):
        """Stop immediately and wait."""
        self.stop_all()
        if self._step_wait:
            self._step_wait(wait)

    # ========================================================================
    # SPEED COMPUTATION (FUZZY INTEGRATION)
    # ========================================================================

    def _get_front_distance(self) -> Optional[float]:
        """Safely read front distance sensor."""
        if self.front_dist_fn is None:
            return None
        
        try:
            d = float(self.front_dist_fn())
            return d if np.isfinite(d) else None
        except Exception:
            return None
        
    def _get_side_distances(self):
        """Get minimum distances on left and right sides."""
        # Prefer explicit side-lidar readings if provided by SensorSuite
        if hasattr(self.sensors, "read_side_distances"):
                left, right = self.sensors.read_side_distances()
                if np.isfinite(left) or np.isfinite(right):
                    return left, right
           

    def _compute_speed(self, side_hint: Optional[str] = None) -> tuple[float, float]:
        """
        Compute vx from frontal distance and vy from the chosen side distance using
        the same fuzzy (keeps FuzzySimple unchanged).
        side_hint: 'left' or 'right' when a lateral move is requested (optional).
        """
        # front distance with fallback
        front = self._get_front_distance()
        if front is None or not np.isfinite(front):
            front = float(getattr(self.fuzzy, "LIDAR_MAX", 0.5))

        # get safe side distances
        left = right = float(getattr(self.fuzzy, "LIDAR_MAX", 0.5))
        if getattr(self, "side_dist_fn", None) is not None:
            try:
                l, r = self.side_dist_fn()
                left = float(l) if np.isfinite(l) else left
                right = float(r) if np.isfinite(r) else right
            except Exception:
                pass

        # choose lateral input
        if side_hint == "left":
            side_input = left
        elif side_hint == "right":
            side_input = right
        else:
            side_input = min(left, right)

        # compute fuzzy outputs (reuse same fuzzy)
        try:
            vx_val, _ = self.fuzzy.compute_velocity(front)
            _, vy_val = self.fuzzy.compute_velocity(side_input)

            try:
                v_max = float(max(self.fuzzy.v.universe))
            except Exception:
                v_max = max(self.forward_speed, abs(vx_val), abs(vy_val))

            vx_speed = np.clip(abs(vx_val), 0.0, v_max)
            vy_speed = np.clip(abs(vy_val), 0.0, v_max)

            if self.debug:
                print(f"[Movement] front={front:.3f} side_in={side_input:.3f} "
                    f"-> fuzzy(vx={vx_val:.3f}, vy={vy_val:.3f}) -> use({vx_speed:.3f}, {vy_speed:.3f})")

            return vx_speed, vy_speed

        except Exception as e:
            if self.debug:
                print(f"[Movement] Fuzzy compute failed: {e} -> fallback speed")
            return self.forward_speed, self.strafe_speed

    # ========================================================================
    # MAIN UPDATE LOOP
    # ========================================================================

    def update(self):
        """
        Main control loop update.
        Reads counters, computes velocities, and issues base.move() command.
        """
        # During picking, freeze base movement
        if self.is_picking:
            self.base.move(0, 0, 0)
            return
        
        # If forward motion blocked, clear forward/back counters
        if self.block_forward:
            self.counters["forward"] = 0
            self.counters["back"] = 0
            
            # If no lateral motion either, stop completely
            if self.counters["left"] == 0 and self.counters["right"] == 0:
                self.base.move(0, 0, 0)
                return
        
        # determine lateral hint
        side_hint = None
        if self.counters["left"] > 0:
            side_hint = "left"
        elif self.counters["right"] > 0:
            side_hint = "right"

        forward_speed, lateral_speed = self._compute_speed(side_hint=side_hint)
                
        # Determine velocities from counters
        vx = 0.0
        if self.counters["forward"] > 0:
            vx = forward_speed
        elif self.counters["back"] > 0:
            vx = -forward_speed
        
        vy = 0.0
        if self.counters["left"] > 0:
            vy = lateral_speed
        elif self.counters["right"] > 0:
            vy = -lateral_speed
        
        # Decrement active counters
        for key in self.counters:
            if self.counters[key] > 0:
                self.counters[key] -= 1
        
        # Issue movement command
        self.base.move(vx, vy, 0)
