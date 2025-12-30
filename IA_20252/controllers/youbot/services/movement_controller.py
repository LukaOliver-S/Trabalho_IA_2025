import numpy as np
from typing import Callable, Optional


class MovementController:
    """Controls robot movement with fuzzy speed adaptation."""

    def __init__(
        self,
        base,
        step_wait: Optional[Callable[[float], None]] = None,
        compass=None,
        forward_speed: float = 0.1,
        strafe_speed: float = 0.1,
        movement_duration: int = 10,
        fuzzy=None,
        front_dist_fn: Optional[Callable[[], float]] = None,
        debug: bool = True,
    ):
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

    def _compute_speed(self) -> tuple[float, float]:
        """
        Compute (vx_speed, vy_speed) using fuzzy controller if available.
        Returns fallback speeds if fuzzy unavailable or fails.
        """
        # No fuzzy controller -> use nominal speeds
        if self.fuzzy is None:
            return self.forward_speed, self.strafe_speed
        
        # Get front distance
        dist = self._get_front_distance()
        
        # If invalid, use LIDAR_MAX as fallback
        if dist is None:
            lidar_max = getattr(self.fuzzy, "LIDAR_MAX", None)
            if lidar_max is None:
                if self.debug:
                    print("[Movement] No valid distance, no LIDAR_MAX -> fallback speed")
                return self.forward_speed, self.strafe_speed
            dist = float(lidar_max)
            if self.debug:
                print(f"[Movement] Invalid distance -> using LIDAR_MAX={dist:.3f}")
        
        # Clip to valid lidar range
        lidar_max = getattr(self.fuzzy, "LIDAR_MAX", None)
        if lidar_max is not None:
            dist = np.clip(dist, 0.0, float(lidar_max))
        
        # Compute fuzzy velocities
        try:
            vx, vy = self.fuzzy.compute_velocity(dist)
            
            # Get fuzzy universe limits (or use nominal as fallback)
            try:
                v_max = float(max(self.fuzzy.v.universe))
            except Exception:
                v_max = max(self.forward_speed, abs(vx), abs(vy))
            
            # Clip to valid range
            vx_speed = np.clip(abs(vx), 0.0, v_max)
            vy_speed = np.clip(abs(vy), 0.0, v_max)
            
            if self.debug:
                print(f"[Movement] dist={dist:.3f} -> fuzzy(vx={vx:.3f}, vy={vy:.3f}) "
                      f"-> use({vx_speed:.3f}, {vy_speed:.3f})")
            
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
        
        # Compute adaptive speeds
        forward_speed, lateral_speed = self._compute_speed()
        
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
