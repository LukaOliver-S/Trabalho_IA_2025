import numpy as np
class ObstacleAvoider:
    """
    Simple obstacle avoidance strategy:
    1. Choose strafe direction (left/right)
    2. Strafe up to MAX_STRAFES times
    3. Try forward nudge if lateral space opens
    4. Stop if path clears
    """
    
    # Default configuration
    MAX_STRAFES = 3
    STRAFE_DURATION = 0.35
    NUDGE_DURATION = 0.10
    MIN_SIDE_CLEARANCE = 0.12
    CENTER_ANGLE_THRESH = 0.06
    SETTLE_TIME = 0.03
    BACKUP_TIME = 0.30

    def __init__(self, base, lidar_low, lidar_high, sensors, step_wait, dt, 
                 obstacle_min_dist=None, debug=False, fuzzy=None):
        self.base = base
        self.lidar_low = lidar_low
        self.lidar_high = lidar_high
        self.sensors = sensors
        self.step_wait = step_wait
        self.dt = dt
        self.debug = debug
        self.fuzzy = fuzzy
        
        # Use provided obstacle_min_dist or default to 0.30
        self.obstacle_min_dist = obstacle_min_dist if obstacle_min_dist is not None else 0.30
        
        # State tracking
        self.cooldown = 0
        self.last_side = None

    def _log(self, msg):
        """Print debug message if debug enabled."""
        if self.debug:
            print(f"[Avoider] {msg}")

    def _compute_side_min(self, ranges, side):
        """Compute minimum distance on specified side (left/right)."""
        if ranges.size == 0 or not np.isfinite(ranges).any():
            return np.inf
        
        # Calculate angles for each point
        num = len(ranges)
        fov = self.lidar_low.getFov()
        angles = (np.arange(num) - num // 2) * (fov / num)
        
        # Select side mask
        if side == "left":
            mask = (angles > 0.05) & (angles < 1.5)
        else:  # right
            mask = (angles < -0.05) & (angles > -1.5)
        
        # Get valid readings
        valid = ranges[mask & np.isfinite(ranges) & (ranges > 0)]
        return float(np.min(valid)) if valid.size > 0 else np.inf

    def _get_side_distances(self, ranges):
        """Get minimum distances on left and right sides."""
        left, right = self.sensors.compute_side_mins(ranges)
        
        # Fallback to manual computation if needed
        if not np.isfinite(left):
            left = self._compute_side_min(ranges, "left")
        if not np.isfinite(right):
            right = self._compute_side_min(ranges, "right")
        
        return left, right

    def _choose_strafe_side(self, obstacle_angle, left_dist, right_dist):
        """Decide which side to strafe based on obstacle position and clearance."""
        # If obstacle is centered, strafe away from it
        if obstacle_angle is not None and abs(obstacle_angle) < self.CENTER_ANGLE_THRESH:
            return "left" if obstacle_angle > 0 else "right"
        
        # If both sides blocked, give up
        if not np.isfinite(left_dist) and not np.isfinite(right_dist):
            return None
        
        # Strafe toward the more open side
        if not np.isfinite(left_dist):
            return "right"
        if not np.isfinite(right_dist):
            return "left"
        
        return "right" if right_dist > left_dist else "left"

    def _execute_strafe(self, side):
        """Execute a single strafe movement."""
        if side == "left":
            self.base.strafe_left()
        else:
            self.base.strafe_right()
        
        if self.step_wait:
            self.step_wait(self.STRAFE_DURATION)
            self.step_wait(0.05)  # Brief settle

    def _execute_nudge(self, speed=0.03):
        """Execute a small forward nudge."""
        self.base.move(speed, 0, 0)
        if self.step_wait:
            self.step_wait(self.NUDGE_DURATION)
            self.step_wait(self.SETTLE_TIME)

    def start_avoid(self):
        """
        Main avoidance routine.
        
        Returns:
            bool: True if obstacle cleared, False otherwise
        """
        # Stop and settle
        self.base.move(0, 0, 0)
        if self.step_wait:
            self.step_wait(self.SETTLE_TIME)
        
        # Handle cooldown
        if self.cooldown > 0:
            self.cooldown = max(0, self.cooldown - 1)
            self._log(f"Cooldown: {self.cooldown}")
            return False
        
        # Get sensor readings
        low_scan, high_scan = self.sensors.snapshot()
        front_low, front_high = self.sensors.read_lidars()
        
        self._log(f"Start | front_low={front_low:.3f} front_high={front_high:.3f}")
        
        # Get obstacle angle from high lidar
        obstacle_angle = None
        if high_scan.size > 0 and np.isfinite(high_scan).any():
            obstacle_angle = self.sensors.get_high_min_angle()
        
        # Get side clearances
        left_dist, right_dist = self._get_side_distances(low_scan)
        self._log(f"Sides | left={left_dist:.3f} right={right_dist:.3f}")
        
        # Choose strafe direction
        side = self._choose_strafe_side(obstacle_angle, left_dist, right_dist)
        if side is None:
            self._log("No valid side to strafe")
            self.cooldown = 2
            return False
        
        self._log(f"Strafing {side}")
        
        # Try strafing with optional forward nudges
        for attempt in range(self.MAX_STRAFES):
            self._log(f"Attempt {attempt + 1}/{self.MAX_STRAFES}")
            
            # Execute strafe
            self._execute_strafe(side)
            
            # Check new front distance
            low_scan, _ = self.sensors.snapshot()
            front_dist, _ = self.sensors.read_lidars()
            
            # If front distance got worse, back off and abort
            if np.isfinite(front_dist) and front_dist < self.obstacle_min_dist:
                self._log(f"Front blocked: {front_dist:.3f} < {self.obstacle_min_dist:.3f}")
                self._execute_nudge(speed=-0.03)
                if self.step_wait:
                    self.step_wait(self.BACKUP_TIME)
                break
            
            # Success - path is clear
            if np.isfinite(front_dist) and front_dist >= self.obstacle_min_dist:
                self._log(f"Path clear: {front_dist:.3f} >= {self.obstacle_min_dist:.3f}")
                self._set_success(side)
                return True
            
            # Check if lateral space opened up
            left_dist, right_dist = self._get_side_distances(low_scan)
            side_dist = left_dist if side == "left" else right_dist
            
            if side_dist >= self.MIN_SIDE_CLEARANCE:
                self._log(f"Side opened ({side_dist:.3f}) -> trying nudge")
                self._execute_nudge()
                
                # Check if nudge cleared the path
                front_dist, _ = self.sensors.read_lidars()
                if np.isfinite(front_dist) and front_dist >= self.obstacle_min_dist:
                    self._log(f"Path clear after nudge: {front_dist:.3f}")
                    self._set_success(side)
                    return True
        
        # Failed to clear obstacle
        self._log("Failed to clear obstacle")
        self.cooldown = 2
        self.base.move(0, 0, 0)
        return False

    def _set_success(self, side):
        """Update state after successful avoidance."""
        self.last_side = side
        self.cooldown = max(1, int(0.6 / max(self.dt, 0.05)))
        self.base.move(0, 0, 0)
        self._log("Avoidance succeeded")