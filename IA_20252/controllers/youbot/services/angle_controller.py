# controllers/youbot/youbot_auxiliar/AngleController.py
import math

class AngleController:
    def __init__(self, compass, base, dt):
        self.compass = compass
        self.base = base
        self.dt = dt
        self.rotating = False
        self.target_angle = None
        self.rotation_started = False
        self.rotation_direction = 0

    def get_current_angle(self):
        north = self.compass.getValues()
        angle_rad = math.atan2(north[0], north[1])
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        return angle_deg

    def angle_diff(self, target, current):
        d = (target - current + 180) % 360 - 180
        return d

    def rotate_to_angle(self, target_angle):
        self.target_angle = target_angle % 360
        self.rotating = True
        self.rotation_started = False
        
    def rotate_left_90(self):
        current = self.get_current_angle()
        self.rotate_to_angle((current - 90) % 360)

    def rotate_right_90(self):
        current = self.get_current_angle()
        self.rotate_to_angle((current + 90) % 360)

    def set_initial_angle(self, angle=None):
            """Set initial heading. If angle is None, sample current compass reading."""
            if angle is None:
                try:
                    angle = self.get_current_angle()
                except Exception:
                    angle = None
            self.initial_angle = angle

    def rotate_to_initial(self):
        if hasattr(self, "initial_angle") and self.initial_angle is not None:
            self.rotate_to_angle(self.initial_angle)
            
    def update_rotation(self):
        if not self.rotating:
            return

        current_angle = self.get_current_angle()
        diff = self.angle_diff(self.target_angle, current_angle)
        deadzone = 0.2
        max_speed = 1.0
        slow_zone = 15.0

        if not self.rotation_started:
            if diff == 0:
                self.rotation_direction = 0
            else:
                self.rotation_direction = -1 if diff > 0 else 1
            self.rotation_started = True
        print(current_angle)
        if abs(diff) <= deadzone:
            self.base.move(0, 0, 0)
            self.rotating = False
            self.rotation_started = False
            return

        speed = max_speed
        if abs(diff) < slow_zone:
            speed *= 0.3

        self.base.move(0, 0, speed * self.rotation_direction)