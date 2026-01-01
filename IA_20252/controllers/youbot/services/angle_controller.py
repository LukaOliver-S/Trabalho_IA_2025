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
        """Diferença mínima entre dois ângulos (±180°)."""
        d = (target - current + 180) % 360 - 180
        return d

    def start_rotation(self, target_angle):
        """Inicia a rotação não-bloqueante."""
        self.target_angle = target_angle % 360
        self.rotating = True


    def rotate_to_initial(self):
        """Gira até o ângulo inicial definido."""
        if self.rotating:
            return
        self.rotate_to_angle(self.initial_angle)

    def rotate_to_angle(self, target_angle):
        """Inicia a rotação em direção a target_angle (não-bloqueante)."""
        self.target_angle = target_angle % 360
        self.rotating = True
        self.rotation_started = False  

    def rotate_left_90(self):
        """Gira 90° para a esquerda a partir do ângulo atual."""
        if self.rotating:
            return
        current = self.get_current_angle()
        self.rotate_to_angle((current - 90) % 360)

    def rotate_right_90(self):
        """Gira 90° para a direita a partir do ângulo atual."""
        if self.rotating:
            return
        current = self.get_current_angle()
        self.rotate_to_angle((current + 90) % 360)

    def update_rotation(self):
        if not self.rotating:
            return

        current_angle = self.get_current_angle()
        diff = self.angle_diff(self.target_angle, current_angle)

        # params
        deadzone = 0.5      
        max_speed = 1.0     
        min_speed = 0.08   
        slow_zone = 15.0    
        #print(current_angle)
        # stop if within deadzone
        if abs(diff) <= deadzone:
            self.base.move(0, 0, 0)
            self.rotating = False
            return

        # speed scaled with error magnitude (smooth, proportional-ish)
        if abs(diff) >= slow_zone:
            speed = max_speed
        else:
            frac = abs(diff) / slow_zone
            speed = min_speed + (max_speed - min_speed) * frac
            speed = max(min_speed, min(speed, max_speed))

        # choose rotation sign based on error each tick (avoid stale direction)
        direction = -1 if diff > 0 else 1
        angular = direction * speed

        self.base.move(0, 0, angular)

