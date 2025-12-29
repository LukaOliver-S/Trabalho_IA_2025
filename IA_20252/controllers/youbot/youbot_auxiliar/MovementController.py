# controllers/youbot/youbot_auxiliar/MovementController.py
import math
from typing import Callable, Optional

class MovementController:
    def __init__(self,
                 base,
                 step_wait: Callable[[float], None] = None,
                 compass: Optional[object] = None):
        self.base = base
        self._step_wait = step_wait
        self.compass = compass  # store the device (Webots Compass)

        # translational parameters
        self.forward_speed = 0.3
        self.strafe_speed = 0.3
        self.movement_duration = 10

        # counters (keyboard-driven)
        self.move_forward_counter = 0
        self.move_backward_counter = 0
        self.strafe_left_counter = 0
        self.strafe_right_counter = 0

        # status flags (explicit)
        self.block_forward = False
        self.is_picking = False

        # rotation state & params (owned here)
        

    # ---- compass reader (internal) ----
    def get_current_angle(self) -> Optional[float]:
        if self.compass is None:
            return None
        north = self.compass.getValues()
        angle_rad = math.atan2(north[0], north[1])
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        return angle_deg

    # ---- keyboard helpers ----
    def forward(self, ticks: int = None):
        self.move_forward_counter = self.movement_duration if ticks is None else ticks

    def backward(self, ticks: int = None):
        self.move_backward_counter = self.movement_duration if ticks is None else ticks

    def strafe_left(self, ticks: int = None):
        self.strafe_left_counter = self.movement_duration if ticks is None else ticks

    def strafe_right(self, ticks: int = None):
        self.strafe_right_counter = self.movement_duration if ticks is None else ticks

    def stop_all(self):
        self.move_forward_counter = 0
        self.move_backward_counter = 0
        self.strafe_left_counter = 0
        self.strafe_right_counter = 0
        self.base.move(0, 0, 0)

    def immediate_stop(self, wait: float = 0.03):
        self.stop_all()
        if self._step_wait:
            self._step_wait(wait)

    # ---- small-step helpers ----
    def strafe_left_small(self, dur: float = 0.15):
        self.base.move(0, self.strafe_speed, 0)
        if self._step_wait: self._step_wait(dur)
        self.base.move(0, 0, 0)

    def strafe_right_small(self, dur: float = 0.15):
        self.base.move(0, -self.strafe_speed, 0)
        if self._step_wait: self._step_wait(dur)
        self.base.move(0, 0, 0)

    # ---- rotation API (owned here) ----
    

    # ---- main update: rotation priority ----
    def update(self):
   
        if self.is_picking:
            self.base.move(0, 0, 0)
            return

        if self.block_forward:
            self.move_forward_counter = 0
            self.move_backward_counter = 0
            if self.strafe_left_counter == 0 and self.strafe_right_counter == 0:
                self.base.move(0, 0, 0)
                return

        vx = vy = omega = 0.0
        if self.move_forward_counter > 0:
            vx = self.forward_speed
            self.move_forward_counter -= 1
        elif self.move_backward_counter > 0:
            vx = -self.forward_speed
            self.move_backward_counter -= 1

        if self.strafe_left_counter > 0:
            vy = self.strafe_speed
            self.strafe_left_counter -= 1
        elif self.strafe_right_counter > 0:
            vy = -self.strafe_speed
            self.strafe_right_counter -= 1

        self.base.move(vx, vy, omega)