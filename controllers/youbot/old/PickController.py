# controllers/youbot/youbot_auxiliar/PickController.py
class PickController:
    def __init__(self, arm, gripper, base, step_wait, debug=False):
        self.arm = arm
        self.gripper = gripper
        self.base = base
        self._step_wait = step_wait
        self.DEBUG = debug
        self.is_picking = False
        self.gripper_descended = False

    def do_pick_blocking(self):
        """Blocking pick routine (same sequence as original)."""
        if self.is_picking:
            if self.DEBUG: print("[PICK] already running")
            return False
        self.is_picking = True
        self.base.move(0, 0, 0)

        if self.DEBUG: print("Abrindo garra...")
        self.gripper.release()
        if self._step_wait: self._step_wait(1.0)

        if self.DEBUG: print("Descendo braço...")
        self.arm.set_orientation(self.arm.FRONT)
        if self._step_wait: self._step_wait(0.05)
        self.arm.set_height(self.arm.FRONT_FLOOR)
        if self._step_wait: self._step_wait(3.5)
        self.arm.motors[1].setPosition(-1.10)
        self.arm.motors[2].setPosition(-1.70)
        self.arm.motors[3].setPosition(-0.70)
        if self._step_wait: self._step_wait(1.5)

        if self.DEBUG: print("Fechando garra...")
        self.gripper.grip()
        if self._step_wait: self._step_wait(1.2)

        if self.DEBUG: print("Subindo braço...")
        self.arm.motors[1].setPosition(-1.0)
        self.arm.motors[2].setPosition(-1.5)
        self.arm.motors[3].setPosition(-0.6)
        if self._step_wait: self._step_wait(1.5)

        if self.DEBUG: print("Levando para trás...")
        self.arm.motors[0].setPosition(0.0)
        self.arm.motors[1].setPosition(0.5)
        self.arm.motors[2].setPosition(1.0)
        self.arm.motors[3].setPosition(0.9)
        self.arm.motors[4].setPosition(-0.5)
        if self._step_wait: self._step_wait(6.0)

        if self.DEBUG: print("Liberando cubo atrás do robô...")
        self.gripper.release()
        if self._step_wait: self._step_wait(0.7)

        if self.DEBUG: print("Retornando braço e garra para posição inicial...")
        self.arm.set_height(self.arm.RESET)
        self.arm.set_orientation(self.arm.FRONT)
        if self._step_wait: self._step_wait(3.0)
        self.gripper.release()
        if self._step_wait: self._step_wait(0.5)

        if self.DEBUG: print("Pegada finalizada.")
        self.gripper_descended = True
        self.is_picking = False
        return True