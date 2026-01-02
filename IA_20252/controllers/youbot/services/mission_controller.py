import numpy as np


class MissionController:
    def __init__(self, navigator, robot):
        self.navigator = navigator
        self.robot = robot

        self.steps = [
            ("path", "busca_principal"),
            ("rotate", 90),
            ("path", "busca_lateral_esquerda1"),
            ("rotate", 90),
            ("path", "busca_lateral_esquerda2"),
            # ("rotate", 180),
            # ("path", "busca_lateral_direita"),
        ]

        self.current_step = 0
        self.active = False

    # =================================================
    # API
    # =================================================

    def start(self):
        self.active = True
        self.current_step = 0
        self._start_current_step()

    def update(self):
        if not self.active:
            return

        step_type, _ = self.steps[self.current_step]

        # ===== CAMINHO =====
        if step_type == "path":
            finished = self.navigator.update()
            if finished:
                self._next_step()

        # ===== ROTAÇÃO =====
        elif step_type == "rotate":
            # rotação termina quando robot.rotating vira False
            if not self.robot.rotating:
                self._next_step()

        # ===== FORWARD =====
        elif step_type == "forward":
            finished = self.navigator.update()
            if finished:
                self._next_step()

    # =================================================
    # CONTROLE DE PASSOS
    # =================================================

    def _start_current_step(self):
        step_type, value = self.steps[self.current_step]

        if step_type == "path":
            self.navigator.start(value)

        elif step_type == "rotate":
            self._start_rotation(value)

        elif step_type == "forward":
            self.navigator.start_forward()

    def _next_step(self):
        self.current_step += 1

        if self.current_step >= len(self.steps):
            print("✅ Missão concluída")
            self.active = False
            self.navigator.stop()
            return

        self._start_current_step()

    # =================================================
    # ROTAÇÃO (delegada ao robô)
    # =================================================

    def _start_rotation(self, delta_deg):
        """
        Inicia rotação relativa usando o controlador do robô.
        Não bloqueia.
        """
        current = self.robot.get_current_angle()
        target = (current + delta_deg) % 360
        self.robot.rotate_to_angle(target)