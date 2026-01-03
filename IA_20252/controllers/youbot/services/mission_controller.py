from cv2 import rotate
import numpy as np


class MissionController:
    """
    Mantém o heading definido por passos "rotate" durante os "path".
    Pausa o path, corrige yaw se necessário e retoma automaticamente.
    Também executa missões de armazenamento de blocos por cor.
    """

    def __init__(self, navigator, robot):
        self.navigator = navigator
        self.robot = robot

        # ===== SEQUÊNCIA DE MISSÃO =====
        self.steps = [
            ("path", "busca_principal"),
            ("rotate", 90),
            ("path", "busca_lateral_esquerda1"),
            ("rotate", 90),
            ("path", "busca_lateral_esquerda2"),
            ("rotate", 180),
            ("path", "busca_lateral_esquerda3"),
            ("rotate", -90),
            ("path", "busca_fundo_direita"),
            ("rotate", -90),
            ("path", "busca_lateral_direita"),
            ("rotate", 90),
            ("path", "caixa_blue"),
            ("store", "blue"),
            ("path", "caixa_blue_re"),
            ("rotate", 90),
            ("path", "caixa_red"),
            ("store", "red"),
            ("path", "caixa_red_re"),
            ("rotate", 90),
            ("path", "caixa_green"),
            ("store", "green"),
        ]

        self.current_step = 0
        self.active = False

        # ===== CONTROLE DE ROTAÇÃO =====
        self.ROTATION_TOL_DEG = 0.5
        self._rotation_target = None
        self._fixed_heading = None
        self._saved_nav_state = None
        self._rotation_interruption = False

    # =====================================================
    # CONTROLE DE MISSÃO
    # =====================================================

    def start(self):
        self.active = True
        self.current_step = 0
        self._start_current_step()

    def update(self):
        if not self.active:
            return

        step_type, value = self.steps[self.current_step]

        # =================================================
        # STORE (armazenar blocos por cor)
        # =================================================
        if step_type == "store":
            if getattr(self.robot, "DEBUG", False):
                print(f"[MissionController] storing blocks: {value}")

            # store é bloqueante
            self.robot.block_handler.store(value)

            self._next_step()
            return

        # =================================================
        # PATH
        # =================================================
        if step_type == "path":
            # correção de heading durante path
            if self._fixed_heading is not None and not self._rotation_interruption:
                cur = self.robot.get_current_angle()
                diff = self.robot.angle_diff(self._fixed_heading, cur)

                if abs(diff) > 5.0:
                    if getattr(self.robot, "DEBUG", False):
                        print(
                            f"[MissionController] deviation {diff:.1f}° -> correcting to {self._fixed_heading:.1f}°"
                        )

                    self._saved_nav_state = self.navigator.save_state()
                    self.navigator.stop()

                    self._rotation_target = self._fixed_heading
                    self._rotation_interruption = True

                    self.robot.rotate_to_angle(self._rotation_target)
                    if hasattr(self.robot, "update_rotation"):
                        self.robot.update_rotation()
                    return

            # aguardando correção
            if self._rotation_interruption:
                cur = self.robot.get_current_angle()
                diff = self.robot.angle_diff(self._rotation_target, cur)

                if abs(diff) <= self.ROTATION_TOL_DEG:
                    try:
                        self.robot.base.move(0, 0, 0)
                    except Exception:
                        pass

                    self.robot.rotating = False
                    self._rotation_interruption = False

                    if self._saved_nav_state:
                        self.navigator.restore_state(self._saved_nav_state)
                        self._saved_nav_state = None
                    return

                if not getattr(self.robot, "rotating", False):
                    self.robot.rotate_to_angle(self._rotation_target)

                if hasattr(self.robot, "update_rotation"):
                    self.robot.update_rotation()
                return

            # path normal
            finished = self.navigator.update()
            if finished:
                self._next_step()
            return

        # =================================================
        # ROTATE
        # =================================================
        if step_type == "rotate":
            if self._rotation_target is None:
                self._rotation_target = (
                    self.robot.get_current_angle() + value
                ) % 360.0

            cur = self.robot.get_current_angle()
            diff = self.robot.angle_diff(self._rotation_target, cur)

            if getattr(self.robot, "DEBUG", False):
                print(
                    f"[MissionController] rotate: target={self._rotation_target:.1f}, "
                    f"cur={cur:.1f}, diff={diff:.2f}"
                )

            if abs(diff) <= self.ROTATION_TOL_DEG:
                try:
                    self.robot.base.move(0, 0, 0)
                except Exception:
                    pass

                self.robot.rotating = False

                if value % 90 == 0:
                    self._fixed_heading = self._rotation_target
                else:
                    self._fixed_heading = None

                self._rotation_target = None
                self._next_step()
                return

            if not getattr(self.robot, "rotating", False):
                self.robot.rotate_to_angle(self._rotation_target)

            if hasattr(self.robot, "update_rotation"):
                self.robot.update_rotation()
            return

    # =====================================================
    # HELPERS
    # =====================================================

    def _start_current_step(self):
        step_type, value = self.steps[self.current_step]

        if step_type == "path":
            if getattr(self.robot, "DEBUG", False):
                print(f"[MissionController] starting path {value}")
            self.navigator.start(value)

        elif step_type == "rotate":
            if getattr(self.robot, "DEBUG", False):
                print(f"[MissionController] starting rotate delta={value}")
            self._start_rotation(value)

        elif step_type == "store":
            if getattr(self.robot, "DEBUG", False):
                print(f"[MissionController] starting store {value}")
            # execução ocorre no update()

    def _next_step(self):
        self.current_step += 1

        if self.current_step >= len(self.steps):
            print("✅ Missão concluída")
            self.active = False
            self.navigator.stop()
            return

        self._start_current_step()

    def _start_rotation(self, delta_deg):
        current = self.robot.get_current_angle()
        target = (current + delta_deg) % 360.0

        self._rotation_target = target

        if delta_deg % 90 == 0:
            self._fixed_heading = target
        else:
            self._fixed_heading = None

        if getattr(self.robot, "DEBUG", False):
            print(
                f"[MissionController] start_rotation -> target={target:.1f} (delta={delta_deg})"
            )

        self.robot.rotate_to_angle(target)
        if hasattr(self.robot, "update_rotation"):
            self.robot.update_rotation()
