from cv2 import rotate
import numpy as np


# Em mission_controller.py

class MissionController:
    def __init__(self, navigator, robot):
        self.navigator = navigator
        self.robot = robot

        # ===== SEQUÊNCIA PRINCIPAL =====
        self.main_steps = [
            
            ('rotate_to', 0),
            
            ("path", "busca_principal"),
            ('rotate_to', 90),
            ("path", "busca_lateral_esquerda1"),
            ('rotate_to', 180),
            ("path", "busca_lateral_esquerda2"),
            ('rotate_to', 0),
            ("path", "busca_lateral_esquerda3"),
            ('rotate_to', 270),
            ("path", "busca_fundo_direita"),
            ('rotate_to', 180),
            ("path", "busca_lateral_direita"),
            ('rotate_to', 270),
            ("path", "caixa_blue"),
            ("store", "blue"),
            ("path", "caixa_blue_re"),
            ('rotate_to', 0),
            ("path", "caixa_red"),
            ("store", "red"),
            ("path", "caixa_red_re"),
            ('rotate_to', 90),
            ("path", "caixa_green"),
            ("store", "green"),
            ("check_cubes", 15),  
        ]

        # ===== SEQUÊNCIA DE FALLBACK =====
        self.fallback_steps = [
            ("rotate_to",180),
            ("path", "fall_back_1"),
            ("push_all_cubes", None),
            ("path", "fall_back_2"),
            ('rotate_to', 0),
            ("restart_search", None),  
        ]

        self.steps = self.main_steps.copy()
        self.current_step = 0
        self.active = False
        self.fallback_attempts = 0
        self.max_fallback_attempts = 10  # máximo de reinicializações

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
        """Inicia a missão do zero"""
        self.active = True
        self.current_step = 0
        self._start_current_step()
    def update(self):
        if not self.active:
            return

        step_type, value = self.steps[self.current_step]

        # =================================================
        # CHECK_CUBES (NOVO)
        # =================================================
        if step_type == "check_cubes":
            target_count = value
            actual_count = self.robot.block_handler.stored

            if getattr(self.robot, "DEBUG", True):
                print(f"[MissionController] Verificação: {actual_count}/{target_count} cubos")
            print(actual_count)
            print(target_count)
            if actual_count >= target_count:
                # Sucesso — missão concluída
                print(f" Missão completa: {actual_count} cubos coletados!")
                self.active = False
                self.navigator.stop()
                return

            # Faltam cubos — verifica se pode fazer fallback
            self.robot.block_handler.counter = 0 
            if self.fallback_attempts >= self.max_fallback_attempts:
                print(f" Fallback limite atingido ({self.fallback_attempts}x) — finalizando com {actual_count} cubos")
                self.active = False
                self.navigator.stop()
                return

            # Ativa fallback
            self.fallback_attempts += 1
            print(f" Fallback #{self.fallback_attempts}: apenas {actual_count} cubos — reiniciando busca")

            # Insere steps de fallback
            self.steps = self.steps[:self.current_step + 1] + self.fallback_steps
            self._next_step()
            return

        # =================================================
        # RESTART_SEARCH (NOVO)
        # =================================================
        if step_type == "restart_search":
            if getattr(self.robot, "DEBUG", False):
                print("[MissionController] Reiniciando missões de busca")

            # Define índice do primeiro "path" de busca (busca_principal)
            search_start_index = 1  # após rotate_to(0)

            # Adiciona buscas novamente
            search_steps = self.main_steps[search_start_index:self.main_steps.index(("store", "green")) + 2]
            
            # Insere após o step atual
            self.steps = self.steps[:self.current_step + 1] + search_steps
            self._next_step()
            return

        # =================================================
        # STORE (MODIFICADO — adiciona contagem)
        # =================================================
        if step_type == "store":
            if getattr(self.robot, "DEBUG", False):
                print(f"[MissionController] storing blocks: {value}")

            before_count = self.robot.block_handler.counter
            self.robot.block_handler.store(value)
            after_count = self.robot.block_handler.counter

            if getattr(self.robot, "DEBUG", False):
                print(f"[MissionController] Store {value}: contador permanece em {after_count}")

            self._next_step()
            return

        if step_type == "push_all_cubes":
            if getattr(self.robot, "DEBUG", False):
                print("[MissionController] Empurrando cubinhos para fora")
            
            self.robot.block_handler.push_all_cubes()
            self._next_step()
            return

        # =================================================
        # PATH
        # =================================================
        if step_type == "path":
            # correção de heading durante path
            print(value)
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
        if step_type == "rotate_to":
            if self._rotation_target is None:
                self._rotation_target = value % 360.0

            cur = self.robot.get_current_angle()
            diff = self.robot.angle_diff(self._rotation_target, cur)

            if abs(diff) <= self.ROTATION_TOL_DEG:
                try:
                    self.robot.base.move(0, 0, 0)
                except Exception:
                    pass

                self.robot.rotating = False
                # Se for múltiplo de 90°, guarda heading fixo (opcional)
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
        elif step_type == "rotate_to":  # ← ADICIONAR ISSO
            if getattr(self.robot, "DEBUG", False):
                print(f"[MissionController] starting rotate_to {value}°")
            self._rotation_target = None  # ← limpa o target anterior


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
