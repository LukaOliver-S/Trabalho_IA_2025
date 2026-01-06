import math
import numpy as np
import csv
from pathlib import Path
from datetime import datetime

class AlignmentController:
    def __init__(self, base, sensors, kp=1.5, max_vy=0.2,
                 deadzone=0.005, forward_speed=0.3,
                 pick_distance=0.104, pick_tol=0.005,
                 min_approach_speed=0.03, debug=False, fuzzy=None,
                 max_align_ticks=300, block_handler=None, max_capacity=5):
        self.base = base
        self.sensors = sensors
        self.kp = kp
        self.max_vy = max_vy
        self.deadzone = deadzone
        self.forward_speed = forward_speed
        self.PICK_DISTANCE = pick_distance
        self.PICK_TOL = pick_tol
        self.MIN_APPROACH_SPEED = min_approach_speed
        self.DEBUG = debug
        self.fuzzy = fuzzy
        
        # ===== CONTROLE DE CAPACIDADE =====
        self.block_handler = block_handler
        self.MAX_CAPACITY = max_capacity
        
        # ===== TIMEOUT GERAL DE ALINHAMENTO =====
        self.max_align_ticks = max_align_ticks
        self.align_ticks = 0
        self.alignment_failed = False
        self.lost_cube_ticks = 0
        self.max_lost_cube_ticks = 50

        # ===== LOGGING DE DADOS =====
        self.log_data = []
        self.log_enabled = False
        self.log_dir = Path("fuzzy_alignment_data")
        self.log_dir.mkdir(exist_ok=True)
        self.current_session_id = None

    def start_logging_session(self):
        """Inicia uma nova sessão de logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_id = timestamp
        self.log_data = []
        if self.DEBUG:
            print(f"📊 Logging iniciado: {self.current_session_id}")

    def _log_alignment_data(self, phase, dist, lateral, vx, vy, speed_factor):
        """Registra dados de alinhamento"""
        if not self.log_enabled or self.current_session_id is None:
            return
        
        self.log_data.append({
            'tick': self.align_ticks,
            'phase': phase,
            'dist_frontal': dist,
            'offset_lateral': lateral if lateral is not None else 0,
            'vx': vx,
            'vy': vy,
            'speed_factor': speed_factor
        })

    def save_logging_session(self):
        """Salva dados da sessão em CSV"""
        if not self.log_data or self.current_session_id is None:
            return
        
        filename = self.log_dir / f"alignment_{self.current_session_id}.csv"
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'tick', 'phase', 'dist_frontal', 'offset_lateral', 
                'vx', 'vy', 'speed_factor'
            ])
            writer.writeheader()
            writer.writerows(self.log_data)
        
        if self.DEBUG:
            print(f"💾 Dados salvos: {filename}")
        
        self.log_data = []
        self.current_session_id = None
        
    def reset_alignment(self):
        """Reseta contador e flag"""
        # Salva dados antes de resetar
        if self.log_enabled and self.current_session_id is not None:
            self.save_logging_session()
        
        self.align_ticks = 0
        self.alignment_failed = False
        self.lost_cube_ticks = 0

    def _check_capacity(self):
        """Verifica se ainda há espaço na base do robô"""
        if self.block_handler is None:
            return True
        
        if self.block_handler.counter >= self.MAX_CAPACITY:
            self.alignment_failed = True
            self.base.move(0, 0, 0)
            if self.DEBUG:
                print(f"⛔ Base cheia: {self.block_handler.counter}/{self.MAX_CAPACITY} — desistindo")
            return False
        return True

    def _get_fuzzy_speed_factor(self, dist):
        """
        Retorna fator de velocidade (0.0 a 1.0) baseado na distância frontal.
        Usado para modular tanto movimento lateral quanto frontal.
        """
        if self.fuzzy is None:
            return 1.0
        
        try:
            if not np.isfinite(dist):
                return 1.0
            
            v_fuzzy, _ = self.fuzzy.compute_velocity(dist)
            # Normaliza pela velocidade máxima
            factor = v_fuzzy / self.fuzzy.v_max
            return max(0.0, min(1.0, factor))
        except Exception:
            return 1.0

    def align_with_cube(self):
        """
        Alinha lateralmente com o cubo usando fuzzy para modular a agressividade.
        """
        # ===== INICIA LOGGING SE NECESSÁRIO =====
        if self.log_enabled and self.current_session_id is None:
            self.start_logging_session()
        
        # ===== VERIFICA CAPACIDADE =====
        if not self._check_capacity():
            return False
        
        lateral = self.sensors.get_cube_lateral_offset()
        
        # ===== SEM CUBO DETECTADO =====
        if lateral is None:
            if self.align_ticks > 0:
                self.align_ticks += 1
                self.lost_cube_ticks += 1
                
                if self.lost_cube_ticks > self.max_lost_cube_ticks:
                    self.alignment_failed = True
                    self.base.move(0, 0, 0)
                    if self.DEBUG:
                        print(f"⛔ Cubo perdido por {self.lost_cube_ticks} ticks")
                    return False
                
                if self.align_ticks > self.max_align_ticks:
                    self.alignment_failed = True
                    self.base.move(0, 0, 0)
                    if self.DEBUG:
                        print(f"⛔ Timeout geral: {self.align_ticks} ticks")
                    return False
                
                if self.DEBUG:
                    print(f"⚠️ Cubo não visível (lost={self.lost_cube_ticks}, total={self.align_ticks})")
            else:
                if self.DEBUG:
                    print("Nenhum cubo detectado para alinhar.")
            
            self.base.move(0, 0, 0)
            return False

        self.lost_cube_ticks = 0
        self.align_ticks += 1

        if self.align_ticks > self.max_align_ticks:
            self.alignment_failed = True
            self.base.move(0, 0, 0)
            if self.DEBUG:
                print(f"⛔ Timeout geral: {self.align_ticks} ticks")
            return False

        # ===== ALINHADO =====
        if abs(lateral) <= self.deadzone:
            self.base.move(0, 0, 0)
            if self.DEBUG:
                print(f"✅ Alinhado lateralmente: lateral={lateral:.4f}")
            return True

        # ===== FUZZY MODULA ALINHAMENTO LATERAL =====
        speed_p = self.kp * abs(lateral)
        
        dist = self.sensors.read_low_filtered()
        speed_factor = self._get_fuzzy_speed_factor(dist)
        
        speed = min(speed_p * speed_factor, self.max_vy)
        vy = math.copysign(speed, lateral)

        self.base.move(0, vy, 0)
        
        # ===== LOGA DADOS =====
        self._log_alignment_data('align', dist, lateral, 0, vy, speed_factor)
        
        if self.DEBUG:
            print(
                f"[Align FUZZY] tick={self.align_ticks}/{self.max_align_ticks} "
                f"dist={dist:.3f} lateral={lateral:.4f} factor={speed_factor:.2f} → vy={vy:.3f}"
            )
        return False

    def auto_approach_cube(self):
        """
        Aproxima do cubo frontalmente usando fuzzy para modular a velocidade.
        """
        # ===== VERIFICA CAPACIDADE =====
        if not self._check_capacity():
            return False
        
        dist = self.sensors.read_low_filtered()
        
        if not np.isfinite(dist):
            if self.DEBUG:
                print("⚠️ Leitura inválida do lidar")
            self.base.move(0, 0, 0)
            return False

        self.align_ticks += 1

        if self.align_ticks > self.max_align_ticks:
            self.alignment_failed = True
            self.base.move(0, 0, 0)
            if self.DEBUG:
                print(f"⛔ Timeout geral: {self.align_ticks} ticks")
            return False

        err = dist - self.PICK_DISTANCE

        # ===== CHEGOU NA DISTÂNCIA ALVO =====
        if abs(err) <= self.PICK_TOL:
            self.base.move(0, 0, 0)
            if self.DEBUG:
                print(f"✅ Approach completo: distância={dist:.4f}")
            self.reset_alignment()
            return True

        # ===== VERIFICA SE AINDA ESTÁ ALINHADO =====
        lateral = self.sensors.get_cube_lateral_offset()
        if lateral is not None and abs(lateral) > self.deadzone:
            if self.DEBUG:
                print("⚠️ Desalinhado — voltar para align")
            self.base.move(0, 0, 0)
            return False

        # ===== FUZZY MODULA APROXIMAÇÃO FRONTAL =====
        vx_p = 0.9 * err
        
        speed_factor = self._get_fuzzy_speed_factor(dist)
        
        vx = vx_p * speed_factor
        vx = max(-self.forward_speed, min(self.forward_speed, vx))

        if abs(vx) < self.MIN_APPROACH_SPEED:
            vx = math.copysign(self.MIN_APPROACH_SPEED, vx)

        self.base.move(vx, 0, 0)
        
        # ===== LOGA DADOS =====
        self._log_alignment_data('approach', dist, lateral, vx, 0, speed_factor)

        if self.DEBUG:
            print(
                f"[Approach FUZZY] tick={self.align_ticks}/{self.max_align_ticks} "
                f"dist={dist:.3f} factor={speed_factor:.2f} → vx={vx:.3f}"
            )

        return False