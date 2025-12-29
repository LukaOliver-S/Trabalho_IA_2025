# controllers/youbot/youbot_auxiliar/ObjectDetector.py
import numpy as np

class ObjectDetector:
    """
    Dada uma SensorSuite, calcula:
      - cube_detected_a_frente
      - obstacle_detected
      - obstacle_blocking_cube
      - min_low, min_high
    Mantém o último resultado em `last`.
    """
    def __init__(self, sensors, distance_diff_thresh=0.01, obstacle_min_dist=0.30, step_wait=None, debug=False):
        self.sensors = sensors
        self.distance_diff_thresh = distance_diff_thresh
        self.obstacle_min_dist = obstacle_min_dist
        self._step_wait = step_wait
        self.debug = debug
        self.last = {
            "cube_detected_a_frente": False,
            "obstacle_detected": False,
            "obstacle_blocking_cube": False,
            "min_low": float("inf"),
            "min_high": float("inf"),
            "diff": float("inf"),
        }

    def detect(self):
        """Executa uma leitura e atualiza `last` com os flags e distâncias."""
        try:
            min_low, min_high = self.sensors.read_lidars()
        except Exception:
            if self.debug:
                print("Erro lendo LiDARs (ObjectDetector.detect)")
            # devolve estado "vazio" (nenhum objeto detectado)
            self.last.update({
                "cube_detected_a_frente": False,
                "obstacle_detected": False,
                "obstacle_blocking_cube": False,
                "min_low": float("inf"),
                "min_high": float("inf"),
                "diff": float("inf"),
            })
            return self.last

        diff = min_high - min_low

        cube = False
        obstacle = False
        blocking = False

        # Nada detectado
        if min_low == float("inf") and min_high == float("inf"):
            # nada a fazer
            pass
        # Apenas chão detecta => cubo pequeno
        elif min_low < float("inf") and min_high == float("inf"):
            if self.debug: print(f"🟦 Cubinho detectado (apenas baixo) | baixo={min_low:.3f} m")
            cube = True
        # Apenas alto detecta => objeto alto (não interessa para pegar)
        elif min_low == float("inf") and min_high < float("inf"):
            if self.debug: print(f"🟨 Objeto alto detectado (apenas alto) | alto={min_high:.3f} m")
        else:
            # Regras envolvendo diferença
            if diff >= self.distance_diff_thresh and min_low < min_high:
                if self.debug: print(f"🔹 Cubo acessível | baixo={min_low:.3f} m alto={min_high:.3f} Δ={diff:.3f} m")
                cube = True
            if min_low < self.obstacle_min_dist:
                obstacle = True
                if np.isfinite(min_high) and abs(diff) <= 0.02:
                    blocking = True
                    if self.debug: print(f"🟥 Obstáculo muito perto (bloqueando) | baixo={min_low:.3f} alto={min_high:.3f} Δ={diff:.3f}")
                else:
                    if self.debug: print(f"⚠️ Obstáculo próximo | baixo={min_low:.3f} m")

        self.last.update({
            "cube_detected_a_frente": cube,
            "obstacle_detected": obstacle,
            "obstacle_blocking_cube": blocking,
            "min_low": min_low,
            "min_high": min_high,
            "diff": diff,
        })
        return self.last

    def get_last(self):
        return self.last.copy()