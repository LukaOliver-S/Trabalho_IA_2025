import numpy as np

class ObjectDetector:
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

    def _dbg(self, msg):
        if self.debug:
            print(msg)

    def detect(self):
        """Read lidars and update `last` flags (keeps prior behavior)."""
        try:
            min_low, min_high = self.sensors.read_lidars()
        except Exception:
            self._dbg("Erro lendo LiDARs (ObjectDetector.detect)")
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
        cube = obstacle = blocking = False

        # none
        if min_low == float("inf") and min_high == float("inf"):
            pass
        # low only => small cube
        elif min_low < float("inf") and min_high == float("inf"):
            self._dbg(f"Cubinho detectado (apenas baixo) | baixo={min_low:.3f} m")
            cube = True
        # high only => tall object
        elif min_low == float("inf") and min_high < float("inf"):
            self._dbg(f"Objeto alto detectado (apenas alto) | alto={min_high:.3f} m")
        else:
            if diff >= self.distance_diff_thresh and min_low < min_high:
                self._dbg(f"Cubo acessível | baixo={min_low:.3f} m alto={min_high:.3f} Δ={diff:.3f} m")
                cube = True
            if min_low < self.obstacle_min_dist:
                obstacle = True
                if np.isfinite(min_high) and abs(diff) <= 0.02:
                    blocking = True
                    self._dbg(f"Obstáculo muito perto (bloqueando) | baixo={min_low:.3f} alto={min_high:.3f} Δ={diff:.3f}")
                else:
                    self._dbg(f"Obstáculo próximo | baixo={min_low:.3f} m")

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