import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzySimple:
    """Simple fuzzy speed selector based only on frontal distance (LIDAR)."""
    LIDAR_MAX = 0.5  # máximo alcance do LIDAR em metros

    def __init__(self):
        # Entrada: distância frontal (ajustada ao alcance do lidar)
        self.dist_front = ctrl.Antecedent(
            np.arange(0.0, self.LIDAR_MAX + 0.001, 0.005), 'dist_front'
        )
        # Entrada compatível (não usada no cálculo, só para visualização)
        self.lateral_offset = ctrl.Antecedent(
            np.arange(-0.5, 0.51, 0.01), 'lateral_offset'
        )
        # Saída única: velocidade (vx = vy), intervalo [0.0, 0.30] m/s
        self.v = ctrl.Consequent(np.arange(0.0, 0.305, 0.005), 'v')

        # Funções de pertinência (distância) — ajustadas para [0, LIDAR_MAX]
        self.dist_front['near'] = fuzz.trapmf(
            self.dist_front.universe, [0.0, 0.0, 0.10, 0.15]
        )
        self.dist_front['medium'] = fuzz.trapmf(
            self.dist_front.universe, [0.12, 0.22, 0.32, 0.42]
        )
        self.dist_front['far'] = fuzz.trapmf(
            self.dist_front.universe, [0.35, 0.40, self.LIDAR_MAX, self.LIDAR_MAX]
        )

        # Funções de pertinência (lateral) — apenas para plot compatível
        self.lateral_offset['left'] = fuzz.trimf(self.lateral_offset.universe, [-0.5, -0.25, 0.0])
        self.lateral_offset['center'] = fuzz.trimf(self.lateral_offset.universe, [-0.05, 0.0, 0.05])
        self.lateral_offset['right'] = fuzz.trimf(self.lateral_offset.universe, [0.0, 0.25, 0.5])

        # Funções de pertinência (velocidade) — baixa / média / alta
        self.v['low'] = fuzz.trimf(self.v.universe, [0.0, 0.05, 0.10])
        self.v['medium'] = fuzz.trimf(self.v.universe, [0.10, 0.15, 0.20])
        self.v['high'] = fuzz.trimf(self.v.universe, [0.16, 0.24, 0.30])

        # Compatibilidade com notebooks que esperam vx, vy
        self.vx = self.v
        self.vy = self.v

        # Regras fuzzy (apenas por distância frontal)
        rules = [
            ctrl.Rule(self.dist_front['near'], self.v['low']),
            ctrl.Rule(self.dist_front['medium'], self.v['medium']),
            ctrl.Rule(self.dist_front['far'], self.v['high']),
        ]

        self.fuzzy_ctrl = ctrl.ControlSystem(rules)
        self._reset_sim()

    def _reset_sim(self):
        # recria a simulação para evitar estados residuais entre computações
        self.fuzzy_sim = ctrl.ControlSystemSimulation(self.fuzzy_ctrl)

    def compute_velocity(self, front_dist, lateral=None):
        """
        front_dist: distância frontal (float). Valores > LIDAR_MAX são truncados.
        lateral: parâmetro aceito para compatibilidade (ignorado aqui)
        Retorna: (vx, vy) onde vx == vy == saída fuzzy em m/s
        """
        d = float(front_dist) if np.isfinite(front_dist) else self.LIDAR_MAX
        d = max(0.0, min(self.LIDAR_MAX, d))

        self._reset_sim()
        self.fuzzy_sim.input['dist_front'] = d
        self.fuzzy_sim.compute()
        v = float(self.fuzzy_sim.output['v'])
        return v, v

    # Alias usado em notebooks/scripts
    def compute(self, front_dist, lateral=None):
        return self.compute_velocity(front_dist, lateral)


__all__ = ['FuzzySimple']