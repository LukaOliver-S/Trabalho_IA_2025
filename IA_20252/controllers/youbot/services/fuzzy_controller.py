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
        self.dist_front['near'] = fuzz.trimf(
    self.dist_front.universe, [0.0, 0.0, 0.22]
)
        self.dist_front['medium'] = fuzz.trimf(
            self.dist_front.universe, [0.10, 0.25, 0.40]
        )
        self.dist_front['far'] = fuzz.trimf(
            self.dist_front.universe, [0.30, self.LIDAR_MAX, self.LIDAR_MAX]
        )


        # Funções de pertinência (lateral) — apenas para plot compatível
        self.lateral_offset['left'] = fuzz.trimf(self.lateral_offset.universe, [-0.5, -0.25, 0.0])
        self.lateral_offset['center'] = fuzz.trimf(self.lateral_offset.universe, [-0.05, 0.0, 0.05])
        self.lateral_offset['right'] = fuzz.trimf(self.lateral_offset.universe, [0.0, 0.25, 0.5])



        #self.v['low'] = fuzz.trimf(self.v.universe, [0.00, 0.05, 0.12])
        #self.v['medium'] = fuzz.trimf(self.v.universe, [0.08, 0.15, 0.22])
        #self.v['high'] = fuzz.trimf(self.v.universe, [0.18, 0.24, 0.30])

        #rules = [
       #     ctrl.Rule(self.dist_front['near'], self.v['low']),
      #      ctrl.Rule(self.dist_front['medium'], self.v['medium']),
     #       ctrl.Rule(self.dist_front['far'], self.v['high']),
     #   ]
     
     
     
        # Funções de pertinência (velocidade) — baixa / média / alta
        self.v['very_low']  = fuzz.trimf(self.v.universe, [0.00, 0.03, 0.06])
        self.v['low']       = fuzz.trimf(self.v.universe, [0.05, 0.08, 0.11])
        self.v['medium']    = fuzz.trimf(self.v.universe, [0.10, 0.15, 0.20])
        self.v['high']      = fuzz.trimf(self.v.universe, [0.19, 0.23, 0.27])
        self.v['very_high'] = fuzz.trimf(self.v.universe, [0.26, 0.29, 0.30])

        # Compatibilidade com notebooks que esperam vx, vy
        self.vx = self.v
        self.vy = self.v

        # Regras fuzzy (apenas por distância frontal)
        # Regras fuzzy (apenas por distância frontal) – versão suave, sem OR no consequente
        rules = [
            # Bem perto: ativa very_low e low com duas regras
            ctrl.Rule(self.dist_front['near'], self.v['very_low']),
            ctrl.Rule(self.dist_front['near'], self.v['low']),

            # Médio: ativa low e medium
            ctrl.Rule(self.dist_front['medium'], self.v['low']),
            ctrl.Rule(self.dist_front['medium'], self.v['medium']),

            # Longe: ativa high e very_high
            ctrl.Rule(self.dist_front['far'], self.v['high']),
            ctrl.Rule(self.dist_front['far'], self.v['very_high']),
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

    def compute(self, front_dist, lateral=None):
        return self.compute_velocity(front_dist, lateral)


__all__ = ['FuzzySimple']