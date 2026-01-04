import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzySimple:
    """Simple fuzzy speed selector based only on frontal distance (LIDAR)."""
    LIDAR_MAX = 0.3  # máximo alcance do LIDAR em metros

    def __init__(self, v_max: float = 0.30):
        self.v_max = v_max 
        # Entrada: distância frontal (ajustada ao alcance do lidar)
        self.dist_front = ctrl.Antecedent(
            np.arange(0.0, self.LIDAR_MAX + 0.001, 0.005), 'dist_front'
        )
      
        # Saída única: velocidade (vx = vy), intervalo [0.0, 0.30] m/s
        self.v = ctrl.Consequent(
            np.arange(0.0, self.v_max + 0.001, 0.005), 'v'
        )


        # Funções de pertinência (distância) — ajustadas para [0, LIDAR_MAX]
        self.dist_front['near'] = fuzz.gaussmf(
    self.dist_front.universe, 0.05, 0.04
        )
        self.dist_front['medium'] = fuzz.gaussmf(
            self.dist_front.universe, 0.15, 0.05
        )
        self.dist_front['far'] = fuzz.gaussmf(
            self.dist_front.universe, 0.25, 0.05
        )


      

        # Funções de pertinência (velocidade) — gaussianas para transições suaves
        self.v['low'] = fuzz.gaussmf(
            self.v.universe, 0.20 * self.v_max, 0.10 * self.v_max
        )
        self.v['medium'] = fuzz.gaussmf(
            self.v.universe, 0.55 * self.v_max, 0.12 * self.v_max
        )
        self.v['high'] = fuzz.gaussmf(
            self.v.universe, 0.90 * self.v_max, 0.10 * self.v_max
        )

        rules = [
           ctrl.Rule(self.dist_front['near'], self.v['low']),
           ctrl.Rule(self.dist_front['medium'], self.v['medium']),
           ctrl.Rule(self.dist_front['far'], self.v['high']),
       ]
     
     
     
        # # Funções de pertinência (velocidade) — baixa / média / alta
        # self.v['very_low']  = fuzz.trimf(self.v.universe, [0.00 * self.v_max, 0.10 * self.v_max, 0.20 * self.v_max])
        # self.v['low']       = fuzz.trimf(self.v.universe, [0.1667 * self.v_max, 0.2667 * self.v_max, 0.3667 * self.v_max])
        # self.v['medium']    = fuzz.trimf(self.v.universe, [0.3333 * self.v_max, 0.50 * self.v_max, 0.6667 * self.v_max])
        # self.v['high']      = fuzz.trimf(self.v.universe, [0.6333 * self.v_max, 0.7667 * self.v_max, 0.90 * self.v_max])
        # self.v['very_high'] = fuzz.trimf(self.v.universe, [0.8667 * self.v_max, 0.9667 * self.v_max, 1.00 * self.v_max])

        # # Compatibilidade com notebooks que esperam vx, vy
        # self.vx = self.v
        # self.vy = self.v

        # # Regras fuzzy (apenas por distância frontal)
        # rules = [
        #     # Bem perto: ativa very_low e low com duas regras
        #     ctrl.Rule(self.dist_front['near'], self.v['very_low']),
        #     ctrl.Rule(self.dist_front['near'], self.v['low']),

        #     # Médio: ativa low e medium
        #     ctrl.Rule(self.dist_front['medium'], self.v['low']),
        #     ctrl.Rule(self.dist_front['medium'], self.v['medium']),

        #     # Longe: ativa high e very_high
        #     ctrl.Rule(self.dist_front['far'], self.v['high']),
        #     ctrl.Rule(self.dist_front['far'], self.v['very_high']),
        # ]


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
        
        try:
            self.fuzzy_sim.input['dist_front'] = d
            self.fuzzy_sim.compute()
            v = float(self.fuzzy_sim.output['v'])
        except (KeyError, ValueError) as e:
            # Fallback se defuzzificação falhar (ex: entrada fora do range de cobertura)
            if d <= 0.08:
                v = 0.20 * self.v_max  # velocidade baixa
            elif d >= 0.22:
                v = 0.90 * self.v_max  # velocidade alta
            else:
                # Interpola linearmente entre low e high
                ratio = (d - 0.08) / (0.22 - 0.08)
                v = (0.20 + ratio * 0.70) * self.v_max
        
        return v, v
   
__all__ = ['FuzzySimple']