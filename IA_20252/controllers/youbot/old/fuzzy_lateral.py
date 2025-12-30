import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyLateralTSK:
    def __init__(self):
        # Variáveis de entrada
        self.dist_front = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'dist_front')
        self.lateral_offset = ctrl.Antecedent(np.arange(-0.5, 0.51, 0.01), 'lateral_offset')

        # Variáveis de saída
        self.vx = ctrl.Consequent(np.arange(-0.1, 0.11, 0.01), 'vx')
        self.vy = ctrl.Consequent(np.arange(-0.1, 0.11, 0.01), 'vy')

        # Funções de pertinência
        self.dist_front['near'] = fuzz.trapmf(self.dist_front.universe, [0, 0, 0.2, 0.4])
        self.dist_front['medium'] = fuzz.trapmf(self.dist_front.universe, [0.2, 0.4, 0.6, 0.8])
        self.dist_front['far'] = fuzz.trapmf(self.dist_front.universe, [0.6, 0.8, 1, 1])

        self.lateral_offset['left'] = fuzz.trimf(self.lateral_offset.universe, [-0.5, -0.25, 0])
        self.lateral_offset['center'] = fuzz.trimf(self.lateral_offset.universe, [-0.05, 0, 0.05])
        self.lateral_offset['right'] = fuzz.trimf(self.lateral_offset.universe, [0, 0.25, 0.5])

        self.vx['back'] = fuzz.trimf(self.vx.universe, [-0.1, -0.05, 0])
        self.vx['stop'] = fuzz.trimf(self.vx.universe, [-0.01, 0, 0.01])
        self.vx['forward'] = fuzz.trimf(self.vx.universe, [0, 0.05, 0.1])

        self.vy['left'] = fuzz.trimf(self.vy.universe, [-0.1, -0.05, 0])
        self.vy['center'] = fuzz.trimf(self.vy.universe, [-0.01, 0, 0.01])
        self.vy['right'] = fuzz.trimf(self.vy.universe, [0, 0.05, 0.1])

        # Regras fuzzy
        rules = [
            ctrl.Rule(self.dist_front['near'], self.vx['back']),
            ctrl.Rule(self.dist_front['medium'], self.vx['stop']),
            ctrl.Rule(self.dist_front['far'], self.vx['forward']),
            ctrl.Rule(self.lateral_offset['left'], self.vy['left']),
            ctrl.Rule(self.lateral_offset['center'], self.vy['center']),
            ctrl.Rule(self.lateral_offset['right'], self.vy['right']),
        ]

        self.fuzzy_ctrl = ctrl.ControlSystem(rules)
        self.fuzzy_sim = ctrl.ControlSystemSimulation(self.fuzzy_ctrl)

    def compute_velocity(self, front_dist, lateral):
        self.fuzzy_sim.input['dist_front'] = front_dist
        self.fuzzy_sim.input['lateral_offset'] = lateral
        self.fuzzy_sim.compute()
        return self.fuzzy_sim.output['vx'], self.fuzzy_sim.output['vy']