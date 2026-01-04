"""Plot membership functions and FLC output map for the TSK lateral controller."""
import numpy as np

class FuzzyLateralTSK:
    def __init__(self, Vmax=0.06, R=0.3, deadzone=0.005, alpha=0.25,debug=False):
        self.Vmax = Vmax
        self.R = R
        self.deadzone = deadzone
        self.alpha = alpha
        self.v_prev = 0.0
        self.gains = {'VeryNear': 1.0, 'Near': 0.6, 'Far': 0.0}
        self.debug = debug
        
    def triangle(self, x, a, b, c):
        """Triangle membership function."""
        if x <= a or x >= c:
            return 0.0
        if x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        return (c - x) / (c - b)

    def trapezoid(self, x, a, b, c, d):
        """Trapezoid membership function."""
        if x <= a or x >= d:
            return 0.0
        if x < b:
            return (x - a) / (b - a)
        if x <= c:
            return 1.0
        return (d - x) / (d - c)

    def fuzz_proximity(self, p):
        """Proximity membership functions."""
        p = np.clip(p, 0.0, self.R)
        return {
            'VeryNear': self.trapezoid(p, 0.0, 0.04, 0.08, 0.12),
            'Near': self.triangle(p, 0.10, 0.16, 0.22),
            'Far': self.trapezoid(p, 0.20, 0.24, 0.28, self.R)
        }

    def fuzz_side(self, s):
        """Side membership functions."""
        s = np.clip(s, -1.0, 1.0)
        return {
            'Left': self.triangle(s, -1.0, -0.6, -0.2),
            'Center': self.triangle(s, -0.3, 0.0, 0.3),
            'Right': self.triangle(s, 0.2, 0.6, 1.0)
        }

    def compute(self, d_left, d_right):
        """Compute fuzzy controller output."""
        if not (np.isfinite(d_left) and np.isfinite(d_right)):
            return 0.0
        
        p = np.clip(min(d_left, d_right), 0.0, self.R)
        s = np.clip((d_right - d_left) / self.R, -1.0, 1.0)
        
        muP = self.fuzz_proximity(p)
        muS = self.fuzz_side(s)
        c = max(0.0, 1.0 - p / self.R)
        
        # TSK defuzzification
        numer = denom = 0.0
        for prox_name, mu_p in muP.items():
            if mu_p > 0:
                K = self.gains[prox_name]
                for mu_s in muS.values():
                    w = mu_p * mu_s
                    if w > 0:
                        z = K * c * s * self.Vmax
                        numer += w * z
                        denom += w
        
        v = numer / (denom + 1e-12) if denom > 0 else 0.0
        
        # Apply deadzone and smoothing
        if abs(v) < self.deadzone:
            v = 0.0
        v = self.alpha * v + (1.0 - self.alpha) * self.v_prev
        self.v_prev = v
        if self.debug: print(f"[FLC] p={p:.3f} s={s:.3f} c={c:.3f} -> v={v:.4f}")
        return np.clip(v, -self.Vmax, self.Vmax)

