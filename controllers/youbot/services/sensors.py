# controllers/youbot/youbot_auxiliar/Sensors.py
from typing import Tuple, Callable
import numpy as np
import math

class SensorSuite:
    def __init__(self, lidar_low, lidar_high, compass, step_wait: Callable[[float], None] = None,
             lidar_high_left=None, lidar_high_right=None):
        self.lidar_low = lidar_low
        self.lidar_high = lidar_high
        self.compass = compass
        self._step_wait = step_wait
        self._angles_cache = {}
        self._last_low = None
        self._last_high = None
        if self.lidar_low: self.lidar_low.enablePointCloud()
        if self.lidar_high: self.lidar_high.enablePointCloud()
    
    def snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read both lidars once, cache results, and return (low, high) arrays."""
        low = np.array(self.lidar_low.getRangeImage(), dtype=np.float32) if self.lidar_low else np.array([], dtype=np.float32)
        high = np.array(self.lidar_high.getRangeImage(), dtype=np.float32) if self.lidar_high else np.array([], dtype=np.float32)
        self._last_low = low
        self._last_high = high
        return low, high

    def _angles_for(self, n: int) -> np.ndarray:
        if n not in self._angles_cache:
            fov = self.lidar_low.getFov()
            self._angles_cache[n] = (np.arange(n) - n // 2) * (fov / n)
        return self._angles_cache[n]

    def _min_in_mask(self, arr: np.ndarray, mask: np.ndarray) -> float:
        vals = arr[mask & np.isfinite(arr) & (arr > 0)]
        return float(np.min(vals)) if vals.size else float("inf")

    def read_lidars(self) -> Tuple[float, float]:
        low = np.array(self.lidar_low.getRangeImage()) if self.lidar_low else np.array([])
        high = np.array(self.lidar_high.getRangeImage()) if self.lidar_high else np.array([])
        low = low[np.isfinite(low) & (low > 0)]
        high = high[np.isfinite(high) & (high > 0)]
        min_low = float(np.min(low)) if low.size else float("inf")
        min_high = float(np.min(high)) if high.size else float("inf")
        return min_low, min_high

    def get_low_ranges(self) -> np.ndarray:
        return np.array(self.lidar_low.getRangeImage(), dtype=np.float32)

    def get_high_ranges(self) -> np.ndarray:
        return np.array(self.lidar_high.getRangeImage(), dtype=np.float32)

    def compute_side_mins(self, ranges_arr: np.ndarray = None) -> Tuple[float, float]:
        """Return (left_min, right_min) using fixed angle windows (keeps existing masks)."""
        if ranges_arr is None:
            ranges_arr, _ = self.snapshot()
        left = right = float('inf')
        if ranges_arr.size and np.isfinite(ranges_arr).any():
            num = len(ranges_arr)
            angles = self._angles_for(num)
            left_mask = (angles > 0.20) & (angles < 1.2)
            right_mask = (angles < -0.20) & (angles > -1.2)
            left = self._min_in_mask(ranges_arr, left_mask)
            right = self._min_in_mask(ranges_arr, right_mask)
        return left, right

    def get_cube_lateral_offset(self, cluster_delta: float = 0.05, max_angle_span: float = 0.4) -> float | None:
        """Preserve cluster-based lateral offset estimator; uses snapshot & cached angles."""
        ranges = self.snapshot()[0]
        if not np.isfinite(ranges).any():
            return None

        num_points = len(ranges)
        valid_mask_all = (np.isfinite(ranges) & (ranges > 0))
        if not valid_mask_all.any():
            return None

        min_index = int(np.nanargmin(np.where(valid_mask_all, ranges, np.inf)))
        min_dist = float(ranges[min_index])
        angles = self._angles_for(num_points)

        close_mask = valid_mask_all & (ranges <= (min_dist + cluster_delta))
        angle_at_min = angles[min_index]
        angular_span_mask = np.abs(angles - angle_at_min) <= max_angle_span

        mask = close_mask & angular_span_mask
        if not mask.any():
            return min_dist * math.sin(angle_at_min)

        idxs = np.where(mask)[0]
        segments = np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1)

        chosen_seg = None
        for seg in segments:
            if min_index in seg:
                chosen_seg = seg
                break

        if chosen_seg is None:
            seg_mins = [np.min(ranges[s]) for s in segments]
            chosen_seg = segments[int(np.argmin(seg_mins))]

        d = ranges[chosen_seg].astype(np.float64)
        ang = angles[chosen_seg]
        ys = d * np.sin(ang)
        centroid_y = float(np.mean(ys))
        return centroid_y

    def get_high_min_angle(self):
        ranges_high = getattr(self, "_last_high", None)
        if ranges_high is None:
            ranges_high = self.get_high_ranges()
        if not ranges_high.size or not np.isfinite(ranges_high).any():
            return None
        valid = (np.isfinite(ranges_high) & (ranges_high > 0))
        if not valid.any():
            return None
        idx = int(np.nanargmin(np.where(valid, ranges_high, np.inf)))
        num = len(ranges_high)
        fov = self.lidar_high.getFov()
        return (idx - num // 2) * (fov / num)

    def read_low_filtered(self, samples=3, settle=0.03) -> float:
        vals = []
        for _ in range(samples):
            if self._step_wait:
                self._step_wait(settle)
            low, _ = self.read_lidars()
            if np.isfinite(low):
                vals.append(low)
        return float(np.median(vals)) if vals else float("inf")