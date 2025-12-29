# controllers/youbot/youbot_auxiliar/Sensors.py
from typing import Tuple, Callable
import numpy as np
import math

class SensorSuite:
    def __init__(self, lidar_low, lidar_high, compass, step_wait: Callable[[float], None] = None):
        self.lidar_low = lidar_low
        self.lidar_high = lidar_high
        self.compass = compass
        self._step_wait = step_wait

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
        if ranges_arr is None:
            ranges_arr = self.get_low_ranges()
        left = right = float('inf')
        if ranges_arr.size and np.isfinite(ranges_arr).any():
            num = len(ranges_arr)
            fov = self.lidar_low.getFov()
            angles = (np.arange(num) - num // 2) * (fov / num)
            left_mask = (angles > 0.20) & (angles < 1.2)
            right_mask = (angles < -0.20) & (angles > -1.2)
            def mv(mask, arr):
                vals = arr[mask & np.isfinite(arr) & (arr > 0)]
                return float(np.min(vals)) if vals.size else float("inf")
            left = mv(left_mask, ranges_arr)
            right = mv(right_mask, ranges_arr)
        return left, right

    def get_cube_lateral_offset(self, cluster_delta: float = 0.05, max_angle_span: float = 0.4) -> float:
            """
            Estimate cube lateral offset (meters) by clustering near-min points and
            returning the lateral coordinate of the cluster centroid. Returns None
            if no valid measurements.
            """
            ranges = self.get_low_ranges()
            if not np.isfinite(ranges).any():
                return None

            num_points = len(ranges)
            min_index = int(np.nanargmin(np.where((np.isfinite(ranges) & (ranges > 0)), ranges, np.inf)))
            min_dist = float(ranges[min_index])
            fov = self.lidar_low.getFov()
            angles = (np.arange(num_points) - num_points // 2) * (fov / num_points)

            # candidate mask: valid, close enough to min, and within max_angle_span from center of cluster
            close_mask = (np.isfinite(ranges) & (ranges > 0) & (ranges <= (min_dist + cluster_delta)))

            # limit angular span around min_index to avoid other objects
            angle_at_min = angles[min_index]
            angular_span_mask = np.abs(angles - angle_at_min) <= max_angle_span

            mask = close_mask & angular_span_mask
            if mask.any():
                d = ranges[mask].astype(np.float64)
                ang = angles[mask]
                xs = d * np.cos(ang)  # forward
                ys = d * np.sin(ang)  # lateral
                centroid_y = float(np.mean(ys))
                return centroid_y

    
            return min_dist * math.sin(angle_at_min)

    def get_high_min_angle(self):
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