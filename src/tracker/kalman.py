"""Constant-velocity Kalman filter for pedestrian center and box size."""

from math import isfinite

from .geometry import Box


def _transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(column) for column in zip(*a)]


def _multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    bt = _transpose(b)
    return [[sum(x * y for x, y in zip(row, column)) for column in bt] for row in a]


def _inverse(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    augmented = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(a)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("singular covariance")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(n):
            if row != column:
                factor = augmented[row][column]
                augmented[row] = [x - factor * y for x, y in zip(augmented[row], augmented[column])]
    return [row[n:] for row in augmented]


def _measurement(box: Box) -> list[float]:
    cx, cy = box.center
    return [cx, cy, box.w, box.h]


class BoxKalmanFilter:
    """State [cx, cy, w, h, vx, vy, vw, vh], one-frame time step."""

    def __init__(self, box: Box, position_variance: float = 10.0,
                 velocity_variance: float = 100.0, measurement_variance: float = 4.0,
                 process_variance: float = 1.0) -> None:
        if min(position_variance, velocity_variance, measurement_variance, process_variance) <= 0:
            raise ValueError("all variances must be positive")
        self.state = _measurement(box) + [0.0] * 4
        self.covariance = [[float(i == j) * (position_variance if i < 4 else velocity_variance)
                            for j in range(8)] for i in range(8)]
        self.measurement_variance = measurement_variance
        self.process_variance = process_variance

    @property
    def box(self) -> Box:
        cx, cy, width, height = self.state[:4]
        width, height = max(width, 1e-3), max(height, 1e-3)
        return Box(cx - width / 2, cy - height / 2, width, height)

    def predict(self) -> Box:
        transition = [[float(i == j or j == i + 4) for j in range(8)] for i in range(8)]
        self.state = [self.state[i] + self.state[i + 4] if i < 4 else self.state[i] for i in range(8)]
        projected = _multiply(_multiply(transition, self.covariance), _transpose(transition))
        self.covariance = [[projected[i][j] + (self.process_variance if i == j else 0.0)
                            for j in range(8)] for i in range(8)]
        return self.box

    def _innovation(self, box: Box) -> tuple[list[float], list[list[float]]]:
        delta = [measured - predicted for measured, predicted in zip(_measurement(box), self.state)]
        covariance = [[self.covariance[i][j] + (self.measurement_variance if i == j else 0.0)
                       for j in range(4)] for i in range(4)]
        return delta, covariance

    def gating_distance(self, box: Box) -> float:
        delta, covariance = self._innovation(box)
        inverse = _inverse(covariance)
        return sum(delta[i] * inverse[i][j] * delta[j] for i in range(4) for j in range(4))

    def update(self, box: Box) -> Box:
        delta, innovation = self._innovation(box)
        inverse = _inverse(innovation)
        gain = _multiply([row[:4] for row in self.covariance], inverse)
        self.state = [self.state[i] + sum(gain[i][j] * delta[j] for j in range(4)) for i in range(8)]
        old = self.covariance
        # Joseph form retains symmetry and nonnegative covariance under rounding.
        a = [[float(i == j) - (gain[i][j] if j < 4 else 0.0) for j in range(8)] for i in range(8)]
        apa = _multiply(_multiply(a, old), _transpose(a))
        self.covariance = [[apa[i][j] + self.measurement_variance * sum(gain[i][k] * gain[j][k] for k in range(4))
                            for j in range(8)] for i in range(8)]
        if not all(isfinite(value) for value in self.state):
            raise ValueError("non-finite Kalman state")
        return self.box
