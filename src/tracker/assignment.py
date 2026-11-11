"""Deterministic minimum-cost Hungarian assignment with explicit unmatched rows."""

from math import isfinite

from .geometry import Box, iou


def hungarian(costs: list[list[float]]) -> list[int]:
    """Return one column per row; requires rows <= columns and finite costs."""
    if not costs:
        return []
    n, m = len(costs), len(costs[0])
    if n > m or m == 0 or any(len(row) != m for row in costs):
        raise ValueError("cost matrix must be rectangular with rows <= columns")
    if any(not isfinite(value) for row in costs for value in row):
        raise ValueError("cost matrix must be finite")
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        minimum = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        j0 = 0
        while True:
            used[j0] = True
            row = p[j0]
            delta, j1 = float("inf"), 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                reduced = costs[row - 1][j - 1] - u[row] - v[j]
                if reduced < minimum[j]:
                    minimum[j], way[j] = reduced, j0
                if minimum[j] < delta:
                    delta, j1 = minimum[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    result = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            result[p[j] - 1] = j - 1
    return result


def match_boxes(tracks: list[Box], detections: list[Box], min_iou: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not 0 <= min_iou <= 1:
        raise ValueError("minimum IoU outside [0, 1]")
    if not tracks:
        return [], [], list(range(len(detections)))
    if not detections:
        return [], list(range(len(tracks))), []
    n, m = len(tracks), len(detections)
    forbidden = 1e6
    unmatched_cost = 1 - min_iou + 1e-6
    costs = [
        [1 - overlap if (overlap := iou(track, detection)) >= min_iou else forbidden
         for detection in detections] + [unmatched_cost] * n
        for track in tracks
    ]
    assignment = hungarian(costs)
    matches = [(i, j) for i, j in enumerate(assignment) if j < m and costs[i][j] < forbidden]
    matched_tracks = {i for i, _ in matches}
    matched_detections = {j for _, j in matches}
    return matches, [i for i in range(n) if i not in matched_tracks], [j for j in range(m) if j not in matched_detections]
