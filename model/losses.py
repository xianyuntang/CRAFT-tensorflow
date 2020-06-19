from tensorflow.python.keras import backend as K


def loss(region_score_pred, affinity_pred, region_score_true, affinity_true):
    part1 = K.linalg_ops.norm((region_score_pred - region_score_true) ** 2)

    part2 = K.linalg_ops.norm((affinity_pred - affinity_true) ** 2)
    cost = part1 + part2
    return cost
