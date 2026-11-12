"""Unchanged DWB formula; historical training loss configuration is unconfirmed."""
import numpy as np


def compute_dwb_weights(loss_0_previous, loss_0_current, loss_1_previous, loss_1_current, T=1):
    """Return the next two loss weights from their current-to-previous ratios."""
    N = 2
    loss_ratio_0 = loss_0_current / loss_0_previous
    loss_ratio_1 = loss_1_current / loss_1_previous
    element_0 = np.exp(loss_ratio_0/T)
    element_1 = np.exp(loss_ratio_1/T)
    total = element_0 + element_1
    return N*element_0 / total, N*element_1 / total
