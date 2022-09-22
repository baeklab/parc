import numpy as np


def visualizeInference(
    data_in: np.ndarray,
    data_out: np.ndarray,
    time_idx: list = None,
    fields: list = None,
):
    """plot the inference results

    Args:
        data_in (np.ndarray):   ground truth label
        data_out (np.ndarray):  model prediction
        time_idx (list[int]):   list of the time index to plot. If None plot all timestep
        fields (list[bool]):    list of selecting which fields to plot. If None, plot all fields

    Returns:
        None
    """


def plot_rmse():
    return None


def plot_r2():
    return None


def plot_sensitivity_area():
    return None


def plot_sensitivity_temperature():
    return None


def plot_saliency():
    return None
