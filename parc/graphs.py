import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

TS = [
    0.79,
    1.58,
    2.37,
    3.16,
    3.95,
    4.74,
    5.53,
    6.32,
    7.11,
    7.9,
    8.69,
    9.48,
    10.27,
    11.06,
    11.85,
    12.64,
    13.43,
    14.22,
    15.01,
]
TS_sensitivity = [
    4.74,
    5.53,
    6.32,
    7.11,
    7.9,
    8.69,
    9.48,
    10.27,
    11.06,
    11.85,
    12.64,
    13.43,
    14.22,
    15.01,
]


def visualize_inference(
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
    """
    # todo: missing


def plot_rmse(all_rmse, t_idx=TS):
    """Root mean squared error plot, plotted as boxplot
    Args:
        all_rmse : total root mean squared output from data
        t_idx (list[int]): list of the time index to plot. If None plot all timestep
    """
    sample_name = "RMSE"
    plt.figure(figsize=[17, 4])
    plt.boxplot(
        all_rmse,
        whis=[5, 95],
        medianprops=dict(linewidth=0),
        meanline=True,
        showmeans=True,
        showfliers=False,
        labels=None,
        positions=TS,
    )
    for i in range(len(all_rmse)):
        plt.scatter(t_idx, all_rmse[i, :], alpha=0.4, color="b")
        
    # Add labels and title
    plt.title(sample_name)
    plt.xlabel("ns")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()


def plot_r2(all_r2, t_idx=TS):
    """R2 score plot using r2 scores calculated in losses module, plotted as a boxplot
    Args:
        all_r2 : R2 score 
        t_idx (list[int]): list of the time index to plot. If None plot all timestep
    """

    sample_name = "R2"
    plt.figure(figsize=[17, 4])

    plt.boxplot(
        all_r2,
        whis=[5, 95],
        medianprops=dict(linewidth=0),
        meanline=True,
        showmeans=True,
        showfliers=False,
        positions=TS,
    )
    for i in range(len(all_r2)):
        plt.scatter(t_idx, all_r2[i, :], alpha=0.4, color="b")

    # Add labels and title
    plt.title(sample_name)
    plt.xlabel("ns")
    plt.ylabel("R2")
    plt.axis([0.5, 15.5, 0, 1])
    plt.legend()
    plt.show()

def sensitivity_single_sample(test_data):
    """single sample sensitivity calculation
    Args:
        test_data (np.ndarray): prediction temp/press values to test sensitivity
    """
    
    threshold = 0.1554  # 875 Temperature(K), max hotspot temperature threshold

    area_list = []
    # Rescale the output
    test_data = (test_data + 1.0) / 2.0

    # Calculate area and avg hotspot
    for i in range(3, 18):
        pred_slice = test_data[:, :, i]
        pred_mask = pred_slice > threshold
        pred_hotspot_area = np.count_nonzero(pred_mask)
        area_list.append(pred_hotspot_area)

    return area_list

def Calculate_avg_sensitivity(y_true, y_pred, t_idx=TS):
    """average sensitivity calculation between prediction and true values
    Args:
        y_true (np.ndarray): true values for temp/press found in input dataset
        y_pred (np.ndarray): model predicted values for temp/press
        t_idx (list[int]): list of the time index to plot. If None plot all timestep
    """
    
    whole_area = []
    for i in range(8):
        area_gt_list = sensitivity_single_sample(y_pred[i, :, :, :])
        whole_area.append(area_gt_list)

    whole_area = np.array(whole_area)

    area_mean = np.mean(whole_area, axis=0)

    area_error1 = np.percentile(whole_area, 95, axis=0)
    area_error2 = np.percentile(whole_area, 5, axis=0)

    gt_whole_area = []

    for i in range(8):
        area_pred_list = sensitivity_single_sample(y_true[i, :, :, :])
        gt_whole_area.append(area_pred_list)
    gt_whole_area = np.array(gt_whole_area)

    gt_area_mean = np.mean(gt_whole_area, axis=0)

    gt_area_error1 = np.percentile(gt_whole_area, 95, axis=0)
    gt_area_error2 = np.percentile(gt_whole_area, 5, axis=0)

    return (
        area_mean,
        area_error1,
        area_error2,
        gt_area_mean,
        gt_area_error1,
        gt_area_error2,
    )
    
def plot_sensitivity_area(y_true, y_pred, t_idx=TS_sensitivity):
    """plot of the average hotspot area rate of change used to show predicted growth
    Args:
        y_true (np.ndarray): true values for temp/press found in input dataset
        y_pred (np.ndarray): model predicted values for temp/press
        t_idx (list[int]): list of the time index to plot. If None plot all timestep
    """
    (
        area_mean,
        area_error1,
        area_error2,
        gt_area_mean,
        gt_area_error1,
        gt_area_error2,
    ) = Calculate_avg_sensitivity(y_pred[:, :, :, 1:], y_true[:, :, :, 1:])

    plt.figure(figsize=(6, 4))

    plt.plot(t_idx, gt_area_mean, "b-", label="Ground truth")
    plt.plot(t_idx, area_mean, "r-", label="Prediction")

    plt.fill_between(t_idx, gt_area_error1, gt_area_error2, color="blue", alpha=0.2)
    plt.fill_between(t_idx, area_error1, area_error2, color="red", alpha=0.2)

    # Add labels and title
    plt.title(r"Ave. Hotspot Area Rate of Change ($\dot{A_{hs}}$)", fontsize=14, pad=15)
    # x-axis: time in nanoseconds
    plt.xlabel(r"t ($ns$)", fontsize=12)
    # y-axis: area/time
    plt.ylabel(r"$\dot{A_{hs}}$ ($\mu m^2$/$ns$)", fontsize=12)
    plt.legend(loc=2, fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig("area_growth_plot.png")
    plt.show()

def plot_sensitivity_temperature(y_true, y_pred, t_idx=TS):
    """plot of the average hotspot temperature rate of change used to show predicted growth
    Args:
        y_true (np.ndarray): true values for temp found in input dataset
        y_pred (np.ndarray): model predicted values for temp
        t_idx (list[int]): list of the time index to plot. If None plot all timestep
    """  
    (
        temp_mean,
        temp_error1,
        temp_error2,
        gt_temp_mean,
        gt_temp_error1,
        gt_temp_error2,
    ) = Calculate_avg_sensitivity(y_pred[:, :, :, 1:], y_true[:, :, :, 1:])

    plt.figure(figsize=(6, 4))

    plt.plot(t_idx, gt_temp_mean, "b-", label="Ground truth")
    plt.plot(t_idx, temp_mean, "r-", label="Prediction")

    plt.fill_between(t_idx, gt_temp_error1, gt_temp_error2, color="blue", alpha=0.2)
    plt.fill_between(t_idx, temp_error1, temp_error2, color="red", alpha=0.2)

    # Add labels and title
    plt.title(
        r"Ave. Hotspot Temperature Rate of Change ($\dot{T_{hs}}$)", fontsize=14, pad=15
    )
    # x-axis: time in nanoseconds
    plt.xlabel(r"t ($ns$)", fontsize=12)
    # y-axis: temperature/time
    plt.ylabel(r"$\dot{T_{hs}}$ ($K$/$ns$)", fontsize=12)
    plt.legend(fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig("temp_growth_plot.png")
    plt.show()

def plot_saliency(y_pred):
    """plot of the saliency of the predicted values, shows where the growth originates in prediction
    Args:
        y_pred (np.ndarray): model predicted values for temp
    """  
    norm_T_max = 4000
    norm_T_min = 300
    threshold = 875  # 875 Temperature(K), max hotspot temperature threshold

    pred_data = np.squeeze(y_pred[0][1, :, :, 22])
    pred_data = (pred_data + 1.0) / 2.0
    pred_data = (pred_data * (norm_T_max - norm_T_min)) + norm_T_min
    pred_mask = pred_data > threshold

    plt.imshow(np.squeeze(pred_mask), cmap="RdGy_r", vmin=-0.0, vmax=1.0)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
