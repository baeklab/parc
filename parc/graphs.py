import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from parc import losses
from parc import IO
from matplotlib import animation


def visualize_inference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    t_idx: list,
    case_num: int,
    norm_min: int,
    norm_max: int,
):
    """plot the inference results

    Args:
        y_true (np.ndarray): ground truth values
        y_pred (np.ndarray): model prediction values
        t_idx (list[int]): list of the time index to plot
        case_num (int): case number to visualize prediction
        norm_min (int): minimum value of the data set used to scale graphs
        norm_max (int): maximum value of the data set used to scale graphs
    """

    t_idx = [round(item, 2) for item in t_idx]
    fig, ax = plt.subplots(2, len(t_idx), figsize=(len(t_idx) * 2, 8))
    plt.subplots_adjust(
        left=0.125, bottom=0.65, right=0.65, top=0.9, wspace=0.01, hspace=0.04
    )
    for i in range(len(t_idx)):
        # Ground truth graph
        ax[0][i].clear()
        ax[0][i].set_xticks([])
        ax[0][i].set_yticks([])
        ax[0][0].set_ylabel("Ground Truth", color="r")
        ax[0][i].imshow(
            np.squeeze(y_true[case_num, :, :, (i)]),
            cmap="jet",
            vmin=norm_min,
            vmax=norm_max,
        )
        # Prediction graph
        ax[1][0].set_ylabel("Prediction", color="r")
        ax[1][i].clear()
        ax[1][i].set_xticks([])
        ax[1][i].set_yticks([])
        ax[1][i].set_xlabel(str(t_idx[i]) + "ns", color="r")
        ax[1][i].imshow(
            np.squeeze(y_pred[case_num, :, :, (i)]),
            cmap="jet",
            vmin=norm_min,
            vmax=norm_max,
        )
    plt.show()


def plot_rmse(all_rmse: np.ndarray, t_idx: list):
    """Root mean squared error plot, plotted as boxplot
    Args:
        all_rmse (np.ndarray): total root mean squared output from data
        t_idx (list[int]): list of the time index to plot
    """
    sample_name = "RMSE"
    plt.figure(figsize=[17, 4])
    t_idx = [round(item, 2) for item in t_idx]

    plt.boxplot(
        all_rmse,
        whis=[5, 95],
        medianprops=dict(linewidth=0),
        meanline=True,
        showmeans=True,
        showfliers=False,
        labels=None,
        positions=t_idx,
    )
    for i in range(len(all_rmse)):
        plt.scatter(t_idx, all_rmse[i, :], alpha=0.4, color="b")

    # Add labels and title
    plt.title(sample_name)
    plt.xlabel("ns")
    plt.ylabel("RMSE")
    plt.show()


def plot_r2(all_r2: np.ndarray, t_idx: list):
    """R2 score plot using r2 scores calculated in losses, plotted as a boxplot
    Args:
        all_r2 (np.ndarray): R2 score
        t_idx (list[int]): list of the time index to plot
    """

    sample_name = "R2"
    plt.figure(figsize=[17, 4])
    t_idx = [round(item, 2) for item in t_idx]

    plt.boxplot(
        all_r2,
        whis=[5, 95],
        medianprops=dict(linewidth=0),
        meanline=True,
        showmeans=True,
        showfliers=False,
        positions=t_idx,
    )
    for i in range(len(all_r2)):
        plt.scatter(t_idx, all_r2[i, :], alpha=0.4, color="b")

    # Add labels and title
    plt.title(sample_name)
    plt.xlabel("ns")
    plt.ylabel("R2")
    plt.show()


def plot_hotspot_area(
    y_true: np.ndarray, y_pred: np.ndarray, t_idx: list, threshold: int
):
    """plot of the average hotspot area
    Args:
        y_true (np.ndarray): true values for temp/pressure found in input dataset
        y_pred (np.ndarray): model predicted values for temp/press
        t_idx (list[int]): list of the time index to plot
        threshold (int): temperature at which a hotspot is detected
    """
    mean_error = losses.calculate_avg_sensitivity(y_true, y_pred, t_idx, threshold)
    area_mean = mean_error.get("Prediction").get("Area").get("mean")
    gt_area_mean = mean_error.get("Ground_truth").get("Area").get("mean")
    area_error1 = mean_error.get("Prediction").get("Area").get("error1")
    area_error2 = mean_error.get("Prediction").get("Area").get("error2")
    gt_area_error1 = mean_error.get("Ground_truth").get("Area").get("error1")
    gt_area_error2 = mean_error.get("Ground_truth").get("Area").get("error2")

    plt.figure(figsize=(6, 4))

    plt.plot(t_idx, gt_area_mean, "b-", label="Ground truth")
    plt.plot(t_idx, area_mean, "r-", label="Prediction")

    plt.fill_between(t_idx, gt_area_error1, gt_area_error2, color="blue", alpha=0.2)
    plt.fill_between(t_idx, area_error1, area_error2, color="red", alpha=0.2)

    # Add labels and title
    plt.title(r"Ave. Hotspot Area ($A_{hs}$)", fontsize=14, pad=15)
    # x-axis: time in nanoseconds
    plt.xlabel(r"t ($ns$)", fontsize=12)
    # y-axis: area/time
    plt.ylabel(r"$\dot{A_{hs}}$ ($\mu m^2$)", fontsize=12)
    plt.legend(loc=2, fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.show()


def plot_hotspot_temperature(
    y_true: np.ndarray, y_pred: np.ndarray, t_idx: list, threshold: int
):
    """plot of the average hotspot temperature
    Args:
        y_true (np.ndarray): true values for temp/pressure found in input dataset
        y_pred (np.ndarray): model predicted values for temp/press
        t_idx (list[int]): list of the time index to plot
        threshold (int): temperature at which a hotspot is detected
    """
    mean_error = losses.calculate_avg_sensitivity(y_true, y_pred, t_idx, threshold)
    temp_mean = mean_error.get("Prediction").get("Temperature").get("mean")
    gt_temp_mean = mean_error.get("Ground_truth").get("Temperature").get("mean")
    temp_error1 = mean_error.get("Prediction").get("Temperature").get("error1")
    temp_error2 = mean_error.get("Prediction").get("Temperature").get("error2")
    gt_temp_error1 = mean_error.get("Ground_truth").get("Temperature").get("error1")
    gt_temp_error2 = mean_error.get("Ground_truth").get("Temperature").get("error2")

    plt.figure(figsize=(6, 4))

    plt.plot(t_idx, gt_temp_mean, "b-", label="Ground truth")
    plt.plot(t_idx, temp_mean, "r-", label="Prediction")

    plt.fill_between(t_idx, gt_temp_error1, gt_temp_error2, color="blue", alpha=0.2)
    plt.fill_between(t_idx, temp_error1, temp_error2, color="red", alpha=0.2)

    # Add labels and title
    plt.title(r"Ave. Hotspot Temperature ($T_{hs}$)", fontsize=14, pad=15)
    # x-axis: time in nanoseconds
    plt.xlabel(r"t ($ns$)", fontsize=12)
    # y-axis: temperature/time
    plt.ylabel(r"$\dot{T_{hs}}$ ($K$)", fontsize=12)
    plt.legend(fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.show()


def plot_hotspot_area_dot(
    y_true: np.ndarray, y_pred: np.ndarray, t_idx: list, threshold: int
):
    """plot of the average hotspot area rate of change used to show predicted growth
    Args:
        y_true (np.ndarray): true values for temp/pressure found in input dataset
        y_pred (np.ndarray): model predicted values for temp/press
        t_idx (list[int]): list of the time index to plot
        threshold (int): temperature at which a hotspot is detected
    """
    mean_error = losses.calculate_avg_sensitivity(y_true, y_pred, t_idx, threshold)
    A_dot = mean_error.get("Prediction").get("Area_gradient").get("mean")
    A_dot_gt = mean_error.get("Ground_truth").get("Area_gradient").get("mean")
    area_error1_deriv = mean_error.get("Prediction").get("Area_gradient").get("error1")
    area_error2_deriv = mean_error.get("Prediction").get("Area_gradient").get("error2")
    gt_area_error1_deriv = (
        mean_error.get("Ground_truth").get("Area_gradient").get("error1")
    )
    gt_area_error2_deriv = (
        mean_error.get("Ground_truth").get("Area_gradient").get("error2")
    )

    plt.figure(figsize=(6, 4))

    plt.plot(t_idx[1:], A_dot_gt, "b-", label="Ground truth")
    plt.plot(t_idx[1:], A_dot, "r-", label="Prediction")

    plt.fill_between(
        t_idx[1:], gt_area_error1_deriv, gt_area_error2_deriv, color="blue", alpha=0.2
    )
    plt.fill_between(
        t_idx[1:], area_error1_deriv, area_error2_deriv, color="red", alpha=0.2
    )

    # Add labels and title
    plt.title(r"Ave. Hotspot Area Rate of Change ($\dot{A_{hs}}$)", fontsize=14, pad=15)
    # x-axis: time in nanoseconds
    plt.xlabel(r"t ($ns$)", fontsize=12)
    # y-axis: temperature/time
    plt.ylabel(r"$\dot{A_{hs}}$ ($\mu m^2$/$ns$)", fontsize=12)
    plt.legend(fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.show()


def plot_hotspot_temp_dot(
    y_true: np.ndarray, y_pred: np.ndarray, t_idx: list, threshold: int
):
    """plot of the average hotspot area rate of change used to show predicted growth
    Args:
        y_true (np.ndarray): true values for temp/pressure found in input dataset
        y_pred (np.ndarray): model predicted values for temp/press
        t_idx (list[int]): list of the time index to plot
        threshold (int): temperature at which a hotspot is detected
    """
    mean_error = losses.calculate_avg_sensitivity(y_true, y_pred, t_idx, threshold)
    T_dot = mean_error.get("Prediction").get("Temperature_gradient").get("mean")
    T_dot_gt = mean_error.get("Ground_truth").get("Temperature_gradient").get("mean")
    temp_error1_deriv = (
        mean_error.get("Prediction").get("Temperature_gradient").get("error1")
    )
    temp_error2_deriv = (
        mean_error.get("Prediction").get("Temperature_gradient").get("error2")
    )
    gt_temp_error1_deriv = (
        mean_error.get("Ground_truth").get("Temperature_gradient").get("error1")
    )
    gt_temp_error2_deriv = (
        mean_error.get("Ground_truth").get("Temperature_gradient").get("error2")
    )

    plt.figure(figsize=(6, 4))

    plt.plot(t_idx[1:], T_dot_gt, "b-", label="Ground truth")
    plt.plot(t_idx[1:], T_dot, "r-", label="Prediction")

    plt.fill_between(
        t_idx[1:], gt_temp_error1_deriv, gt_temp_error2_deriv, color="blue", alpha=0.2
    )
    plt.fill_between(
        t_idx[1:], temp_error1_deriv, temp_error2_deriv, color="red", alpha=0.2
    )

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
    plt.show()


def plot_saliency(y_pred: np.ndarray, cn: int, ts: int, threshold: int):
    """plot of the saliency of the predicted values, shows where the growth originates in prediction
    Args:
        y_pred (np.ndarray): model predicted values for temp
        cn (int): which case number to display saliency at
        ts (int): which timestep to display saliency at
        threshold (int): max hotspot temperature threshold
    """
    pred_data = np.squeeze(y_pred[cn, :, :, ts])
    pred_mask = pred_data > threshold

    plt.imshow(np.squeeze(pred_mask), cmap="coolwarm", vmin=-0.0, vmax=1.0)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])


def animation_graph(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    case_num: int,
    norm_min: int,
    norm_max: int,
):
    """creates and saves an animation of the inference results against the ground truth data

    Args:
        y_true (np.ndarray): ground truth values
        y_pred (np.ndarray): model prediction values
        case_num (int): case number to visualize prediction
        norm_min (int): minimum value of the data set used to scale graphs
        norm_max (int): maximum value of the data set used to scale graphs
    """

    fig, ax = plt.subplots(1, 2)

    def iterator_img(i):
        ax[1].clear()
        ax[0].clear()
        fig.suptitle(
            "Case Number: " + str(case_num), fontsize=18, y=0.9, x=0.5, color="black"
        )
        ax[1].set_title("Predicted Result", color="r")
        ax[1].set_xlabel("Time step = " + str(i), color="r")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].imshow(
            np.squeeze(y_pred[case_num, :, :, i]),
            cmap="jet",
            vmin=norm_min,
            vmax=norm_max,
        )
        ax[0].set_title("Ground Truth", color="r")
        ax[0].set_xlabel("Time step = " + str(i), color="r")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].imshow(
            np.squeeze(y_true[case_num, :, :, i]),
            cmap="jet",
            vmin=norm_min,
            vmax=norm_max,
        )

        return fig

    ani = animation.FuncAnimation(
        fig, iterator_img, frames=19, interval=300, blit=False, repeat_delay=1000
    )
    ani.save("Sample " + str(case_num) + " Prediction.mp4", writer="ffmpeg")
