import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
from scipy.ndimage import gaussian_filter
from PIL import ImageFilter, Image
from IPython.display import display


def plot_microstructure(microstructure, idx):
    """plots selected microstructure
    :param microstructure:  (numpy) array of microstructures
    :param idx:             (int)   array index to select the case
    """
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(np.squeeze(microstructure[idx, :, :, 0]), cmap="gray", vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def plot_field_evolution(y_pred, y_true, test_sample_no, state_var_type="temperature"):
    """Visualize the prediction
    :param y_pred:          (numpy) predicted fields
                            [0, ...] predicted fields (F)
                            [1, ...] predicted change of fields (F_dot)
    :param y_true:          (numpy) true label of the fields
    :param test_sample_no:  (int)   array index to select the test case
    :state_var_type:        (str)   indicate which fields to plot the result to apply correct scaling
    """

    # get correct scaling terms
    if state_var_type == "temperature":
        opts = 0
        min_val = 300  # min temperature (K)
        max_val = 4000  # max temperature (K)
        unit = "(K)"
    elif state_var_type == "pressure":
        opts = 1
        min_val = -2  # min pressure (GPa)
        max_val = 50  # max pressure (GPa)
        unit = "(GPa)"
    else:
        print(
            state_var_type,
            " is not supported. Choose either 'temperature' or 'pressure'.",
        )
        return None

    # plot the prediction results
    x_num = np.linspace(0.79, 15.01, 19)  # discrete timesteps
    fig, ax = plt.subplots(2, 7, figsize=(28, 9))
    plt.subplots_adjust(wspace=0.06, hspace=0.07, top=0.85)
    fig.suptitle("Pressure field evolution ($P_s$ = 9.5 GPa) ", fontsize=36)
    for i in range(7):
        ax[0][i].clear()
        ax[0][i].clear()
        ax[0][i].set_xticks([])
        ax[0][i].set_yticks([])
        im = ax[0][i].imshow(
            np.squeeze(y_true[test_sample_no, :, :, (i) * 6 + opts]),
            cmap="jet",
            vmin=-1,
            vmax=1,
        )
        ax[0][i].set_title(("t = " + "%.2f" % x_num[i * 3] + " (ns)"), fontsize=32)

        ax[1][i].set_xticks([])
        ax[1][i].set_yticks([])
        ax[1][i].imshow(
            np.squeeze(y_pred[0][test_sample_no, :, :, (i) * 6 + opts]),
            cmap="jet",
            vmin=-1,
            vmax=1,
        )
    ax[0][0].set_ylabel("Ground truth", fontsize=32)
    ax[1][0].set_ylabel("PARC", fontsize=32)
    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.96, 0.125, 0.015, 0.725])
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="jet"), cax=cbar_ax, ax=im
    )
    cbar.set_label(label=unit, weight="bold", fontsize=36)
    cbar.ax.tick_params(labelsize=36)
    plt.show()


def _calculate_hotspot_tempNarea(Ts, n_timesteps=16):
    """calculates the hotspot temperature and area for single case
    :param test_data:   (numpy) temperature for single case with timesteps; [width, height, timesteps]
    :param n_timesteps: (int)   number of timesteps to calculate the sensitivity
    """
    hotspot_threshold = 875  # (K) Temperature threshold to distinguise between hotspot and non-hotspot area

    hotspot_areas = []
    hotspot_temperatures = []

    # Calculate area and avg hotspot temperature
    for i in range(n_timesteps):
        temp_i = Ts[:, :, i]
        hotspot_mask = temp_i > hotspot_threshold
        hotspot_area = np.count_nonzero(hotspot_mask)

        hotspot_area_rescaled = hotspot_area * ((2 * 25 / 485) ** 2)
        hotspot_areas.append(hotspot_area_rescaled)

        hotspot_temperature = temp_i * hotspot_mask

        if hotspot_area == 0:
            avg_hotspot_temperatures = 0.0
        else:
            avg_hotspot_temperatures = np.sum(hotspot_temperature) / hotspot_area
        hotspot_temperatures.append(avg_hotspot_temperatures)
    return hotspot_areas, hotspot_temperatures


def calculate_hotspot_tempNarea(T_cases, cases_range, n_timesteps):
    """calculates hotspot temperature and area for given cases
    :param T_cases:     (numpy) temperature fieds for different cases
    :param cases_range: (tuple) range of cases to test
    :param n_timesteps: (int) number of timesteps
    :return hs_temp:    (tuple) hotspot temperatures
                        [0] (float) mean
                        [1] (float) 5th percentile
                        [2] (float) 95th percentile
                        [3] (numpy) hotspot temperatures
    :return hs_area:    (tuple) hotspot area
                        [0] (float) mean
                        [1] (float) 5th percentile
                        [2] (float) 95th percentile
                        [3] (numpy) hotspot area
    """
    # calculate average hotspot area and temperature across cases
    hotspot_areas, hotspot_temperatures = [], []
    for i in range(cases_range[0], cases_range[1]):
        hotspot_areas_i, hotspot_temperatures_i = _calculate_hotspot_tempNarea(
            T_cases[i, :, :, :], n_timesteps
        )
        hotspot_areas.append(hotspot_areas_i)
        hotspot_temperatures.append(hotspot_temperatures_i)

    hotspot_areas = np.array(hotspot_areas)
    hotspot_temperatures = np.array(hotspot_temperatures)

    mean_hotspot_temperatures = np.mean(hotspot_temperatures, axis=0)
    mean_hotspot_areas = np.mean(hotspot_areas, axis=0)

    temp_error1 = np.percentile(hotspot_temperatures, 95, axis=0)
    temp_error2 = np.percentile(hotspot_temperatures, 5, axis=0)
    area_error1 = np.percentile(hotspot_areas, 95, axis=0)
    area_error2 = np.percentile(hotspot_areas, 5, axis=0)

    hs_temp = (
        mean_hotspot_temperatures,
        temp_error1,
        temp_error2,
        hotspot_temperatures,
    )
    hs_area = (mean_hotspot_areas, area_error1, area_error2, hotspot_areas)
    return hs_temp, hs_area


def calculate_hotspot_tempNarea_rateOfChange(T_cases, cases_range, n_timesteps):
    """
    :param T_cases:         (numpy) temperature fields for different cases
    :param cases_range:     (tuple) range of cases to test
    :param n_timesteps:     (int)   number of timesteps
    :return rate_hs_temp:   (tuple) the rate of change for hotspot temperature
                            [0] (float) mean
                            [1] (float) 5th percentile
                            [2] (float) 95th percentile
                            [3] (numpy) rate of change for hotspot temperatures
    :return rate_hs_area:   (tuple) the rate of change for hotspot area
                            [0] (float) mean
                            [1] (float) 5th percentile
                            [2] (float) 95th percentile
                            [3] (numpy) rate of change for hotspot area
    """
    hotspot_areas, hotspot_temperatures = [], []
    for i in range(cases_range[0], cases_range[1]):
        hotspot_areas_i, hotspot_temperatures_i = _calculate_hotspot_tempNarea(
            T_cases[i, :, :, :], n_timesteps
        )
        hotspot_areas.append(hotspot_areas_i)
        hotspot_temperatures.append(hotspot_temperatures_i)

    hotspot_areas = np.array(hotspot_areas)
    hotspot_temperatures = np.array(hotspot_temperatures)

    change_hotspot_areas = hotspot_areas[:, 1:] - hotspot_areas[:, 0:-1]
    change_hotspot_areas = change_hotspot_areas / (0.79)

    change_hotspot_temperatures = (
        hotspot_temperatures[:, 1:] - hotspot_temperatures[:, 0:-1]
    )
    change_hotspot_temperatures = change_hotspot_temperatures / (0.79)

    mean_Tdot_temperatures = np.mean(change_hotspot_temperatures, axis=0)
    mean_Tdot_areas = np.mean(change_hotspot_areas, axis=0)

    rate_temp_error1 = np.percentile(change_hotspot_temperatures, 95, axis=0)
    rate_temp_error2 = np.percentile(change_hotspot_temperatures, 5, axis=0)
    rate_area_error1 = np.percentile(change_hotspot_areas, 95, axis=0)
    rate_area_error2 = np.percentile(change_hotspot_areas, 5, axis=0)

    rate_hs_temp = (
        mean_Tdot_temperatures,
        rate_temp_error1,
        rate_temp_error2,
        change_hotspot_temperatures,
    )
    rate_hs_area = (
        mean_Tdot_areas,
        rate_area_error1,
        rate_area_error2,
        change_hotspot_areas,
    )
    return rate_hs_temp, rate_hs_area


def plot_sensitivity(y_true, y_pred, metric, ts):
    """sensitivity plot comparing true and prediction
    :param y_true:  (tuple)
    :param y_pred:  (tuple)
    :param metric:  (str)   metric for plotting. {hs_temp, hs_area, rate_hs_temp, rate_hs_area}
    """

    if metric == "hs_temp" or metric == "hs_area":
        ts = np.linspace(3.16, 15.01, ts)
    elif metric == "rate_hs_temp" or metric == "rate_hs_area":
        ts = np.linspace(3.16, 14.22, ts - 1)
    else:
        print(
            "Wrong metric selection. Possible metrics are: 'hs_temp', 'hs_area', 'rate_hs_temp', 'rate_hs_area"
        )

    col_true, col_pred = "#023FA3", "#D91820"
    plt.figure(figsize=(13, 10))

    # mean values
    plt.plot(ts, y_true[0], color=col_true, lw=2.5, label="Ground truth")
    plt.plot(ts, y_pred[0], color=col_pred, lw=2.5, label="Prediction")

    # plot intervals
    plt.fill_between(ts, y_true[1], y_true[2], color=col_true, alpha=0.2)
    plt.fill_between(ts, y_pred[1], y_pred[2], color=col_pred, alpha=0.2)

    # corresponding titles and wordings based on the metric
    if metric == "hs_temp":
        plt.title(r"Ave. Hotspot Temperature ($T_{hs}$)", fontsize=32, pad=15)
        plt.xlabel(r"t ($ns$)", fontsize=28)
        plt.ylabel(r" $T_{hs}$ ($K$)", fontsize=28)
        plt.axis([3.16, 15.01, 0, 5000])
    elif metric == "hs_area":
        plt.title(r"Hotspot Area ($A_{hs}$)", fontsize=32, pad=15)
        plt.xlabel(r"t ($ns$)", fontsize=28)
        plt.ylabel(r"$A_{hs}$ ($\mu m^2$)", fontsize=28)
        plt.axis([3.16, 15.01, 0, 400])
    elif metric == "rate_hs_temp":
        plt.title(
            r"Ave. Hotspot Temperature Rate of Change ($\dot{T_{hs}}$)",
            fontsize=32,
            pad=15,
        )
        plt.xlabel(r"t ($ns$)", fontsize=28)
        plt.ylabel(r"$\dot{T_{hs}}$ ($K$/$ns$)", fontsize=28)
        plt.axis([3.16, 15.01, -30, 650])
    else:
        plt.title(r"Hotspot Area Rate of Change ($\dot{A_{hs}}$)", fontsize=32, pad=15)
        plt.xlabel(r"t ($ns$)", fontsize=28)
        plt.ylabel(r"$\dot{A_{hs}}$ ($\mu m^2$/$ns$)", fontsize=28)
        plt.axis([3.16, 15.01, 0, 50])

    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.legend(loc=2, fontsize=28)
    plt.show()


def compare_field_evolution(
    y_trues,
    y_preds_parc,
    y_preds_unet,
    y_preds_imaginator,
    case_idx,
    state_var_type="temperature",
):
    """compares with other models by visualization
    :param y_trues:             (numpy) true label of temperature fields
    :param y_preds_parc:        (numpy) predicted temperature fields by PARC
    :param y_preds_unet:        (numpy) predicted temperature fields by UNet
    :param y_preds_imaginator:  (numpy) predicted temperature fields by Imaginator
    :param case_dix:            (int)   array index for case number
    :param state_var_type:      (str)   field type for correct scaling. {temperature, pressure}
    """

    # get correct scaling terms
    if state_var_type == "temperature":
        opts = 0
        min_val = 300  # min temperature (K)
        max_val = 4000  # max temperature (K)
        unit = "(K)"
        title = "Temperature field evolution ($P_s$ = 9.5 GPa)"

    elif state_var_type == "pressure":
        opts = 1
        min_val = -2  # min pressure (GPa)
        max_val = 50  # max pressure (GPa)
        unit = "(GPa)"
        title = "Pressure field evolution ($P_s$ = 9.5 GPa)"
    else:
        print(
            state_var_type,
            " is not supported. Choose either 'temperature' or 'pressure'.",
        )
        return None

    x_num = np.linspace(0.79, 15.01, 19)
    fig, ax = plt.subplots(4, 7, figsize=(28, 17))
    plt.subplots_adjust(wspace=0.0, hspace=0.06)
    fig.suptitle(title, fontsize=36)

    for i in range(7):
        ax[0][i].clear()
        ax[0][i].clear()
        ax[0][i].set_xticks([])
        ax[0][i].set_yticks([])
        im = ax[0][i].imshow(
            np.squeeze(y_trues[case_idx, :, :, (i) * 6 + opts]),
            cmap="jet",
            vmin=-1,
            vmax=1,
        )
        ax[0][i].set_title(("t = " + "%.2f" % x_num[i * 3] + " (ns)"), fontsize=32)

        ax[1][i].set_xticks([])
        ax[1][i].set_yticks([])
        ax[1][i].imshow(
            np.squeeze(y_preds_parc[0][case_idx, :, :, (i) * 6 + opts]),
            cmap="jet",
            vmin=-1,
            vmax=1,
        )

        ax[2][i].set_xticks([])
        ax[2][i].set_yticks([])
        ax[2][i].imshow(
            np.squeeze(y_preds_unet[case_idx, :, :, (i) * 6 + opts]),
            cmap="jet",
            vmin=-1,
            vmax=1,
        )

        ax[3][i].set_xticks([])
        ax[3][i].set_yticks([])
        ax[3][i].imshow(
            np.squeeze(y_preds_imaginator[case_idx, (i) * 3, opts, :, :]),
            cmap="jet",
            vmin=0,
            vmax=1,
        )

    ax[0][0].set_ylabel("Ground truth", fontsize=32)
    ax[1][0].set_ylabel("PARC", fontsize=32)
    ax[2][0].set_ylabel("U-Net", fontsize=32)
    ax[3][0].set_ylabel("Imaginator", fontsize=32)
    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.96, 0.125, 0.015, 0.755])
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="jet"), cax=cbar_ax, ax=im
    )
    cbar.set_label(label=unit, weight="bold", fontsize=36)
    cbar.ax.tick_params(labelsize=36)


def compare_sensitivity_plot(y_true, parc, unet, imaginator, metric):
    """compare models by their sensitivity plots
    :param y_true:      (tuple) [0] mean, [1] 5th percentile, [2] 95th percentile, [3] all values
    :param parc:        (tuple) same as above
    :param unet:        (tuple) same as above
    :param imaginator:  (tuple) same as above
    :param metric:      (str)   to correctly apply apply discritized timesteps. {hs_area, rate_hs_area}
    """
    plt.figure(figsize=(13, 10))

    # get timesteps
    if metric == "hs_area":
        ts = np.linspace(3.16, 15.01, 16)
    elif metric == "rate_hs_area":
        ts = np.linspace(3.16, 14.22, 15)
    else:
        print("Wrong metric selection. Possible metrics are: 'hs_area', 'rate_hs_area")

    plt.plot(ts, y_true[0], color="#023FA3", lw=2.5, label="Ground truth")
    plt.plot(ts, parc[0], color="#D91820", lw=2.5, label="Prediction")
    plt.plot(ts, unet[0], "--", color="#7CFC00", lw=2.5, label="U-Net")
    plt.plot(ts, imaginator[0], "--", color="#FDD451", lw=2.5, label="Imaginator")

    if metric == "hs_area":
        plt.title(r"Hotspot Area ($A_{hs}$)", fontsize=32, pad=15)
        plt.xlabel(r"t ($ns$)", fontsize=28)
        plt.ylabel(r"$A_{hs}$ ($\mu m^2$)", fontsize=28)
        plt.axis([3.16, 15.01, 0, 400])
    else:
        plt.title(r"Hotspot Area Rate of Change ($\dot{A_{hs}}$)", fontsize=32, pad=15)
        plt.xlabel(r"t ($ns$)", fontsize=28)
        plt.ylabel(r"$\dot{A_{hs}}$ ($\mu m^2$/$ns$)", fontsize=28)
        plt.axis([3.16, 15.01, 0, 50])

    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.legend(loc=2, fontsize=28)
    plt.show()


def plot_field_evolution_extendedTS(y_true, y_pred, case_idx, temp_range):
    # parc_pred, test_Y, test_sample_no, temp_range):
    """plot true label and predicted fields for longer timesteps
    :param y_true:      (numpy) true label of fieds
    :param y_pred:      (numpy) predicted fields
    :param case_idx     (int)   array index for specific case
    :param temp_range   (tuple) range of the fields; [0] min, [1] max
    """
    unit = "(K)"
    x_num = np.linspace(0.79, 28.44, 36)

    fig, ax = plt.subplots(4, 9, figsize=(40, 21.5), constrained_layout=True)
    plt.subplots_adjust(wspace=0.05, hspace=0.0)
    fig.suptitle("Temperature field evolution ($P_s$ = 9.5 GPa) ", fontsize=36)
    for i in range(9):
        ax[0][i].clear()
        ax[0][i].clear()
        ax[0][i].set_xticks([])
        ax[0][i].set_yticks([])
        im = ax[0][i].imshow(
            np.squeeze(y_true[case_idx, :, :, (i) * 4]),
            cmap="jet",
            vmin=-1,
            vmax=1,
        )
        ax[0][i].set_title(("t = " + "%.2f" % x_num[i * 2] + " (ns)"), fontsize=32)
        ax[0][i].invert_yaxis()

        ax[1][i].set_xticks([])
        ax[1][i].set_yticks([])
        ax[1][i].imshow(
            np.squeeze(y_pred[0][case_idx, :, :, (i) * 4]),
            cmap="jet",
            vmin=-1,
            vmax=1,
        )

        ax[2][i].set_title(
            ("t = " + "%.2f" % x_num[(i + 9) * 2] + " (ns)"), fontsize=32
        )
        ax[2][i].set_xticks([])
        ax[2][i].set_yticks([])
        ax[2][i].imshow(
            np.squeeze(y_true[case_idx, :, :, (i + 9) * 4]),
            cmap="jet",
            vmin=-1,
            vmax=1,
        )
        ax[2][i].invert_yaxis()

        ax[3][i].set_xticks([])
        ax[3][i].set_yticks([])
        ax[3][i].imshow(
            np.squeeze(y_pred[0][case_idx, :, :, (i + 9) * 4]),
            cmap="jet",
            vmin=-1,
            vmax=1,
        )

    ax[0][0].set_ylabel("Ground truth", fontsize=32)
    ax[1][0].set_ylabel("PARC", fontsize=32)
    ax[2][0].set_ylabel("Ground truth", fontsize=32)
    ax[3][0].set_ylabel("PARC", fontsize=32)
    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.96, 0.145, 0.0125, 0.72])
    norm = mpl.colors.Normalize(vmin=temp_range[0], vmax=temp_range[1])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="jet"), cax=cbar_ax, ax=im
    )
    cbar.set_label(label=unit, weight="bold", fontsize=36)
    cbar.ax.tick_params(labelsize=36)

    fig.legend()
    plt.show()


def compute_saliency_map(cases, cases_init, model, case_idx, cut_off_val):
    """compute saliency map for specified case
    :param cases        (numpy) list of microstructures
    :param cases_init   (numpy) pressure and temperature values for initial timestep
    :param model        (parc)  trained parc model
    :param case_idx     (int)   array index to specify the case
    :param cut_off_val  (int)   threshold for saliency map
    """
    idx = (case_idx, case_idx + 1)
    microstructure = tf.convert_to_tensor(cases[idx[0] : idx[1], :, :, :])
    state_init = cases_init[idx[0] : idx[1], :, :, :]
    with tf.GradientTape() as tape:
        tape.watch(microstructure)
        pred = model([microstructure, state_init])
        loss = -pred[0][:, :, :, 0]
        for i in range(2, 36, 2):  # todo: briefly mention what this iterates
            loss += -pred[0][:, :, :, i]

    gradient = tape.gradient(loss, microstructure)

    # take maximum across channels
    gradient = gradient[:, :, :, 0]
    gradient = gradient.numpy()

    # normaliz between 0 and 1
    min_val, max_val = np.min(gradient), np.max(gradient)
    saliency_map = (gradient - min_val) / (max_val - min_val)
    min_val, max_val = np.min(saliency_map), np.max(saliency_map)

    saliency_map = gaussian_filter(saliency_map, sigma=0.5)
    saliency_map = saliency_map > cut_off_val
    return saliency_map


def plot_saliency_map(saliency_mask, microstructure):
    plt.imshow(np.squeeze(saliency_mask), cmap="RdGy_r", vmin=-0.0, vmax=1.0)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("saliency_map.png", dpi=300)

    plt.imshow(np.squeeze(microstructure), cmap="gray", vmin=-1, vmax=1)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("microstructure.png", dpi=300)

    simg = Image.open("saliency_map.png")
    simg = simg.filter(ImageFilter.SMOOTH_MORE)
    mimg = Image.open("microstructure.png")
    result = Image.blend(simg, mimg, alpha=0.3)
    display(result)
