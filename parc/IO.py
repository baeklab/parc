import os
import os.path as osp
import numpy as np
from PIL import Image
import cv2
import skimage
from parc import IO


def parse_data(dir_data: str, time_steps: int, del_t: int) -> np.ndarray:
    """parse the raw data and return numpy arrays with microstructure images and temp/pressure outputs

    Args:
        dir_data (str) : root directory for data set
        time_steps (int) : number of time steps used
        del_t (int) : change of time per each time step

    Returns:
        microstructure_data (np.ndarray) : microstructure data
        output_data (np.ndarray) : fields (e.g., temp, pressure) and change of fields (e.g., temp_dot, pres_dot)
        initial_vals (np.ndarray) : initial values for temperature and pressure
        normalizing_constants (dict) : dictionary containing max and mins for temperature, pressure, and derivatives
    """

    # get image size
    filepath = dir_data + "/microstructures/data_01.pgm"
    img = Image.open(filepath)
    width = img.width
    height = img.height

    # indicate total case numbers in the data set
    case_numbers = len(os.listdir(dir_data + "/microstructures"))

    # initialize and format data arrays
    microstructure_data = np.zeros(
        (case_numbers, width, height, 2)
    )  # [0: microstructure, 1: distance map]
    output_data = np.zeros(
        (case_numbers, width, height, time_steps, 4)
    )  # output dimension 5 shape: [0:T,1:P,2:T_dot,3:P_dot]
    initial_vals = np.zeros(
        (case_numbers, width, height, 2)
    )  # [0: temperature initial value, 1: pressure initial value]

    # create wave map
    wave_map = IO.wave_map(width, height)

    # iterate over cases
    for case_idx in range(1, case_numbers + 1):
        # Load Original Microstructure Image
        img_raw = osp.join(
            dir_data, "microstructures", "data_" + str(format(case_idx, "02d")) + ".pgm"
        ).replace("\\", "/")
        img = cv2.imread(img_raw)
        img = img[:, :, 1]

        # Combine Microstructure image and distance map
        microstructure_data[case_idx - 1, :, :, 0] = img
        microstructure_data[case_idx - 1, :, :, 1] = wave_map
        # flip microstructure data to match temp/pressure data
        microstructure_data[case_idx - 1, :, :, 0] = np.flipud(
            microstructure_data[case_idx - 1, :, :, 0]
        )

        # initialize temperature and pressure
        temp = np.full((width, height), 300.0)
        initial_vals[case_idx - 1, :, :, 0] = temp
        press = np.full((width, height), 0)
        initial_vals[case_idx - 1, :, :, 1] = press

        # iterate over timestep for the fields:
        for time_idx in range(1, time_steps + 1):
            dir_temperature = osp.join(
                dir_data,
                "temperatures",
                "data_" + str(format(case_idx, "02d")),
                "Temp_" + str(format(time_idx, "02d")) + ".txt",
            ).replace("\\", "/")
            temperature_img = np.loadtxt(dir_temperature)
            # reshape temp values to image size
            temp = np.reshape(temperature_img, (width, height))
            # clip the temperature value such that it ranges between 300K and 4000K
            temp = np.clip(temp, 300, 4000)

            # Swap the axis of error cases (temperature)
            if 16 < case_idx < 23:
                temp = np.swapaxes(temp, 0, 1)

            output_data[case_idx - 1, :, :, time_idx - 1, 0] = temp

            dir_pressure = osp.join(
                dir_data,
                "pressures",
                "data_" + str(format(case_idx, "02d")),
                "pres_" + str(format(time_idx, "02d")) + ".txt",
            ).replace("\\", "/")
            pressure_img = np.loadtxt(dir_pressure)
            # reshape pressure values to image size
            pressure = np.reshape(pressure_img, (width, height))

            # Swap the axis of error cases (pressure)
            if 16 < case_idx < 23:
                pressure = np.swapaxes(pressure, 0, 1)

            output_data[case_idx - 1, :, :, time_idx - 1, 1] = pressure

            # Calculate T_dot --> T_dot = (T(t+del_t)-T(t))/del_t
            currentT = output_data[case_idx - 1, :, :, time_idx - 1, 0]
            currentP = output_data[case_idx - 1, :, :, time_idx - 1, 1]
            if time_idx == 1:
                previousT = initial_vals[case_idx - 1, :, :, 0]
                previousP = initial_vals[case_idx - 1, :, :, 1]
            else:
                previousT = output_data[case_idx - 1, :, :, time_idx - 2, 0]
                previousP = output_data[case_idx - 1, :, :, time_idx - 2, 1]
            Tdot = (currentT - previousT) / del_t
            Pdot = (currentP - previousP) / del_t

            # Save Tdot and Pdot into output data array
            output_data[case_idx - 1, :, :, time_idx - 1, 2] = Tdot
            output_data[case_idx - 1, :, :, time_idx - 1, 3] = Pdot

    # calculate max and min of fields
    T_max = np.amax(output_data[:, :, :, :, 0])
    T_min = np.amin(output_data[:, :, :, :, 0])
    P_max = np.amax(output_data[:, :, :, :, 1])
    P_min = np.amin(output_data[:, :, :, :, 1])
    Tdot_max = np.amax(output_data[:, :, :, :, 2])
    Tdot_min = np.amin(output_data[:, :, :, :, 2])
    Pdot_max = np.amax(output_data[:, :, :, :, 3])
    Pdot_min = np.amin(output_data[:, :, :, :, 3])

    # create dictionary for the mins and maxs of the data fields
    normalizing_constants = {
        "Pressure": {
            "min": P_min,
            "max": P_max,
        },
        "Temperature": {
            "min": T_min,
            "max": T_max,
        },
        "Pressure_gradient": {
            "min": Pdot_min,
            "max": Pdot_max,
        },
        "Temperature_gradient": {
            "min": Tdot_min,
            "max": Tdot_max,
        },
    }

    # Normalize initial values to range [-1,1]
    for channel in range(0, 2):
        norm_max = np.amax(output_data[:, :, :, :, channel])
        norm_min = np.amin(output_data[:, :, :, :, channel])
        initial_vals[:, :, :, channel] = (initial_vals[:, :, :, channel] - norm_min) / (
            norm_max - norm_min
        )
        initial_vals[:, :, :, channel] = (initial_vals[:, :, :, channel] * 2.0) - 1.0

    # Normalize fields to range [-1,1]
    for channel in range(0, 4):
        norm_max = np.amax(output_data[:, :, :, :, channel])
        norm_min = np.amin(output_data[:, :, :, :, channel])
        output_data[:, :, :, :, channel] = (
            output_data[:, :, :, :, channel] - norm_min
        ) / (norm_max - norm_min)
        output_data[:, :, :, :, channel] = (
            output_data[:, :, :, :, channel] * 2.0
        ) - 1.0
        print("max and min of channel " + str(channel) + " are: ", norm_max, norm_min)

    # Normalize input data to range [-1,1]
    microstructure_data[:, :, :, 0] = (
        microstructure_data[:, :, :, 0] - np.amin(microstructure_data[:, :, :, 0])
    ) / (
        np.amax(microstructure_data[:, :, :, 0])
        - np.amin(microstructure_data[:, :, :, 0])
    )
    microstructure_data[:, :, :, 1] = (
        microstructure_data[:, :, :, 1] - np.amin(microstructure_data[:, :, :, 1])
    ) / (
        np.amax(microstructure_data[:, :, :, 1])
        - np.amin(microstructure_data[:, :, :, 1])
    )
    microstructure_data[:, :, :, 0] = microstructure_data[:, :, :, 0] > 0.5
    microstructure_data = (microstructure_data * 2.0) - 1.0

    output_data = output_data[:, :480, :480, :, :]
    microstructure_data = microstructure_data[:, :480, :480, :]
    initial_vals = initial_vals[:, :480, :480, :]

    # downsample to half of image size
    output_data = skimage.measure.block_reduce(output_data, (1, 2, 2, 1, 1), np.max)
    microstructure_data = skimage.measure.block_reduce(
        microstructure_data, (1, 2, 2, 1), np.mean
    )
    microstructure_data[:, :, :, :1] = microstructure_data[:, :, :, :1] > 0
    microstructure_data[:, :, :, :1] = (microstructure_data[:, :, :, :1] * 2.0) - 1.0
    initial_vals = skimage.measure.block_reduce(initial_vals, (1, 2, 2, 1), np.mean)
    initial_vals[:, :, :, :1] = initial_vals[:, :, :, :1] > 0
    initial_vals[:, :, :, :1] = (initial_vals[:, :, :, :1] * 2.0) - 1.0

    print("Finished Processing Data")
    print("shape of microstructure data is: ", microstructure_data.shape)
    print("shape of output data is: ", output_data.shape)

    return microstructure_data, output_data, initial_vals, normalizing_constants


def wave_map(width: int, height: int):
    """Generates Distance map (normalized distance from y-axis) where the size
    of the map is same as the original microsturcture image size

    Args:
        width (int) : size of image width in pixels
        height (int): size of image height in pixels

    Returns:
        wave_map (np.ndarray): distance map for point positioning
    """
    wave_map = np.zeros((width, height))
    for w in range(1, width):
        wave_map[:, w] = w / width
    return wave_map


def split_data(
    data_in: np.ndarray, output_data: np.ndarray, initial_vals: np.ndarray, splits: list
) -> np.ndarray:
    """split the data into training, validation, and testing cases

    Args:
        data_in (np.ndarray): microstructure data
        output_data (np.ndarray): temp/pressure/temperature_dot/pressure_dot outputs
        initial_vals (np.ndarray): initial temp and pressure values
        splits (list[int]): train, val, test split

    Returns:
        X_train, y_train, X_val, y_val, test_X, test_Y (np.ndarray): split data
    """
    case_numbers = len(output_data)
    train = int(case_numbers * splits[0])
    valid = int(train + (case_numbers * splits[1]))
    test = int(valid + (case_numbers * splits[2]))

    # Training
    X_train = data_in[:train]
    y_train = output_data[:train]
    X_train_init = initial_vals[:train]

    # Validation
    X_val = data_in[train:valid]
    y_val = output_data[train:valid]
    X_val_init = initial_vals[train:valid]

    # Test
    test_X = data_in[valid:test]
    test_Y = output_data[valid:test]
    test_X_init = initial_vals[valid:test]

    return (
        X_train,
        y_train,
        X_train_init,
        X_val,
        y_val,
        X_val_init,
        test_X,
        test_Y,
        test_X_init,
    )


def reshape_old(new_data: np.ndarray, n_fields: int):
    """reshapes data from new format to old (5 dimensional to 4 dimensional), assumes each field has a respective derivative
    Args:
        new_data (np.ndarray): output data in 5 dimensional format
        n_fields (int): number of fields in the input data to be converted (ex. temp, press, etc.)
    Returns:
        old_data (np.ndarray): output data in 4 dimensional format
    """
    case_numbers = new_data.shape[0]
    time_steps = new_data.shape[3]
    img_size = new_data.shape[2]
    old_data = np.zeros((case_numbers, img_size, img_size, (time_steps * n_fields * 2)))
    print("Starting shape of data: ", new_data.shape)

    for field_idx in range(n_fields):
        for case_idx in range(case_numbers):
            for time_idx in range(time_steps):
                old_data[
                    case_idx, :, :, ((n_fields * (time_idx)) + field_idx)
                ] = new_data[case_idx, :, :, time_idx, (field_idx * 2)]
                old_data[
                    case_idx,
                    :,
                    :,
                    (time_steps * n_fields) + (n_fields * (time_idx)) + field_idx,
                ] = new_data[case_idx, :, :, time_idx, (field_idx * 2) + 1]

    print("Reformatted data shape: ", old_data.shape)

    return old_data


def reshape_new(old_data: np.ndarray, n_fields: int):
    """reshapes data from old format to new (4 dimensional to 5 dimensional)
    Args:
        old_data (np.ndarray): output data in 4 dimensional format
        channels (int): number of output channels for reshaping
        fields (int): number of fields needed to reshape
    Returns:
        new_data (np.ndarray): output data in 5 dimensional format
    """
    case_numbers = old_data.shape[0]
    time_steps = int((old_data.shape[3]) / (n_fields * 2))
    img_size = old_data.shape[2]
    new_data = np.zeros((case_numbers, img_size, img_size, time_steps, (n_fields * 2)))
    print("Starting shape of data: ", old_data.shape)

    for field_idx in range(n_fields):
        for case_idx in range(case_numbers):
            for time_idx in range(time_steps):
                new_data[case_idx, :, :, time_idx, (field_idx * 2)] = old_data[
                    case_idx, :, :, ((n_fields * (time_idx)) + field_idx)
                ]
                new_data[case_idx, :, :, time_idx, (field_idx * 2) + 1] = old_data[
                    case_idx,
                    :,
                    :,
                    (time_steps * n_fields) + (n_fields * (time_idx)) + field_idx,
                ]

    print("Reformatted data shape: ", new_data.shape)

    return new_data


def rescale(normalized_data: np.ndarray, norm_min: int, norm_max: int):
    """rescales data from normalized to full scale output
    Args:
        normalized_data (np.ndarray): output data normalized from -1 to 1
        norm_min (int): minimum value for unscaled data
        norm_max (int): maximum value for unscaled data
    Returns:
        rescaled (np.ndarray): unnormalized data
    """
    rescaled = (normalized_data + 1.0) / 2.0
    rescaled = (rescaled * (norm_max - norm_min)) + norm_min

    return rescaled


def calculate_derivative(input_data: np.ndarray, t_idx: list):
    """calculates the derivate for each time step in input data set
    Args:
        input_data (np.ndarray): data to calculate derivative of
        t_idx (list[int]) : time index of the test cases
    Returns:
        derivative (np.ndarray): derivative data
    """
    derivative = np.zeros((input_data.shape[0], input_data.shape[1] - 1))
    for case_idx in range(input_data.shape[0]):
        for time_idx in range(input_data.shape[1] - 1):
            current = input_data[case_idx, time_idx + 1]
            previous = input_data[case_idx, time_idx]
            del_t = t_idx[time_idx + 1] - t_idx[time_idx]
            dot = (current - previous) / del_t
            derivative[case_idx, time_idx] = dot

    return derivative
