def scale_temperature(temperatures, start_ts, max_temp, min_temp):
    """scale back temperature to original scales
    :param temperatures:        (numpy) temperature fields
    :param start_ts:            (int)   start index of the timesteps
    :param max_temp:            (float) maximum temperature value
    :param min_temp:            (float) minimum temperature value

    :return temperatures_scaled:(numpy) scaled temperature fields
    """
    temperatures = temperatures[:, :, :, start_ts:]

    temperatures_scaled = (temperatures + 1.0) / 2.0
    temperatures_scaled = (temperatures_scaled * (max_temp - min_temp)) + min_temp
    return temperatures_scaled
