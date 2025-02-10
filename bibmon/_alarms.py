import numpy as np

###############################################################################

def detecOutlier(data, lim, count=False, count_limit=1):
    """
    Detects outliers in the given data using a specified limit.

    Parameters
    ----------
    data: array-like
        The input data to be analyzed.
    lim: float
        The limit value used to detect outliers.
    count: bool, optional
        If True, counts the number of outliers 
        exceeding the limit. Default is False.
    count_limit: int, optional
        The maximum number of outliers allowed. 
        Default is 1.

    Returns
    ----------
    alarm: ndarray or int
        If count is False, returns an array indicating 
        the outliers (0 for values below or equal to lim,
        1 for values above lim).
        If count is True, returns the number of outliers 
        exceeding the limit.
    """

    if np.isnan(data).any():
        data = np.nan_to_num(data)

    if count == False:
        alarm = np.copy(data)
        alarm = np.where(alarm <= lim, 0, alarm)
        alarm = np.where(alarm > lim, 1, alarm)
        return alarm
    else:
        alarm = 0
        local_count = np.count_nonzero(data > lim)
        if local_count > count_limit:
            alarm = +1
        return alarm

###############################################################################

def detect_drift(data, window_size, threshold):
    """
    Detects drift in the given data over a specified window size.

    Parameters
    ----------
    data: array-like
        The input data to be analyzed.
    window_size: int
        The size of the rolling window to calculate the mean.
    threshold: float
        The threshold for detecting drift. If the difference between
        the mean of the current window and the previous window exceeds
        this threshold, a drift is detected.

    Returns
    ----------
    alarm: int
        Returns 1 if drift is detected, otherwise 0.
    """
    if len(data) < window_size * 2:
        raise ValueError("Data length must be at least twice the window size.")

    alarm = 0
    for i in range(window_size, len(data)):
        prev_window_mean = np.mean(data[i - window_size:i])
        current_window_mean = np.mean(data[i:i + window_size])
        if abs(current_window_mean - prev_window_mean) > threshold:
            alarm = 1
            break
    return alarm

###############################################################################

def detect_bias(data, expected_mean, threshold):
    """
    Detects bias in the given data compared to an expected mean.

    Parameters
    ----------
    data: array-like
        The input data to be analyzed.
    expected_mean: float
        The expected mean value of the data.
    threshold: float
        The threshold for detecting bias. If the absolute difference
        between the mean of the data and the expected mean exceeds
        this threshold, a bias is detected.

    Returns
    ----------
    alarm: int
        Returns 1 if bias is detected, otherwise 0.
    """
    alarm = 0
    data_mean = np.mean(data)
    if abs(data_mean - expected_mean) > threshold:
        alarm = 1
    return alarm

###############################################################################

def nelson_rule_1(data, mean, std_dev):
    """
    Nelson Rule 1: Detects if any point is above 3 standard deviations
    from the mean.

    Parameters
    ----------
    data: array-like
        The input data to be analyzed.
    mean: float
        The mean value of the data.
    std_dev: float
        The standard deviation of the data.

    Returns
    ----------
    alarm: int
        Returns 1 if any point is above 3 standard deviations
        from the mean, otherwise 0.
    """
    alarm = 0
    for value in data:
        if abs(value - mean) > 3 * std_dev:
            alarm = 1
            break
    return alarm

###############################################################################

def nelson_rule_2(data, mean):
    """
    Nelson Rule 2: Detects if 9 or more consecutive points are on the
    same side of the mean.

    Parameters
    ----------
    data: array-like
        The input data to be analyzed.
    mean: float
        The mean value of the data.

    Returns
    ----------
    alarm: int
        Returns 1 if 9 or more consecutive points are on the same side
        of the mean, otherwise 0.
    """
    alarm = 0
    consecutive_count = 0
    for value in data:
        if value > mean:
            consecutive_count += 1
        else:
            consecutive_count = 0
        if consecutive_count >= 9:
            alarm = 1
            break
    return alarm