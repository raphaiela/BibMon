import bibmon
from sklearn.preprocessing import StandardScaler
import numpy as np
from bibmon._alarms import detect_drift

def test_detect_drift():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert detect_drift(data, window_size=3, threshold=2) == 1

def test_detect_drift_no_drift():
    data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert detect_drift(data, window_size=3, threshold=2) == 0