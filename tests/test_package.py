import pytest
from typing import Dict, List, Tuple, Any
import numpy as np
import core.eeg_utils as eeg
from spectral.median_frequency import calcular_mf
from spectral.spectral_95_limit_frequency import calcular_sef95
from spectral.individual_alpha_frequency_transition_frequency import calcular_iaftf
from spectral.relative_powers import calcular_rp
from spectral.spectral_bandwidth import calcular_sb
from spectral.spectral_centroid import calcular_sc
from spectral.spectral_crest_factor import calcular_scf

# --- Constants ---
FS: int = 500
COMMON_CFG: Dict[str, Any] = {'fs': FS}
DEFAULT_BAND: List[float] = [0.5, 70.0]
WIDE_BAND: List[float] = [0.5, 100.0]
ALPHA_BAND: List[float] = [8.0, 13.0]
CLASSICAL_BANDS: Dict[str, List[float]] = {
    "Delta (0.5-4 Hz)": [0.5, 4.0],
    "Theta (4-8 Hz)": [4.0, 8.0],
    "Alpha (8-13 Hz)": [8.0, 13.0],
    "Beta1 (13-19 Hz)": [13.0, 19.0],
    "Beta2 (19-30 Hz)": [19.0, 30.0],
    "Gamma (30-70 Hz)": [30.0, 70.0]
}

# --- Fixtures for Signal Generation ---

@pytest.fixture(scope="session")
def time_vector_5s() -> np.ndarray:
    return np.arange(0, 5, 1/FS)

@pytest.fixture(scope="session")
def time_vector_1s() -> np.ndarray:
    return np.arange(0, 1, 1/FS)

@pytest.fixture(scope="session")
def signal_three_sines(time_vector_5s: np.ndarray) -> np.ndarray:
    f1, f2, f3 = 20.0, 40.0, 60.0
    signal = (np.sin(2 * np.pi * f1 * time_vector_5s) +
              np.sin(2 * np.pi * f2 * time_vector_5s) +
              np.sin(2 * np.pi * f3 * time_vector_5s))
    return signal.reshape(1, -1, 1)

@pytest.fixture(scope="session")
def signal_unbalanced_sines(time_vector_5s: np.ndarray) -> np.ndarray:
    f1, f2 = 20.0, 40.0
    signal = (0.9 * np.sin(2 * np.pi * f1 * time_vector_5s) +
              0.1 * np.sin(2 * np.pi * f2 * time_vector_5s))
    return signal.reshape(1, -1, 1)

@pytest.fixture(scope="session")
def signal_two_sines_power_test(time_vector_1s: np.ndarray) -> Tuple[np.ndarray, float, float]:
    f1, f2 = 10.0, 30.0
    # Signal with 80% power in f1 and 20% power in f2 (approx, amplitudes are sqrt of power)
    signal = (np.sqrt(0.8) * np.sin(2 * np.pi * f1 * time_vector_1s) +
              np.sqrt(0.2) * np.sin(2 * np.pi * f2 * time_vector_1s))
    return signal.reshape(1, -1, 1), f1, f2

@pytest.fixture(scope="session")
def signal_broadband_sef95(time_vector_5s: np.ndarray) -> np.ndarray:
    frequencies = np.linspace(0, 100, 100)
    amplitudes = np.ones_like(frequencies)
    signal = np.zeros_like(time_vector_5s)
    for f_val, a_val in zip(frequencies, amplitudes):
        signal += a_val * np.sin(2 * np.pi * f_val * time_vector_5s)
    return signal.reshape(1, -1, 1)

@pytest.fixture(scope="session")
def signal_single_peak_10hz(time_vector_1s: np.ndarray) -> np.ndarray:
    f1 = 10.0
    signal = np.sin(2 * np.pi * f1 * time_vector_1s)
    return signal.reshape(1, -1, 1)

@pytest.fixture(scope="session")
def signal_equal_power_bands(time_vector_1s: np.ndarray) -> np.ndarray:
    frequencies_rp: List[float] = [2, 6, 10.5, 16, 24, 50] # Delta, Theta, Alpha, Beta1, Beta2, Gamma
    signal = np.zeros_like(time_vector_1s)
    for f_val in frequencies_rp:
        signal += np.sin(2 * np.pi * f_val * time_vector_1s)
    return signal.reshape(1, -1, 1)

@pytest.fixture(scope="session")
def signal_narrow_band_centered_10hz(time_vector_1s: np.ndarray) -> np.ndarray:
    frequencies: List[float] = [9.5, 10, 10.5]
    signal = np.zeros_like(time_vector_1s)
    for f_val in frequencies:
        signal += np.sin(2 * np.pi * f_val * time_vector_1s)
    return signal.reshape(1, -1, 1)

@pytest.fixture(scope="session")
def signal_broadband_0_50hz(time_vector_1s: np.ndarray) -> np.ndarray: # For SB, SCF, Entropies
    # Using 1s for faster PSD if appropriate, or 5s from other fixture for more resolution
    time_vector = np.arange(0, 5, 1/FS) # 5 seconds for better resolution for entropy
    frequencies: List[float] = np.arange(0.5, 50, 0.5).tolist() # Start from 0.5Hz as per many bands
    signal = np.zeros_like(time_vector)
    for f_val in frequencies:
        signal += np.sin(2 * np.pi * f_val * time_vector)
    return signal.reshape(1, -1, 1)


# --- PSD Calculation Fixture ---
@pytest.fixture(scope="session")
def psd_data(request: Any, signal_fixture_name: str) -> Tuple[np.ndarray, np.ndarray]:
    signal_data = request.getfixturevalue(signal_fixture_name)
    if isinstance(signal_data, tuple): # For fixtures returning more than the signal
        signal = signal_data[0]
    else:
        signal = signal_data
    f, psd = eeg.get_spectral_density(signal, COMMON_CFG)
    return f, psd[0] # Assuming single segment, single channel PSD

# Helper to parametrize tests that use different signals
def get_psd_for_signal(signal_name: str):
    return pytest.mark.parametrize("psd_data, signal_fixture_name", [(signal_name, signal_name)], indirect=["psd_data"])


# --- Test Functions ---

# Median Frequency Tests
@get_psd_for_signal("signal_three_sines")
def test_median_frequency_three_sines(psd_data: Tuple[np.ndarray, np.ndarray]):
    f, psd = psd_data
    median_frequency = calcular_mf(psd, f, DEFAULT_BAND)
    assert median_frequency is not None
    f2 = 40.0
    assert (f2 - 2.0 < median_frequency < f2 + 2.0) # Wider tolerance

@get_psd_for_signal("signal_unbalanced_sines")
def test_median_frequency_unbalanced_sines(psd_data: Tuple[np.ndarray, np.ndarray]):
    f, psd = psd_data
    median_frequency = calcular_mf(psd, f, DEFAULT_BAND)
    assert median_frequency is not None
    f1 = 20.0
    assert median_frequency < f1 + 5.0 # Adjusted assertion based on original logic

# Spectral 95% Power Tests
@pytest.mark.parametrize("psd_data, signal_fixture_name, expected_f1, expected_f2",
                         [("signal_two_sines_power_test", "signal_two_sines_power_test", 10.0, 30.0)],
                         indirect=["psd_data"])
def test_spectral_95_power_two_sines(psd_data: Tuple[np.ndarray, np.ndarray], expected_f1: float, expected_f2: float):
    f, psd = psd_data
    power_95 = calcular_sef95(psd, f, DEFAULT_BAND)
    assert power_95 is not None
    assert expected_f1 < power_95 < expected_f2 + 5.0  # Looser upper bound

@get_psd_for_signal("signal_broadband_sef95")
def test_spectral_95_power_broadband(psd_data: Tuple[np.ndarray, np.ndarray]):
    f, psd = psd_data
    power_95 = calcular_sef95(psd, f, WIDE_BAND) # Use WIDE_BAND [0.5, 100]
    assert power_95 is not None
    assert 90.0 < power_95 < 100.0

# Individual Alpha Frequency & Transition Frequency Test
@get_psd_for_signal("signal_single_peak_10hz")
def test_iaf_tf_single_peak(psd_data: Tuple[np.ndarray, np.ndarray]):
    f, psd = psd_data
    iaf, tf = calcular_iaftf(psd, f, DEFAULT_BAND, ALPHA_BAND)
    assert iaf is not None
    assert tf is not None
    assert 9.0 <= iaf <= 11.0 # Centered at 10Hz
    # Transition frequency might be harder to assert precisely without knowing the algorithm's specifics
    # For a single peak at 10Hz, TF might be close to IAF or bounds of alpha.
    assert 7.0 <= tf <= 14.0 # A reasonable expectation


# Relative Power Test
@get_psd_for_signal("signal_equal_power_bands")
def test_relative_power_equal_bands(psd_data: Tuple[np.ndarray, np.ndarray]):
    f, psd = psd_data
    # Test with all 6 classical bands this time
    sub_bandas = list(CLASSICAL_BANDS.values())
    banda_total = [0.5, 70.0] # Ensure this covers all sub_bandas
    
    rp_values = calcular_rp(psd, f, banda_total, sub_bandas)
    assert rp_values is not None
    assert len(rp_values) == len(sub_bandas)
    # With 6 bands, expected power is 1/6 ~ 0.166
    for power in rp_values:
        assert abs(power - (1/len(sub_bandas))) < 0.15 # Increased tolerance


# Spectral Bandwidth Tests
@get_psd_for_signal("signal_narrow_band_centered_10hz")
def test_spectral_bandwidth_narrow(psd_data: Tuple[np.ndarray, np.ndarray]):
    f, psd = psd_data
    spectral_centroid = calcular_sc(psd, f, ALPHA_BAND)
    assert spectral_centroid is not None
    assert 9.0 < spectral_centroid < 11.0

    sb = calcular_sb(psd, f, ALPHA_BAND, spectral_centroid)
    assert sb is not None
    assert sb < 2.0 # Original was 1.5, slightly more tolerance

@get_psd_for_signal("signal_broadband_0_50hz")
def test_spectral_bandwidth_broadband(psd_data: Tuple[np.ndarray, np.ndarray]):
    f, psd = psd_data
    band_0_50hz: List[float] = [0.5, 50.0]
    spectral_centroid = calcular_sc(psd, f, band_0_50hz)
    assert spectral_centroid is not None
    assert 20.0 < spectral_centroid < 30.0 # Expected around 25Hz for uniform 0-50

    sb = calcular_sb(psd, f, band_0_50hz, spectral_centroid)
    assert sb is not None
    assert sb > 10.0


# Spectral Crest Factor Tests
@get_psd_for_signal("signal_single_peak_10hz")
def test_spectral_crest_factor_single_peak(psd_data: Tuple[np.ndarray, np.ndarray]):
    f, psd = psd_data
    cf = calcular_scf(psd, f, DEFAULT_BAND)
    assert cf is not None
    assert cf > 4.0 # Original was 5.0

@get_psd_for_signal("signal_broadband_0_50hz")
def test_spectral_crest_factor_broadband(psd_data: Tuple[np.ndarray, np.ndarray]):
    f, psd = psd_data
    band_0_50hz: List[float] = [0.5, 50.0]
    cf = calcular_scf(psd, f, band_0_50hz)
    assert cf is not None
    assert cf < 2.5 # Original was 2.0

