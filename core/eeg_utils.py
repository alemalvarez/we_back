import os
from typing import Tuple, Optional, List
from loguru import logger
import numpy as np
import scipy.io as sio  # type: ignore
from scipy import signal  # type: ignore
from typing import Dict

CLASSICAL_BANDS: Dict[str, List[float]] = {
    "Delta (0.5-4 Hz)": [0.5, 4.0],
    "Theta (4-8 Hz)": [4.0, 8.0],
    "Alpha (8-13 Hz)": [8.0, 13.0],
    "Beta1 (13-19 Hz)": [13.0, 19.0],
    "Beta2 (19-30 Hz)": [19.0, 30.0],
    "Gamma (30-70 Hz)": [30.0, 70.0]
}

def load_files_from_folder(folder_path: str) -> Tuple[list[dict], list[str]]:
    """Load all .mat files from the specified folder."""
    contents = []
    names = []
    try:
        for file in os.listdir(folder_path):
            if file.endswith('.mat'):
                contents.append(sio.loadmat(os.path.join(folder_path, file)))
                names.append(file)
    except Exception as e:
        logger.error(f"Error loading mat files from folder: {e}")
        raise e
    return contents, names


def _extract_important_params(cfg_data: dict) -> dict:
    """
    Helper function to extract important parameters from the configuration data.
    
    Args:
        cfg_data: The configuration data from the MATLAB .mat file
        
    Returns:
        dict: A dictionary containing the extracted important parameters
    """
    cfg = cfg_data[0,0][0]
    
    params = {
        'fs': int(cfg['fs'][0][0][0]),  # Sampling rate
        
        # Filtering info
        'filtering': [
            {
                'type': f['type'][0],
                'band': f['band'][0].tolist(),
                'order': int(f['order'][0][0])
            }
            for f in cfg['filtering'][0][0]
        ],
        
        # Trial length in seconds
        'trial_length_secs': float(cfg['trial_length_secs'][0][0][0]),
        
        # Head model info
        'head_model': str(cfg['head_model'][0][0]),

        
        # Source orientation
        'source_orientation': str(cfg['source_orientation'][0][0][0][0]),
        
        # Atlas information
        'atlas': str(cfg['ROIs'][0][0]['Atlas'][0][0][0]),
        
        # Number of discarded ICA components
        'N_discarded_ICA': int(cfg['N_discarded_ICA'][0][0][0])
    }
    
    return params

def _extract_important_params_bbdds(cfg_data: dict) -> dict:
    # Get the first element where actual data starts
    cfg = cfg_data[0][0]
    
    params = {
        'fs': int(cfg['fs'][0][0][0]),  # Sampling rate
        
        # Filtering info
        'filtering': [
            {
                'type': 'BandPass' if f['type'][0][0][0] == 'B' else 'Notch' if f['type'][0][0][0] == 'N' else 'Unknown',
                'band': f['band'][0].tolist(),
                'order': int(f['order'][0][0])
            }
            for f in cfg['filtering'][0][0][0]
        ],
        
        # Trial length in seconds
        'trial_length_secs': float(cfg['artifacts'][0][0][0][0]['trial_length_secs'][0]),

        
        # Number of discarded ICA components
        'N_discarded_ICA': int(cfg['N_discarded_ICA'][0][0][0])
    }
    
    return params

def _flatten_data(data: np.ndarray, cfg: dict) -> np.ndarray:
    """Reshape the data into segments of specified length.
    
    Args:
        data: Input data array of shape (n_total_samples, n_channels)
        cfg: Configuration dict containing 'trial_length_secs' and 'fs'
        
    Returns:
        Reshaped array of shape (n_segments, n_samples_per_segment, n_channels)
    """
    data = data[0, 0]  # Extract actual data from nested structure
    
    n_samples_per_segment = int(cfg['trial_length_secs'] * cfg['fs'])
    n_channels = data.shape[1]  # Should be 68
    
    # Calculate number of complete segments
    n_total_samples = data.shape[0]
    n_segments = n_total_samples // n_samples_per_segment
    
    # Reshape into segments, truncating any incomplete segment
    return data[:n_segments * n_samples_per_segment].reshape(n_segments, n_samples_per_segment, n_channels)

def get_nice_data(
    raw_data: dict, 
    name: str,
    positive_classes_binary: Optional[List[str]] = ['AD'],
    negative_classes_binary: Optional[List[str]] = None,
    ignore_classes_binary: Optional[List[str]] = None,
    comes_from_bbdds: bool = True
) -> Tuple[np.ndarray, dict, Optional[int]]:
    """Get the nice data from the MATLAB .mat file.

    Args:
        raw_data: The raw data loaded from a .mat file.
        name: The name of the file.
        positive_classes_binary: List of filename patterns indicating positive cases for binary classification.
        negative_classes_binary: List of filename patterns indicating negative cases for binary classification.
        ignore_classes_binary: List of filename patterns to ignore entirely for processing.
        comes_from_bbdds: Flag indicating if the data comes from BBDDs structure.

    Returns:
        A tuple (signal_array, config_dict, binary_target) or None.
        signal_array: The processed EEG signal.
        config_dict: Extracted configuration.
        binary_target: 1 if positive, 0 if negative, None if not specified for binary task.
        Returns None if the file is marked to be ignored by ignore_classes_binary.
    """

    # Handle ignored classes first
    if ignore_classes_binary:
        for ignore_class_pattern in ignore_classes_binary:
            if ignore_class_pattern.upper() in name.upper():
                logger.debug(f"File {name} matches ignore pattern '{ignore_class_pattern}'. Skipping this file.")
                return (np.array([]), {}, None)

    data = raw_data['data']
    cfg_data = data['cfg'] # Renamed for clarity before passing to specific extractors

    try:
        cfg = _extract_important_params(cfg_data) if not comes_from_bbdds else _extract_important_params_bbdds(cfg_data)
    except ValueError as e:
        logger.error(f"You probably messed up comes_from_bbdds flag... try the other one!" )
        raise e
    
    cfg['name'] = name

    signal_arr = data['signal'][0, 0] if not comes_from_bbdds else _flatten_data(data['signal'], cfg) # Renamed for clarity

    binary_target: Optional[int] = None
    determined_explicitly = False

    if positive_classes_binary:
        for pos_pattern in positive_classes_binary:
            if pos_pattern.upper() in name.upper():
                binary_target = 1
                determined_explicitly = True
                break
    
    if not determined_explicitly and negative_classes_binary: # Only check if not already positive
        for neg_pattern in negative_classes_binary:
            if neg_pattern.upper() in name.upper():
                binary_target = 0
                determined_explicitly = True
                break
    
    # If binary_target is still None after checking explicit lists, it remains None.
    # The consumer of this function will decide how to handle segments with a None binary_target.

    return signal_arr, cfg, binary_target

def get_spectral_density(
    signal_data: np.ndarray, 
    cfg: dict, 
    nperseg: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Power Spectral Density (PSD) for each segment using Welch's method,
    averaging across channels within each segment.

    Args:
        signal_data (np.ndarray): EEG signal with shape (n_segments, n_samples, n_channels).
        cfg (dict): Configuration dictionary containing at least 'fs'.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - f: Frequencies (shape: n_freqs)
            - Pxx_segments: Power Spectral Density averaged across channels for each segment 
                          (shape: n_segments, n_freqs)
    """
    fs = cfg['fs']
    n_segments, n_samples, n_channels = signal_data.shape
    
    freqs = None
    Pxx_segments = []

    if nperseg is None:
        nperseg = n_samples

    for s in range(n_segments):
        Pxx_channels_in_segment = []
        for c in range(n_channels):
            # Compute PSD for segment s, channel c
            # Use n_samples as the segment length for welch calculation on each segment
            f, P = signal.welch(signal_data[s, :, c], fs=fs, nperseg=nperseg, scaling='density')
            
            if freqs is None:
                freqs = f  # Store frequencies from the first calculation
            Pxx_channels_in_segment.append(P)
        
        # Average PSDs across channels for the current segment
        if Pxx_channels_in_segment: # Ensure there are channels
            Pxx_segment_mean = np.mean(Pxx_channels_in_segment, axis=0)
            Pxx_segments.append(Pxx_segment_mean)
        # else: handle case with 0 channels if necessary, though shape implies >= 1

    # Stack segment PSDs into a single array
    Pxx_segments_stacked = np.stack(Pxx_segments, axis=0)  # Shape: (n_segments, n_freqs)

    if freqs is None:
        # Handle case where there are no segments (or channels)
        logger.warning("No segments found to compute PSD.")
        return np.array([]), np.array([])

    return freqs, Pxx_segments_stacked

def plot_segment(
    segment: np.ndarray,
    cfg: dict,
) -> None:
    """
    Plot information about an EEG segment including time domain signal, 
    power spectral density, filter information, and configuration details.
    
    Args:
        segment (np.ndarray): EEG segment with shape (n_samples, n_channels)
        cfg (dict): Configuration dictionary containing parameters like 'fs'
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract necessary parameters
    fs = cfg.get('fs', 0)
    n_samples, n_channels = segment.shape
    
    # Create time vector
    time = np.arange(n_samples) / fs if fs > 0 else np.arange(n_samples)
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Time domain signal for each channel
    for ch in range(n_channels):
        axs[0].plot(time, segment[:, ch], label=f'Channel {ch+1}')
    
    axs[0].set_title('EEG Signal in Time Domain')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude (μV)')
    axs[0].grid(True)
    if n_channels <= 10:  # Only show legend if not too many channels
        axs[0].legend()
    
    # Plot 2: Power Spectral Density using get_spectral_density function
    # Reshape segment to match the expected input shape (1, n_samples, n_channels)
    segment_reshaped = segment.reshape(1, n_samples, n_channels)
    f, Pxx = get_spectral_density(segment_reshaped, cfg)
    
    # Since get_spectral_density returns averaged PSD across channels,
    # we can directly plot it (Pxx shape is (1, n_freqs))
    axs[1].semilogy(f, Pxx[0], label='Average across channels')
    
    axs[1].set_title('Power Spectral Density (Averaged Across Channels)')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('PSD (μV²/Hz)')
    axs[1].grid(True)
    
    # Add filter information if available in cfg
    if 'filtering' in cfg:
        for filter_info in cfg['filtering']:
            if 'type' in filter_info and 'band' in filter_info:
                filter_type = filter_info['type']
                band = filter_info['band']
                
                if filter_type == 'highpass':
                    axs[1].axvline(x=band[0], color='r', linestyle='--', 
                                  label=f"Highpass {band[0]} Hz")
                elif filter_type == 'lowpass':
                    axs[1].axvline(x=band[0], color='g', linestyle='--', 
                                  label=f"Lowpass {band[0]} Hz")
                elif filter_type == 'bandpass' and len(band) >= 2:
                    axs[1].axvspan(band[0], band[1], 
                                  alpha=0.2, color='yellow', label=f"Bandpass {band} Hz")
                elif filter_type == 'notch':
                    axs[1].axvline(x=band[0], color='b', linestyle=':', 
                                  label=f"Notch {band[0]} Hz")
    
    axs[1].legend()
    
    # Add text with configuration information
    info_text = f"Segment Shape: {segment.shape}\n"
    info_text += f"Sampling Rate: {fs} Hz\n"
    info_text += f"Duration: {n_samples/fs:.2f} s\n"
    
    # Add other relevant config info
    for key, value in cfg.items():
        if key not in ['fs', 'filtering'] and not isinstance(value, (dict, list, np.ndarray)):
            info_text += f"{key}: {value}\n"
    
    plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))  # Adjust layout to make room for text
    plt.show()
