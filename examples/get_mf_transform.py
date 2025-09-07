import numpy as np
from scipy import signal  # type: ignore
import matplotlib.pyplot as plt
# Assuming CalculoMF.py is in the same directory or accessible in the path
from spectral.median_frequency import calcular_mf 

# --- 1. Generate Sample Raw Time-Domain Data ---
fs = 1000  # Sampling frequency in Hz
duration = 5  # seconds
t = np.linspace(0, duration, int(fs * duration), endpoint=False) # Time vector

# Create a signal with components at 60 Hz and 150 Hz + some noise
freq1 = 60
freq2 = 150
signal_raw = (0.5 * np.sin(2 * np.pi * freq1 * t) + 
              0.2 * np.sin(2 * np.pi * freq2 * t) + 
              0.1 * np.random.randn(len(t)))

# --- 2. Calculate PSD using Welch's Method ---
# nperseg: Length of each segment. Higher value gives better frequency resolution, 
#          lower value gives better time resolution / more averaging.
#          Often chosen as a power of 2 for FFT efficiency.
nperseg_val = 1024 # Example value, adjust based on your signal properties

# f: Array of sample frequencies.
# Pxx: Power Spectral Density or Power Spectrum of x.
f, Pxx = signal.welch(signal_raw, fs, nperseg=nperseg_val) 

# --- 3. Calculate Median Frequency using your function ---
banda_interes = [10.0, 200.0] # Define the frequency band of interest (use floats)

mf = calcular_mf(Pxx, f, banda_interes)

# --- 4. Output and Visualization ---
print(f"Sampling Frequency: {fs} Hz")
print(f"Calculated PSD using Welch's method with {nperseg_val} points per segment.")

if mf is not None:
    print(f"Median Frequency (MF) in band {banda_interes} Hz: {mf:.2f} Hz")
else:
    print(f"Could not calculate Median Frequency in band {banda_interes} Hz.")

# Plotting
plt.figure(figsize=(10, 7))

plt.subplot(2, 1, 1)
plt.plot(t[:fs*2], signal_raw[:fs*2]) # Plot first 2 seconds of raw signal
plt.title('Raw Time-Domain Signal (first 2 seconds)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.semilogy(f, Pxx) # Plot PSD on a logarithmic scale for better visibility
plt.title('Power Spectral Density (PSD) using Welch Method')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]') # Adjust unit based on your signal's unit
plt.fill_between(f, Pxx, where=((f>=banda_interes[0]) & (f<=banda_interes[1])), color='skyblue', alpha=0.4, label=f'Band {banda_interes} Hz')
if mf is not None:
    plt.axvline(mf, color='r', linestyle='--', label=f'Median Freq ({mf:.2f} Hz)')
plt.legend()
plt.grid(True)
plt.xlim([0, fs/2]) # Show up to Nyquist frequency

plt.tight_layout()
plt.show()
