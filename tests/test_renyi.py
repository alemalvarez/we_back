import pytest
import numpy as np

from core.eeg_utils import get_spectral_density
from spectral.renyi_entropy import calcular_re



class TestGetSpectralDensity:
    """Test suite for get_spectral_density function."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for tests."""
        return {'fs': 1000}  # 1000 Hz sampling rate
    
    @pytest.fixture
    def simple_signal(self):
        """Simple test signal: sine wave at 10 Hz."""
        fs = 1000
        t = np.linspace(0, 1, fs, endpoint=False)
        # Create a 10 Hz sine wave
        signal = np.sin(2 * np.pi * 10 * t)
        # Shape: (n_segments=1, n_samples=1000, n_channels=1)
        return signal.reshape(1, -1, 1)
    
    @pytest.fixture
    def multi_channel_signal(self):
        """Multi-channel test signal."""
        fs = 1000
        t = np.linspace(0, 1, fs, endpoint=False)
        # Channel 1: 10 Hz sine wave
        ch1 = np.sin(2 * np.pi * 10 * t)
        # Channel 2: 20 Hz sine wave
        ch2 = np.sin(2 * np.pi * 20 * t)
        # Shape: (n_segments=1, n_samples=1000, n_channels=2)
        signal = np.stack([ch1, ch2], axis=-1)
        return signal.reshape(1, -1, 2)
    
    def test_return_type_and_shape(self, simple_signal, basic_config):
        """Test that function returns correct types and shapes."""
        f, Pxx = get_spectral_density(simple_signal, basic_config)
        
        assert isinstance(f, np.ndarray), "Frequencies should be numpy array"
        assert isinstance(Pxx, np.ndarray), "PSD should be numpy array"
        assert f.ndim == 1, "Frequencies should be 1D array"
        assert Pxx.ndim == 2, "PSD should be 2D array"
        assert Pxx.shape[0] == simple_signal.shape[0], "PSD first dim should match n_segments"
        assert Pxx.shape[1] == len(f), "PSD second dim should match frequency bins"
    
    def test_frequency_range(self, simple_signal, basic_config):
        """Test that frequency range is correct."""
        f, _ = get_spectral_density(simple_signal, basic_config)
        fs = basic_config['fs']
        
        assert f[0] >= 0, "Minimum frequency should be non-negative"
        assert f[-1] <= fs/2, "Maximum frequency should not exceed Nyquist frequency"
        assert np.all(np.diff(f) > 0), "Frequencies should be monotonically increasing"
    
    def test_parseval_theorem(self, simple_signal, basic_config):
        """Test Parseval's theorem: energy conservation between time and frequency domains."""
        f, Pxx = get_spectral_density(simple_signal, basic_config)
        fs = basic_config['fs']
        
        # Time domain energy (per segment, per channel)
        time_energy = np.mean(simple_signal[0, :, 0] ** 2)
        
        # Frequency domain energy (integrate PSD)
        # Note: Need to account for scaling factors in Welch's method
        freq_energy = np.trapz(Pxx[0], f)
        
        # Should be approximately equal (allowing for numerical errors and windowing effects)
        ratio = freq_energy / time_energy
        assert 0.1 < ratio < 10, f"Energy ratio {ratio} should be reasonable"
    
    def test_peak_frequency_detection(self, simple_signal, basic_config):
        """Test that the function detects the correct peak frequency."""
        f, Pxx = get_spectral_density(simple_signal, basic_config)
        
        # Find peak frequency
        peak_idx = np.argmax(Pxx[0])
        peak_freq = f[peak_idx]
        
        # Should be close to 10 Hz (the input sine wave frequency)
        assert abs(peak_freq - 10) < 2, f"Peak frequency {peak_freq} should be near 10 Hz"
    
    def test_multi_channel_averaging(self, multi_channel_signal, basic_config):
        """Test that multi-channel signals are properly averaged."""
        f, Pxx = get_spectral_density(multi_channel_signal, basic_config)
        
        # The result should show peaks at both 10 Hz and 20 Hz
        # since we're averaging across channels
        peak_indices = np.argsort(Pxx[0])[-2:]  # Top 2 peaks
        peak_freqs = f[peak_indices]
        
        # Should have peaks near 10 Hz and 20 Hz
        assert any(abs(pf - 10) < 2 for pf in peak_freqs), "Should detect 10 Hz peak"
        assert any(abs(pf - 20) < 2 for pf in peak_freqs), "Should detect 20 Hz peak"
    
    def test_multiple_segments(self, basic_config):
        """Test handling of multiple segments."""
        fs = basic_config['fs']
        t = np.linspace(0, 1, fs, endpoint=False)
        
        # Create 3 segments with different frequencies
        seg1 = np.sin(2 * np.pi * 10 * t).reshape(1, -1, 1)
        seg2 = np.sin(2 * np.pi * 15 * t).reshape(1, -1, 1)
        seg3 = np.sin(2 * np.pi * 20 * t).reshape(1, -1, 1)
        
        signal = np.concatenate([seg1, seg2, seg3], axis=0)
        f, Pxx = get_spectral_density(signal, basic_config)
        
        assert Pxx.shape[0] == 3, "Should have 3 segments"
        
        # Each segment should have its peak at the correct frequency
        for i, expected_freq in enumerate([10, 15, 20]):
            peak_idx = np.argmax(Pxx[i])
            peak_freq = f[peak_idx]
            assert abs(peak_freq - expected_freq) < 2, \
                f"Segment {i} peak {peak_freq} should be near {expected_freq} Hz"
    
    def test_noise_floor(self, basic_config):
        """Test behavior with white noise."""
        # White noise should have relatively flat spectrum
        np.random.seed(42)
        noise = np.random.randn(1, 1000, 1)
        
        f, Pxx = get_spectral_density(noise, basic_config)
        
        # For white noise, PSD should be relatively flat
        # Check that variance across frequencies is not too high
        psd_std = np.std(Pxx[0])
        psd_mean = np.mean(Pxx[0])
        cv = psd_std / psd_mean  # coefficient of variation
        
        assert cv < 2, f"White noise should have relatively flat spectrum, CV={cv}"
    
    def test_input_validation(self, basic_config):
        """Test input validation and error handling."""
        # Wrong shape - should be 3D
        with pytest.raises((ValueError, IndexError)):
            get_spectral_density(np.random.randn(100), basic_config)
        
        # Missing fs in config
        with pytest.raises(KeyError):
            get_spectral_density(np.random.randn(1, 100, 1), {})
    
    def test_zero_signal(self, basic_config):
        """Test behavior with zero signal."""
        zero_signal = np.zeros((1, 1000, 1))
        f, Pxx = get_spectral_density(zero_signal, basic_config)
        
        # PSD of zero signal should be close to zero
        assert np.all(Pxx >= 0), "PSD should be non-negative"
        assert np.max(Pxx) < 1e-10, "PSD of zero signal should be very small"


class TestCalcularRE:
    """Test suite for calcular_re (Rényi entropy) function."""
    
    @pytest.fixture
    def uniform_psd(self):
        """Uniform PSD for maximum entropy tests."""
        f = np.linspace(0, 50, 100)
        psd = np.ones_like(f)
        return psd, f
    
    @pytest.fixture
    def delta_psd(self):
        """Delta-like PSD for minimum entropy tests."""
        f = np.linspace(0, 50, 100)
        psd = np.zeros_like(f)
        psd[-1] = 1.0  # Single peak
        return psd, f
    
    @pytest.fixture
    def realistic_psd(self):
        """Realistic PSD with 1/f characteristics."""
        f = np.linspace(1, 50, 100)
        psd = 1 / f  # 1/f noise
        return psd, f
    
    def test_return_type(self, uniform_psd):
        """Test return type is correct."""
        psd, f = uniform_psd
        banda = [10.0, 40.0]
        result = calcular_re(psd, f, banda, q_param=2.0)
        
        assert isinstance(result, (float, type(None))), "Should return float or None"
    
    def test_uniform_distribution_max_entropy(self, uniform_psd):
        """Test that uniform distribution gives maximum entropy."""
        psd, f = uniform_psd
        banda = [10.0, 40.0]
        
        # For uniform distribution, entropy should be high
        entropy_uniform = calcular_re(psd, f, banda, q_param=2.0)
        
        # Compare with a peaked distribution
        psd_peaked = psd.copy()
        idx_band = (f >= banda[0]) & (f <= banda[1])
        psd_peaked[idx_band] = 0
        psd_peaked[np.argmin(np.abs(f - 25))] = np.sum(psd[idx_band])  # Put all power at 25 Hz
        
        entropy_peaked = calcular_re(psd_peaked, f, banda, q_param=2.0)
        
        if entropy_uniform is not None and entropy_peaked is not None:
            assert entropy_uniform > entropy_peaked, \
                "Uniform distribution should have higher entropy than peaked distribution"
    
    def test_delta_distribution_min_entropy(self, delta_psd):
        """Test that delta-like distribution gives minimum entropy."""

        psd, f = delta_psd
        banda = [40.0, 60.0]  # Band containing the delta peak
        
        entropy = calcular_re(psd, f, banda, q_param=.5)
        
        # Delta distribution should have very low entropy
        # (0.0 for single point according to docstring)
        assert entropy is not None, "Should return valid entropy for delta distribution"
        assert entropy <= 0.1, "Delta distribution should have very low entropy"
    
    def test_q_parameter_monotonicity(self, realistic_psd):
        """Test monotonicity property with respect to q parameter."""
        psd, f = realistic_psd
        banda = [10.0, 40.0]
        
        q_values = [0.5, 1.5, 2.0, 3.0]
        entropies = []
        
        for q in q_values:
            entropy = calcular_re(psd, f, banda, q_param=q)
            if entropy is not None:
                entropies.append(entropy)
        
        # For most distributions, Rényi entropy is decreasing in q
        if len(entropies) > 1:
            # Allow for some numerical tolerance
            diffs = np.diff(entropies)
            # Most differences should be negative (decreasing)
            assert np.sum(diffs < 0.01) >= len(diffs) * 0.7, \
                "Entropy should generally decrease with increasing q"
    
    def test_normalization(self, realistic_psd):
        """Test that entropy is properly normalized."""
        psd, f = realistic_psd
        banda = [10.0, 40.0]
        
        entropy = calcular_re(psd, f, banda, q_param=2.0)
        
        if entropy is not None:
            assert 0 <= entropy <= 1, f"Normalized entropy {entropy} should be between 0 and 1"
    
    def test_invalid_q_parameter(self, uniform_psd):
        """Test handling of invalid q parameters."""
        psd, f = uniform_psd
        banda = [10.0, 40.0]
        
        # q = 1 should return None (undefined for Rényi entropy)
        result = calcular_re(psd, f, banda, q_param=1.0)
        assert result is None, "q=1 should return None"
        EPSILON_Q_ONE = 1e-6 
        # q very close to 1 should return None
        result = calcular_re(psd, f, banda, q_param=1.0 + EPSILON_Q_ONE)
        assert result is None, "q very close to 1 should return None"
        
        # Negative q should return None
        result = calcular_re(psd, f, banda, q_param=-0.5)
        assert result is None, "Negative q should return None"
    
    def test_empty_band(self, uniform_psd):
        """Test behavior with empty frequency band."""
        psd, f = uniform_psd
        
        # Band outside frequency range
        banda = [100.0, 200.0]
        result = calcular_re(psd, f, banda, q_param=2.0)
        assert result is None, "Empty band should return None"
        
        # Invalid band (min > max)
        banda = [40.0, 10.0]
        with pytest.raises(ValueError):
            calcular_re(psd, f, banda, q_param=2.0)
    
    def test_zero_power_band(self, uniform_psd):
        """Test behavior when band has zero total power."""
        psd, f = uniform_psd
        banda = [20.0, 30.0]
        
        # Set PSD to zero in the band
        psd_zero = psd.copy()
        idx_band = (f >= banda[0]) & (f <= banda[1])
        psd_zero[idx_band] = 0
        
        result = calcular_re(psd_zero, f, banda, q_param=2.0)
        assert result is None, "Zero power band should return None"
    
    def test_single_point_band(self, uniform_psd):
        """Test behavior with single frequency point in band."""
        psd, f = uniform_psd
        
        # Very narrow band that might contain only one point
        f_center = f[50]
        banda = [f_center - 0.01, f_center + 0.01]
        
        result = calcular_re(psd, f, banda, q_param=2.0)
        
        # According to docstring, single point should give entropy 0.0
        if result is not None:
            assert abs(result) < 1e-10, "Single point should give entropy ≈ 0"
    
    def test_nan_handling(self, uniform_psd):
        """Test handling of NaN values in PSD."""
        psd, f = uniform_psd
        banda = [10.0, 40.0]
        
        # Introduce NaN values
        psd_with_nan = psd.copy()
        psd_with_nan[20:25] = np.nan
        
        result = calcular_re(psd_with_nan, f, banda, q_param=2.0)
        
        # Should handle NaN gracefully (either return valid result or None)
        assert result is None or (isinstance(result, float) and not np.isnan(result)), \
            "Should handle NaN values gracefully"
    
    def test_input_validation(self, uniform_psd):
        """Test input validation and error handling."""
        psd, f = uniform_psd
        banda = [10.0, 40.0]
        
        # Mismatched array lengths
        with pytest.raises((TypeError, ValueError)):
            calcular_re(psd[:-10], f, banda, q_param=2.0)
        
        # Wrong types
        with pytest.raises(TypeError):
            calcular_re("not_array", f, banda, q_param=2.0)
        
        with pytest.raises(TypeError):
            calcular_re(psd, f, "not_list", q_param=2.0)
        
        # Wrong banda format
        with pytest.raises((TypeError, IndexError)):
            calcular_re(psd, f, [10], q_param=2.0)  # Only one element
        
        with pytest.raises((TypeError, IndexError)):
            calcular_re(psd, f, [10, 20, 30], q_param=2.0)  # Too many elements
    
    def test_frequency_band_boundary_conditions(self, uniform_psd):
        """Test behavior at frequency band boundaries."""
        psd, f = uniform_psd
        
        # Band exactly at frequency edges
        banda = [f[0], f[-1]]
        result = calcular_re(psd, f, banda, q_param=2.0)
        assert result is not None, "Full band should return valid result"
        
        # Very small band
        banda = [f[40], f[42]]
        result = calcular_re(psd, f, banda, q_param=2.0)
        # Should return valid result or None
        assert result is None or isinstance(result, float)
    
    def test_scaling_invariance(self, realistic_psd):
        """Test that entropy is invariant to PSD scaling."""
        psd, f = realistic_psd
        banda = [10.0, 40.0]
        
        entropy1 = calcular_re(psd, f, banda, q_param=2.0)
        entropy2 = calcular_re(psd * 10, f, banda, q_param=2.0)  # Scale by 10
        
        if entropy1 is not None and entropy2 is not None:
            assert abs(entropy1 - entropy2) < 1e-10, \
                "Entropy should be invariant to PSD scaling"


class TestIntegration:
    """Integration tests combining both functions."""
    
    def test_spectral_density_to_renyi_entropy_pipeline(self):
        """Test the complete pipeline from signal to entropy."""
        # Create a signal with known spectral characteristics
        fs = 1000
        t = np.linspace(0, 2, 2*fs, endpoint=False)
        
        # Mix of frequencies: strong 10 Hz, weak 25 Hz
        signal = (2 * np.sin(2 * np.pi * 10 * t) + 
                 0.5 * np.sin(2 * np.pi * 25 * t) + 
                 0.1 * np.random.randn(len(t)))
        
        signal = signal.reshape(1, -1, 1)
        cfg = {'fs': fs}
        
        # Get spectral density
        f, Pxx = get_spectral_density(signal, cfg)
        
        # Calculate entropy in different bands
        low_band = [5.0, 15.0]    # Should have high power (low entropy)
        high_band = [20.0, 30.0]  # Should have lower power
        broad_band = [5.0, 45.0]  # Broad band
        
        entropy_low = calcular_re(Pxx[0], f, low_band, q_param=2.0)
        entropy_high = calcular_re(Pxx[0], f, high_band, q_param=2.0)
        entropy_broad = calcular_re(Pxx[0], f, broad_band, q_param=2.0)
        
        # All should be valid
        assert all(e is not None for e in [entropy_low, entropy_high, entropy_broad])
        
        # Broad band should have higher entropy than narrow bands
        assert entropy_broad >= min(entropy_low, entropy_high), \
            "Broad band should have higher or equal entropy"


if __name__ == "__main__":
    # Run with: python -m pytest test_spectral_renyi.py -v
    pytest.main([__file__, "-v"])