import pytest
import numpy as np
from scipy import signal # type: ignore

# Assuming your function is named shannon_entropy_psd
# from your_module import shannon_entropy_psd

from spectral.shannon_entropy import calcular_se as shannon_entropy_psd

from core.eeg_utils import get_spectral_density as compute_psd

DEFAULT_BAND = [0.5, 300.0]
CFG = {
    "fs": 1000
}


class TestShannonEntropyPSD:
    """
    Test suite for Shannon entropy calculated from Power Spectral Density (PSD)
    
    Tests verify compliance with fundamental Shannon entropy properties:
    1. Non-negativity
    2. Maximum entropy for uniform distribution
    3. Zero entropy for deterministic signals
    4. Monotonicity with respect to concentration
    5. Continuity
    6. Additivity for independent components
    7. Scale invariance
    8. Concavity
    """
    
    def setup_method(self):
        """Setup common test parameters"""
        self.fs = 1000  # Sampling frequency
        self.duration = 2.0
        self.t = np.linspace(0, self.duration, int(self.fs * self.duration), endpoint=False)
        self.tolerance = 1e-10
        
    def generate_test_signals(self):
        """Generate various test signals with known properties"""
        signals = {}
        
        # Pure sinusoid (deterministic)
        signals['pure_sine'] = np.sin(2 * np.pi * 50 * self.t)
        
        # White noise (maximum entropy for given variance)
        np.random.seed(42)
        signals['white_noise'] = np.random.normal(0, 1, len(self.t))
        
        # Narrowband signal (low entropy)
        signals['narrowband'] = (np.sin(2 * np.pi * 50 * self.t) + 
                                0.1 * np.sin(2 * np.pi * 51 * self.t))
        
        # Multi-tone signal
        signals['multi_tone'] = (np.sin(2 * np.pi * 20 * self.t) + 
                                np.sin(2 * np.pi * 100 * self.t) + 
                                np.sin(2 * np.pi * 300 * self.t))
        
        # Colored noise (filtered white noise)
        b, a = signal.butter(4, 0.1, 'low')
        signals['colored_noise'] = signal.filtfilt(b, a, signals['white_noise'])
        
        return signals

    
    @pytest.mark.parametrize("signal_name", ['pure_sine', 'white_noise', 'narrowband', 
                                           'multi_tone', 'colored_noise'])
    def test_non_negativity(self, signal_name):
        """Test that entropy is always non-negative"""
        signals = self.generate_test_signals()
        signal = signals[signal_name].reshape(1, len(signals[signal_name]), 1)
        f, psd = compute_psd(signal, CFG)
        psd = psd[0, :]
        
        entropy = shannon_entropy_psd(psd, f, DEFAULT_BAND)
        
        assert entropy >= -self.tolerance, f"Entropy should be non-negative, got {entropy}"
    
    def test_zero_entropy_deterministic(self):
        """Test that pure deterministic signals have zero or very low entropy"""
        signals = self.generate_test_signals()
        signal = signals['pure_sine'].reshape(1, len(signals['pure_sine']), 1)
        f, psd = compute_psd(signal, CFG)
        psd = psd[0, :]
        
        entropy = shannon_entropy_psd(psd, f, DEFAULT_BAND)
        
        # For a pure sinusoid, most energy should be concentrated at one frequency
        # Entropy should be very low (close to zero)
        assert entropy < 1.0, f"Pure sinusoid should have low entropy, got {entropy}"
    
    def test_maximum_entropy_white_noise(self):
        """Test that white noise has higher entropy than structured signals"""
        signals = self.generate_test_signals()
        
        # Compute entropy for different signal types
        entropies = {}
        for name, sig in signals.items():
            signal = sig.reshape(1, len(sig), 1)
            f, psd = compute_psd(signal, CFG)
            psd = psd[0, :]
            entropies[name] = shannon_entropy_psd(psd, f, DEFAULT_BAND)
        
        # White noise should have higher entropy than deterministic signals
        assert entropies['white_noise'] > entropies['pure_sine']
        assert entropies['white_noise'] > entropies['multi_tone']
        
        # White noise should have higher entropy than colored noise (more uniform PSD)
        assert entropies['white_noise'] >= entropies['colored_noise']
    
    def test_monotonicity_with_concentration(self):
        """Test that entropy decreases as energy becomes more concentrated"""
        # Create signals with increasing concentration
        entropies = []
        
        for bandwidth_factor in [1.0, 0.5, 0.25, 0.1]:
            # Create filtered noise with decreasing bandwidth
            np.random.seed(42)
            white_noise = np.random.normal(0, 1, len(self.t))
            
            # Apply low-pass filter with decreasing cutoff
            b, a = signal.butter(4, bandwidth_factor * 0.4, 'low')
            filtered_signal = signal.filtfilt(b, a, white_noise)
            
            filtered_signal = filtered_signal.reshape(1, len(filtered_signal), 1)
            f, psd = compute_psd(filtered_signal, CFG)
            psd = psd[0, :]
            entropy = shannon_entropy_psd(psd, f, DEFAULT_BAND)
            entropies.append(entropy)
        
        # Entropy should generally decrease as bandwidth decreases
        # (allowing for some numerical tolerance)
        for i in range(len(entropies) - 1):
            assert entropies[i] >= entropies[i+1] - 0.1, \
                f"Entropy should decrease with concentration: {entropies}"
    
    def test_continuity(self):
        """Test continuity by slightly perturbing a signal"""
        signals = self.generate_test_signals()
        base_signal = signals['narrowband']
        base_signal = base_signal.reshape(1, len(base_signal), 1)
        f, psd_base = compute_psd(base_signal, CFG)
        psd_base = psd_base[0, :]
        entropy_base = shannon_entropy_psd(psd_base, f, DEFAULT_BAND)
        
        # Add small perturbation
        perturbation_levels = [0.001, 0.01, 0.1]
        
        for eps in perturbation_levels:
            np.random.seed(42)
            noise = np.random.normal(0, eps, len(base_signal))
            perturbed_signal = base_signal + noise
            
            f, psd_perturbed = compute_psd(perturbed_signal, CFG)
            psd_perturbed = psd_perturbed[0, :]
            entropy_perturbed = shannon_entropy_psd(psd_perturbed, f, DEFAULT_BAND)
            
            # Small perturbations should cause small changes in entropy
            relative_change = abs(entropy_perturbed - entropy_base) / abs(entropy_base + 1e-10)
            
            # For small perturbations, relative change should be bounded
            if eps <= 0.01:
                assert relative_change < 1.0, \
                    f"Large entropy change {relative_change} for small perturbation {eps}"
    
    def test_scale_invariance(self):
        """Test that entropy is invariant to signal scaling"""
        signals = self.generate_test_signals()
        base_signal = signals['white_noise']
        
        # Test different scaling factors
        scale_factors = [0.1, 0.5, 2.0, 10.0]
        entropies = []
        
        for scale in scale_factors:
            scaled_signal = scale * base_signal
            scaled_signal = scaled_signal.reshape(1, len(scaled_signal), 1)
            f, psd = compute_psd(scaled_signal, CFG)
            psd = psd[0, :]
            entropy = shannon_entropy_psd(psd, f, DEFAULT_BAND)
            entropies.append(entropy)
        
        # All entropies should be approximately equal
        # (PSD scales with power, but normalized entropy should be invariant)
        reference_entropy = entropies[0]
        for entropy in entropies[1:]:
            relative_diff = abs(entropy - reference_entropy) / abs(reference_entropy + 1e-10)
            assert relative_diff < 0.1, \
                f"Entropy should be scale-invariant: {entropies}"
    
    def test_additivity_independent_signals(self):
        """Test additivity property for independent signal components"""
        # Create two independent narrowband signals at different frequencies
        signal1 = np.sin(2 * np.pi * 20 * self.t)
        signal2 = np.sin(2 * np.pi * 25 * self.t)

        signal1 = signal1.reshape(1, len(signal1), 1)
        signal2 = signal2.reshape(1, len(signal2), 1)
        
        # Compute individual entropies
        f1, psd1 = compute_psd(signal1, CFG)
        f2, psd2 = compute_psd(signal2, CFG)
        
        psd1 = psd1[0, :]
        psd2 = psd2[0, :]
        entropy1 = shannon_entropy_psd(psd1, f1, DEFAULT_BAND)
        entropy2 = shannon_entropy_psd(psd2, f2, DEFAULT_BAND)
        
        # Combine signals elementwise and compute entropy
        combined_signal = signal1 + signal2  # elementwise sum since shapes match

        f_combined, psd_combined = compute_psd(combined_signal, CFG)
        psd_combined = psd_combined[0, :]
        entropy_combined = shannon_entropy_psd(psd_combined, f_combined, DEFAULT_BAND)
        
        # For independent signals in different frequency bands,
        # combined entropy should be related to individual entropies
        # This is a weaker test since exact additivity depends on implementation
        assert entropy_combined >= max(entropy1, entropy2), \
            "Combined entropy should be at least as large as individual entropies"
    
    def test_concavity_property(self):
        """Test concavity through Jensen's inequality"""
        # Create two different PSDs
        signals = self.generate_test_signals()
        
        signal1 = signals['narrowband'].reshape(1, len(signals['narrowband']), 1)
        signal2 = signals['colored_noise'].reshape(1, len(signals['colored_noise']), 1)
        f1, psd1 = compute_psd(signal1, CFG)
        f2, psd2 = compute_psd(signal2, CFG)
        psd1 = psd1[0, :]
        psd2 = psd2[0, :]
        # Ensure PSDs have same length for mixing
        min_len = min(len(psd1), len(psd2))
        psd1 = psd1[:min_len]
        psd2 = psd2[:min_len]
        f1 = f1[:min_len]  # Truncate frequency array too

        psd1_norm = psd1 / np.sum(psd1)  # Now sums to 1.0
        psd2_norm = psd2 / np.sum(psd2)  # Now sums to 1.0
        
        # Test Jensen's inequality: H(λp1 + (1-λ)p2) ≥ λH(p1) + (1-λ)H(p2)
        lambda_val = 0.5
        mixed_psd = lambda_val * psd1_norm + (1 - lambda_val) * psd2_norm

        entropy1 = shannon_entropy_psd(psd1_norm, f1, DEFAULT_BAND)
        entropy2 = shannon_entropy_psd(psd2_norm, f2, DEFAULT_BAND)
        entropy_mixed = shannon_entropy_psd(mixed_psd, f1, DEFAULT_BAND)

        print(entropy1)
        print(entropy2)
        print(entropy_mixed)
        
        expected_lower_bound = lambda_val * entropy1 + (1 - lambda_val) * entropy2
        
        assert entropy_mixed >= expected_lower_bound - self.tolerance, \
            f"Concavity violated: {entropy_mixed} < {expected_lower_bound}"
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        
        # Test with very small PSD values
        small_psd = np.full(100, 1e-15)
        f = np.linspace(0, 1, len(small_psd))
        try:
            entropy_small = shannon_entropy_psd(small_psd, f, DEFAULT_BAND)
            assert not np.isnan(entropy_small), "Should handle small PSD values"
            assert entropy_small >= -self.tolerance, "Should remain non-negative"
        except (ValueError, RuntimeWarning):
            pass  # Acceptable if function explicitly handles this case
        
        # Test with single frequency component (impulse-like PSD)
        impulse_psd = np.zeros(100)
        impulse_psd[50] = 1.0
        f = np.linspace(0, 1, len(impulse_psd))
        entropy_impulse = shannon_entropy_psd(impulse_psd, f, DEFAULT_BAND)
        assert entropy_impulse >= -self.tolerance, "Impulse PSD should have non-negative entropy"
        assert entropy_impulse < 1.0, "Impulse PSD should have low entropy"
    
    def test_normalization_independence(self):
        """Test that entropy is independent of PSD normalization"""
        signals = self.generate_test_signals()
        signal = signals['white_noise'].reshape(1, len(signals['white_noise']), 1)
        f, psd = compute_psd(signal, CFG)
        psd = psd[0, :]
        
        # Test different normalizations
        normalizations = [
            psd,  # Original
            psd / np.sum(psd),  # Normalized to sum to 1
            psd / np.max(psd),  # Normalized to max of 1
            psd / np.trapz(psd, f)  # Normalized by integral
        ]
        
        entropies = [shannon_entropy_psd(p, f, DEFAULT_BAND) for p in normalizations]
        
        # All normalized versions should give similar entropy values
        # (exact equality depends on implementation details)
        reference = entropies[0]
        for entropy in entropies[1:]:
            relative_diff = abs(entropy - reference) / abs(reference + 1e-10)
            # Allow some tolerance for numerical differences
            assert relative_diff < 0.5, \
                f"Entropy should be relatively independent of normalization: {entropies}"

# Example usage and additional utility functions
def run_comprehensive_test():
    """Run all tests and provide summary"""
    pytest.main([__file__, "-v"])

if __name__ == "__main__":
    # You would need to import or define your shannon_entropy_psd function here
    # Example placeholder:
    
    # Run tests
    run_comprehensive_test()