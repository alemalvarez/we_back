import pytest
import numpy as np
from scipy import signal # type: ignore

from spectral.tsallis_entropy import calcular_te as tsallis_entropy_psd
from spectral.shannon_entropy import calcular_se as shannon_entropy_psd

from core.eeg_utils import get_spectral_density as compute_psd

DEFAULT_BAND = [0.5, 70.0]
CFG = {
    "fs": 1000
}



class TestTsallisEntropyPSD:
    """
    Test suite for Tsallis entropy calculated from Power Spectral Density (PSD)
    
    Tsallis entropy is defined as: S_q = (1 - Σ p_i^q) / (q - 1) for q ≠ 1
    For q → 1, it reduces to Shannon entropy: S_1 = -Σ p_i log(p_i)
    
    Tests verify compliance with fundamental Tsallis entropy properties:
    1. Non-negativity for q > 0
    2. Convergence to Shannon entropy as q → 1
    3. Monotonicity in q for fixed distribution
    4. Concavity for q > 1, convexity for 0 < q < 1
    5. Additivity properties (pseudo-additivity)
    6. Extremal properties (maximum/minimum values)
    7. Scale invariance properties
    8. Continuity in both q and distribution
    """
    
    def setup_method(self):
        """Setup common test parameters"""
        self.fs = 1000  # Sampling frequency
        self.duration = 2.0
        self.t = np.linspace(0, self.duration, int(self.fs * self.duration), endpoint=False)
        self.tolerance = 1e-10
        self.q_values = [0.5, 0.8, 0.99, 1.01, 1.2, 1.5, 2.0, 3.0]  # Various q parameters
        
    def generate_test_signals(self):
        """Generate various test signals with known properties"""
        signals = {}
        
        # Pure sinusoid (deterministic - concentrated PSD)
        signals['pure_sine'] = np.sin(2 * np.pi * 50 * self.t)
        
        # White noise (uniform-like PSD)
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
    
    def compute_psd(self, x):
        """Compute PSD using Welch's method"""
        f, psd = signal.welch(x, fs=self.fs, nperseg=len(x)//4)
        return f, psd
    
    def shannon_entropy_reference(self, psd, freqs, band=[0.5, 70.0]):
        """Reference Shannon entropy implementation for comparison - band limited"""
        
        # Filter to band of interest
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        psd_band = psd[band_mask]

        print(f"Original PSD length: {len(psd)}")
        print(f"Band-filtered PSD length: {len(psd_band)}")
        print(f"Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
        print(f"Band range: {band[0]} - {band[1]} Hz")
        
        # Check if band has any significant power
        total_power = np.sum(psd_band)
        if total_power <= 1e-9:
            return 0.0
        
        # Normalize within the band
        psd_norm = psd_band / total_power

        # print(psd_norm[:50])
        
        # Remove zeros to avoid log(0)
        psd_norm = psd_norm[psd_norm > 1e-9]
        print(psd_norm.shape)
        print(psd_norm[:50])
        
        if psd_norm.size == 0:
            return 0.0
        
        # Compute Shannon entropy
        shannon_entropy_sum = -np.sum(psd_norm * np.log(psd_norm))

        # return shannon_entropy_sum

        print(f"Shannon entropy pre normalized: {shannon_entropy_sum}")
        print(f"we hare going to divide by {np.log(len(psd_band))}, len(psd) is {len(psd_band)}")

        normalized_shannon_entropy = shannon_entropy_sum / np.log(psd_band.size)
        print(f"Shannon entropy post normalized: {normalized_shannon_entropy}")
        return normalized_shannon_entropy       
    
    @pytest.mark.parametrize("signal_name", ['pure_sine', 'white_noise', 'narrowband', 
                                           'multi_tone', 'colored_noise'])
    @pytest.mark.parametrize("q", [0.5, 1.2, 2.0, 3.0])
    def test_non_negativity_positive_q(self, signal_name, q):
        """Test that Tsallis entropy is non-negative for q > 0"""
        signals = self.generate_test_signals()
        signal = signals[signal_name].reshape(1, len(signals[signal_name]), 1)
        f, psd = compute_psd(signal, CFG)
        psd = psd[0, :]
        
        entropy = tsallis_entropy_psd(psd, f, DEFAULT_BAND, q)
        
        assert entropy >= -self.tolerance, \
            f"Tsallis entropy should be non-negative for q={q}, got {entropy}"

    def test_convergence_to_shannon(self):
        """Test that Tsallis entropy converges to Shannon entropy as q → 1"""
        signals = self.generate_test_signals()



        sig = signals['pure_sine']

        signal = sig.reshape(1, len(sig), 1)
        f, psd = compute_psd(signal, CFG)
        psd = psd[0, :]


        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(8, 4))
        # plt.plot(f, psd, label="PSD")
        # plt.axvspan(DEFAULT_BAND[0], DEFAULT_BAND[1], color='orange', alpha=0.2, label=f"Band {DEFAULT_BAND[0]}-{DEFAULT_BAND[1]} Hz")
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("PSD")
        # plt.title("Power Spectral Density and Default Band")
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        # psd_norm = psd / np.sum(psd)

        shannon_entropy_ref = self.shannon_entropy_reference(psd, f, DEFAULT_BAND)
        shannon_entropy = shannon_entropy_psd(psd, f, DEFAULT_BAND)
        tsallis_entropy = tsallis_entropy_psd(psd, f, DEFAULT_BAND, 0.99)

        print("Shannon entropy reference")
        print(shannon_entropy_ref)
        print("Shannon entropy")
        print(shannon_entropy)
        print("Tsallis entropy")
        print(tsallis_entropy)

        # assert 1==2
        
        for signal_name, sig in signals.items():
            print(signal_name)
            signal = sig.reshape(1, len(sig), 1)
            f, psd = compute_psd(signal, CFG)
            psd = psd[0, :]
            shannon_entropy = self.shannon_entropy_reference(psd, f, DEFAULT_BAND)
            shannon_entropy = shannon_entropy_psd(psd, f, DEFAULT_BAND)
            
            # Test convergence from both sides of q = 1
            q_values_near_1 = [0.99, 0.9999999, 1.001, 1.01]
            
            for q in q_values_near_1:
                tsallis_entropy = tsallis_entropy_psd(psd, f, DEFAULT_BAND, q)
                relative_error = abs(tsallis_entropy - shannon_entropy) / abs(shannon_entropy + 1e-10)

                print(tsallis_entropy)
                print(shannon_entropy)
                print(q)
                
                # Closer to q=1 should give smaller error
                tolerance = 0.1 if abs(q - 1) > 0.01 else 0.01
                assert relative_error < tolerance, \
                    f"Tsallis entropy should converge to Shannon as q→1: " \
                    f"q={q}, Tsallis={tsallis_entropy}, Shannon={shannon_entropy}"
    
    @pytest.mark.parametrize("signal_name", ['white_noise', 'narrowband', 'colored_noise'])
    def test_monotonicity_in_q(self, signal_name):
        """Test monotonicity properties of Tsallis entropy with respect to q"""
        signals = self.generate_test_signals()
        signal = signals[signal_name].reshape(1, len(signals[signal_name]), 1)
        f, psd = compute_psd(signal, CFG)
        psd = psd[0, :]
        
        # For non-uniform distributions, test expected monotonicity
        q_sequence = [0.5, 0.8, 1.2, 1.5, 2.0, 2.5]
        entropies = []
        
        for q in q_sequence:
            entropy = tsallis_entropy_psd(psd, f, DEFAULT_BAND, q)
            entropies.append(entropy)
        
        # For most distributions, Tsallis entropy should change monotonically with q
        # The exact direction depends on the distribution characteristics
        # We test that the function is well-behaved (no wild oscillations)
        differences = np.diff(entropies)
        sign_changes = np.sum(np.diff(np.sign(differences)) != 0)
        
        # Should not have too many sign changes (indicates monotonic or unimodal behavior)
        assert sign_changes <= 2, \
            f"Too many non-monotonic changes in Tsallis entropy vs q: {entropies}"
    
    def test_concavity_convexity_properties(self):
        """Test concavity/convexity properties for different q values"""
        signals = self.generate_test_signals()
        
        # Use two different PSDs for mixing
        signal1 = signals['narrowband'].reshape(1, len(signals['narrowband']), 1)
        signal2 = signals['colored_noise'].reshape(1, len(signals['colored_noise']), 1)
        f1, psd1 = compute_psd(signal1, CFG)
        f2, psd2 = compute_psd(signal2, CFG)
        psd1 = psd1[0, :]
        psd2 = psd2[0, :]
        
        # Ensure same length and normalize PSDs
        min_len = min(len(psd1), len(psd2))
        psd1 = psd1[:min_len]
        psd2 = psd2[:min_len]

        # Normalize the PSDs
        psd1 = psd1 / np.sum(psd1) if np.sum(psd1) > 0 else psd1
        psd2 = psd2 / np.sum(psd2) if np.sum(psd2) > 0 else psd2

        lambda_val = 0.5
        mixed_psd = lambda_val * psd1 + (1 - lambda_val) * psd2

        q_concave = [0.5, 0.8]        # Should be concave (0 < q < 1)
        
        for q in q_concave:
            entropy1 = tsallis_entropy_psd(psd1, f1, DEFAULT_BAND, q)
            entropy2 = tsallis_entropy_psd(psd2, f2, DEFAULT_BAND, q)
            entropy_mixed = tsallis_entropy_psd(mixed_psd, f1, DEFAULT_BAND, q)
            
            expected_linear = lambda_val * entropy1 + (1 - lambda_val) * entropy2
            
            # For 0 < q < 1, should be concave: S(λp1 + (1-λ)p2) ≥ λS(p1) + (1-λ)S(p2)
            assert entropy_mixed >= expected_linear - self.tolerance, \
                f"Concavity violated for q={q}: {entropy_mixed} < {expected_linear}"
        
    # THERE IS NO CONVEXITY FOR TSALLIS ENTROPY
    
    def test_pseudo_additivity(self):
        """Test pseudo-additivity property of Tsallis entropy"""
        # For independent systems A and B:
        # S_q(A+B) = S_q(A) + S_q(B) + (1-q)S_q(A)S_q(B)
        
        # Create two independent narrowband signals at different frequencies
        signal1 = np.sin(2 * np.pi * 50 * self.t)
        signal2 = np.sin(2 * np.pi * 200 * self.t)
        
        signal1 = signal1.reshape(1, len(signal1), 1)
        signal2 = signal2.reshape(1, len(signal2), 1)
        f1, psd1 = compute_psd(signal1, CFG)
        f2, psd2 = compute_psd(signal2, CFG)
        psd1 = psd1[0, :]
        psd2 = psd2[0, :]
        
        # Create combined signal
        combined_signal = signal1 + signal2
        combined_signal = combined_signal.reshape(1, -1, 1)
        f_combined, psd_combined = compute_psd(combined_signal, CFG)
        psd_combined = psd_combined[0, :]
        
        q_values = [0.5, 1.5, 2.0]
        
        for q in q_values:
            entropy1 = tsallis_entropy_psd(psd1, f1, DEFAULT_BAND, q)
            entropy2 = tsallis_entropy_psd(psd2, f2, DEFAULT_BAND, q)
            entropy_combined = tsallis_entropy_psd(psd_combined, f_combined, DEFAULT_BAND, q)
            
            # Pseudo-additivity formula
            expected_pseudo_additive = entropy1 + entropy2 + (1 - q) * entropy1 * entropy2
            
            # This is an approximate test since signals may not be perfectly independent
            # in the frequency domain representation
            relative_error = abs(entropy_combined - expected_pseudo_additive) / \
                           abs(expected_pseudo_additive + 1e-10)
            
            # Allow larger tolerance due to approximation
            assert relative_error < 0.5, \
                f"Pseudo-additivity test failed for q={q}: " \
                f"combined={entropy_combined}, expected={expected_pseudo_additive}"
    
    @pytest.mark.parametrize("q", [0.5, 1.5, 2.0])
    def test_extremal_values(self, q):
        """Test extremal properties of Tsallis entropy"""
        # Maximum entropy should occur for uniform distribution
        # Minimum entropy should occur for maximally concentrated distribution
        
        n = 100
        
        # Uniform distribution (maximum entropy)
        uniform_psd = np.ones(n)
        f = np.linspace(0, 1, len(uniform_psd))
        entropy_uniform = tsallis_entropy_psd(uniform_psd, f, DEFAULT_BAND, q)
        
        # Concentrated distribution (minimum entropy)
        concentrated_psd = np.zeros(n)
        concentrated_psd[0] = 1.0
        entropy_concentrated = tsallis_entropy_psd(concentrated_psd, f, DEFAULT_BAND, q)
        
        # Intermediate distribution
        intermediate_psd = np.zeros(n)
        intermediate_psd[:10] = 1.0  # Spread over 10 bins
        entropy_intermediate = tsallis_entropy_psd(intermediate_psd, f, DEFAULT_BAND, q)
        
        # Uniform should have highest entropy
        assert entropy_uniform >= entropy_intermediate - self.tolerance
        assert entropy_uniform >= entropy_concentrated - self.tolerance
        
        # Concentrated should have lowest entropy
        assert entropy_concentrated <= entropy_intermediate + self.tolerance
        assert entropy_concentrated <= entropy_uniform + self.tolerance
    
    def test_continuity_in_q(self):
        """Test continuity of Tsallis entropy with respect to parameter q"""
        signals = self.generate_test_signals()
        signal = signals['white_noise'].reshape(1, len(signals['white_noise']), 1)
        f, psd = compute_psd(signal, CFG)
        psd = psd[0, :]
        
        # Test continuity around various q values
        q_centers = [0.5, 1.5, 2.0]
        
        for q_center in q_centers:
            entropy_center = tsallis_entropy_psd(psd, f, DEFAULT_BAND, q_center)
            
            # Test small perturbations in q
            delta_q_values = [0.001, 0.01, 0.1]
            
            for delta_q in delta_q_values:
                entropy_plus = tsallis_entropy_psd(psd, f, DEFAULT_BAND, q_center + delta_q)
                entropy_minus = tsallis_entropy_psd(psd, f, DEFAULT_BAND, q_center - delta_q)
                
                # Check continuity
                diff_plus = abs(entropy_plus - entropy_center)
                diff_minus = abs(entropy_minus - entropy_center)
                
                # For small delta_q, differences should be small
                if delta_q <= 0.01:
                    max_expected_change = 1.0  # Reasonable bound
                    assert diff_plus < max_expected_change, \
                        f"Large discontinuity in q: Δq={delta_q}, Δentropy={diff_plus}"
                    assert diff_minus < max_expected_change, \
                        f"Large discontinuity in q: Δq={delta_q}, Δentropy={diff_minus}"
    
    def     uity_in_distribution(self):
        """Test continuity with respect to the probability distribution"""
        signals = self.generate_test_signals()
        base_signal = signals['narrowband']
        base_signal = base_signal.reshape(1, len(base_signal), 1)
        f, psd_base = compute_psd(base_signal, CFG)
        psd_base = psd_base[0, :]
        
        q = 1.5  # Fixed q for this test
        entropy_base = tsallis_entropy_psd(psd_base, f, DEFAULT_BAND, q)
        
        # Add small perturbations
        perturbation_levels = [0.001, 0.01, 0.1]
        
        for eps in perturbation_levels:
            np.random.seed(42)
            noise = np.random.normal(0, eps, len(base_signal))
            perturbed_signal = base_signal + noise
            
            perturbed_signal = perturbed_signal.reshape(1, -1, 1)
            f, psd_perturbed = compute_psd(perturbed_signal, CFG)
            psd_perturbed = psd_perturbed[0, :]
            entropy_perturbed = tsallis_entropy_psd(psd_perturbed, f, DEFAULT_BAND, q)
            
            relative_change = abs(entropy_perturbed - entropy_base) / abs(entropy_base + 1e-10)
            
            # Small perturbations should cause bounded changes
            if eps <= 0.01:
                assert relative_change < 1.0, \
                    f"Large entropy change {relative_change} for small perturbation {eps}"
    
    def test_scale_properties(self):
        """Test scaling properties of Tsallis entropy"""
        signals = self.generate_test_signals()
        base_signal = signals['white_noise']
        
        scale_factors = [0.1, 0.5, 2.0, 10.0]
        q = 1.5
        entropies = []
        
        for scale in scale_factors:
            scaled_signal = scale * base_signal
            scaled_signal = scaled_signal.reshape(1, len(scaled_signal), 1)
            f, psd = compute_psd(scaled_signal, CFG)
            psd = psd[0, :]
            entropy = tsallis_entropy_psd(psd, f, DEFAULT_BAND, q)
            entropies.append(entropy)
        
        # Tsallis entropy should be relatively stable under scaling
        # (depends on normalization method in implementation)
        reference_entropy = entropies[0]
        for entropy in entropies[1:]:
            relative_diff = abs(entropy - reference_entropy) / abs(reference_entropy + 1e-10)
            # Allow some variation due to implementation details
            assert relative_diff < 0.5, \
                f"Large scale sensitivity: {entropies}"
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        
        # Test with q = 0 (if supported)
        try:
            psd = np.array([0.5, 0.3, 0.2])
            f = np.linspace(0, 1, len(psd))
            entropy_q0 = tsallis_entropy_psd(psd, f, DEFAULT_BAND, 0)
            # For q=0, Tsallis entropy should equal (N-1) where N is number of non-zero elements
            expected_q0 = len(psd) - 1
            # well this is not true as we are normalizing it. it goes between 0 and 1
            expected_q0 = 1
            assert abs(entropy_q0 - expected_q0) < 0.1, \
                f"q=0 case failed: got {entropy_q0}, expected {expected_q0}"
        except (ValueError, ZeroDivisionError):
            pass  # Acceptable if q=0 is not supported
        
        # Test with very large q
        psd = np.array([0.5, 0.3, 0.2])
        f = np.linspace(0, 1, len(psd))
        entropy_large_q = tsallis_entropy_psd(psd, f, DEFAULT_BAND, 10.0)
        assert not np.isnan(entropy_large_q), "Should handle large q values"
        assert entropy_large_q >= -self.tolerance, "Should remain non-negative"
        
        # # Test with single element (delta-like distribution)
        # delta_psd = np.array([1.0])
        # f = np.linspace(0, 1, len(delta_psd))
        # for q in [0.5, 1.5, 2.0]:
        #     entropy_delta = tsallis_entropy_psd(delta_psd, f, DEFAULT_BAND, q)
        #     assert abs(entropy_delta) < self.tolerance, \
        #         f"Delta distribution should have zero entropy, got {entropy_delta}"
    
    def test_q_parameter_validation(self):
        """Test handling of invalid q parameter values"""
        psd = np.array([0.4, 0.3, 0.3])
        f = np.linspace(0, 1, len(psd))
        # Test q = 1 (should either handle specially or raise appropriate error)
        try:
            entropy_q1 = tsallis_entropy_psd(psd, f, DEFAULT_BAND, 1.0)
            # If q=1 is handled, should give Shannon entropy, or None is also valid
            shannon_ref = self.shannon_entropy_reference(psd, f, DEFAULT_BAND)
            if entropy_q1 is not None:
                assert abs(entropy_q1 - shannon_ref) < 0.01, \
                    "q=1 should give Shannon entropy"
        except (ValueError, ZeroDivisionError):
            pass  # Acceptable if q=1 requires special handling
        
        # Test negative q (behavior depends on implementation choice)
        try:
            entropy_neg_q = tsallis_entropy_psd(psd, f, DEFAULT_BAND, -0.5)
            # Should either work or raise appropriate error
            assert not np.isnan(entropy_neg_q), "Should handle negative q gracefully"
        except ValueError:
            pass  # Acceptable to reject negative q

# Example usage and additional utility functions
def run_comprehensive_test():
    """Run all tests and provide summary"""
    pytest.main([__file__, "-v"])

if __name__ == "__main__":
    # You would need to import or define your tsallis_entropy_psd function here
    # Example placeholder:
    
    # Run tests
    run_comprehensive_test()