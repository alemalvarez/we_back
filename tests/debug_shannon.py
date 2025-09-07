import numpy as np

def analyze_specific_failures():
    """
    Analyze the three specific test failures you reported
    """
    
    print("=== ANALYSIS OF YOUR THREE SPECIFIC FAILURES ===\n")
    
    # Reference implementations for comparison
    def shannon_entropy_ref(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    
    def tsallis_entropy_ref(p, q):
        p = p[p > 0]
        if abs(q - 1.0) < 1e-12:
            return shannon_entropy_ref(p)
        else:
            return (1 - np.sum(p**q)) / (q - 1)
    
    print("1. FAILURE: Convergence to Shannon")
    print("-" * 40)
    print("Your result: q=0.9999999, Tsallis=0.78969009170901, Shannon=0.8675632284814613")
    print("Error: 0.07787 (9% difference)")
    print()
    
    # Let's see what the reference implementation gives
    # We need to figure out what distribution was being tested
    # The test likely used a PSD from a real signal
    
    print("LIKELY CAUSES:")
    print("a) Your q≈1 handling:")
    print("   - Not using special case for q very close to 1")
    print("   - Numerical instability in (1-Σp^q)/(q-1) when q≈1")
    print()
    
    print("b) Different normalization:")
    print("   - Your PSD normalization differs from Shannon calculation")
    print("   - Using different log base (natural log vs log2)")
    print()
    
    print("DEBUGGING APPROACH:")
    print("   # Check if you handle q≈1 specially:")
    print("   if abs(q - 1.0) < 1e-10:")
    print("       return shannon_entropy_psd(psd)")
    print("   # Check log base consistency")
    print("   # Shannon: -Σp*ln(p) or -Σp*log2(p)")
    print()
    
    print("2. FAILURE: Concavity violation")
    print("-" * 40)
    print("Your result: q=1.2, got 0.0 < expected 0.7568821352568419")
    print("This suggests your function returned 0.0 - likely an error case")
    print()
    
    print("LIKELY CAUSES:")
    print("a) Division by zero or invalid calculation")
    print("b) All PSD values became zero after processing")
    print("c) Error in the mixing of two PSDs")
    print()
    
    # Let's simulate what might cause this
    print("DEBUGGING - Testing what gives 0.0:")
    
    # Test cases that might give 0.0
    test_cases = [
        ("All zeros", np.array([0.0, 0.0, 0.0])),
        ("Single 1.0", np.array([1.0, 0.0, 0.0])),
        ("Very small values", np.array([1e-15, 1e-15, 1e-15])),
        ("Normal case", np.array([0.5, 0.3, 0.2]))
    ]
    
    for name, psd in test_cases:
        try:
            if np.sum(psd) > 0:
                psd_norm = psd / np.sum(psd)
                result = tsallis_entropy_ref(psd_norm, 1.2)
                print(f"   {name:15}: {result:.6f}")
            else:
                print(f"   {name:15}: Zero sum - would cause error")
        except Exception as e:
            print(f"   {name:15}: ERROR - {e}")
    
    print()
    print("If you're getting 0.0, check:")
    print("   - PSD preprocessing (zeros, normalization)")
    print("   - Error handling in your function")
    print("   - Mixed PSD calculation in the test")
    print()
    
    print("3. FAILURE: q=0 edge case")
    print("-" * 40)
    print("Your result: got 1.0, expected 2")
    print("This means your function returned 1.0 instead of 2 for q=0")
    print()
    
    print("THEORY: For q=0, Tsallis entropy = (number of non-zero elements - 1)")
    print()
    
    # Let's check what would give expected=2
    print("If expected=2, then distribution had 3 non-zero elements")
    print("Examples of distributions with expected S_0 = 2:")
    
    test_dists = [
        np.array([1/3, 1/3, 1/3]),  # 3 equal elements
        np.array([0.5, 0.3, 0.2]),  # 3 unequal elements
        np.array([0.8, 0.1, 0.1]),  # 3 elements, concentrated
        np.array([0.6, 0.4, 0.0])   # 2 non-zero elements (S_0 = 1)
    ]
    
    for i, dist in enumerate(test_dists):
        n_nonzero = len(dist[dist > 0])
        expected_s0 = n_nonzero - 1
        actual_s0 = tsallis_entropy_ref(dist, 0.0)
        print(f"   Dist {i+1}: {dist} -> non-zero: {n_nonzero}, S_0: {actual_s0} (expected: {expected_s0})")
    
    print()
    print("Your function likely:")
    print("   a) Doesn't handle q=0 specially")
    print("   b) Uses wrong formula for q=0")
    print("   c) Has normalization issues")
    print()
    
    print("CORRECT q=0 implementation:")
    print("   if q == 0:")
    print("       return len(psd_normalized[psd_normalized > 0]) - 1")
    print()
    
    print("="*60)
    print("SYSTEMATIC DEBUGGING PLAN")
    print("="*60)
    
    print("""
STEP 1: Test your function with simple known inputs
-------
# Simple 3-element uniform distribution
psd_test = np.array([1/3, 1/3, 1/3])

# Expected results:
# Shannon: -3*(1/3)*ln(1/3) = ln(3) ≈ 1.0986
# q=0: 3-1 = 2
# q=0.5: (1-3*(1/3)^0.5)/(0.5-1) = (1-3/sqrt(3))/(-0.5)
# q=2: (1-3*(1/3)^2)/(2-1) = (1-3/9)/1 = 2/3

your_shannon = your_shannon_entropy_psd(psd_test)  
your_q0 = your_tsallis_entropy_psd(psd_test, 0.0)
your_q05 = your_tsallis_entropy_psd(psd_test, 0.5)
your_q2 = your_tsallis_entropy_psd(psd_test, 2.0)

print(f"Shannon: got {your_shannon:.6f}, expected ~1.0986")
print(f"q=0: got {your_q0:.6f}, expected 2")
print(f"q=0.5: got {your_q05:.6f}, expected ~{expected_q05:.6f}")
print(f"q=2: got {your_q2:.6f}, expected ~0.6667")

STEP 2: Check PSD normalization
-------
# Your PSD should sum to 1 after normalization
psd_raw = np.array([10, 20, 30])  # Raw PSD values
psd_norm = psd_raw / np.sum(psd_raw)  # Should be [1/6, 1/3, 1/2]
print(f"Normalized PSD sum: {np.sum(psd_norm)}")  # Should be 1.0

STEP 3: Verify special cases handling
-------
# Check if you handle q=1 and q=0 specially
def your_tsallis_fixed(psd, q):
    psd_norm = psd / np.sum(psd)
    psd_norm = psd_norm[psd_norm > 0]  # Remove zeros
    
    if abs(q) < 1e-12:  # q = 0 case
        return len(psd_norm) - 1
    elif abs(q - 1.0) < 1e-12:  # q = 1 case
        return -np.sum(psd_norm * np.log(psd_norm))
    else:  # General case
        return (1 - np.sum(psd_norm**q)) / (q - 1)

STEP 4: Test convergence around q=1
-------
test_dist = np.array([0.4, 0.3, 0.2, 0.1])
shannon_target = -np.sum(test_dist * np.log(test_dist))

for q in [0.99, 0.999, 1.001, 1.01]:
    tsallis_val = your_tsallis_fixed(test_dist, q)
    error = abs(tsallis_val - shannon_target)
    print(f"q={q}: Tsallis={tsallis_val:.6f}, error={error:.6f}")
    """)

if __name__ == "__main__":
    analyze_specific_failures()