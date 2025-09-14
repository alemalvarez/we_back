import numpy as np
from typing import List, Optional

def calcular_re(psd: np.ndarray, f: np.ndarray, banda: List[float], q_param: float) -> Optional[float]:
    """
    Calcula la Entropía de Rényi normalizada de la PSD dentro de una banda de frecuencia específica.

    Args:
        psd: Array de NumPy 1D que representa la Densidad Espectral de Potencia.
             Se asume que los valores de PSD son no negativos.
        f: Array de NumPy 1D que representa el vector de frecuencias correspondiente a psd.
        banda: Lista con dos elementos numéricos [f_min, f_max] especificando
               la banda de frecuencia de interés.
        q_param: El parámetro q (orden) para la entropía de Rényi. 
                 Debe ser un float, q >= 0 y q != 1.

    Returns:
        La Entropía de Rényi normalizada calculada como un float.
        Devuelve None si:
        - No se encuentran frecuencias en la banda especificada.
        - La banda (después de quitar NaNs) está vacía.
        - El parámetro q es demasiado cercano a 1 o negativo.
        - La potencia total en la banda es cercana a cero.
        - El número de puntos válidos N_nz en la PDF es 0.
        Devuelve 0.0 si:
        - Hay exactamente 1 punto válido en la PDF (N_nz = 1), ya que la entropía normalizada es 0.
        
    Raises:
        TypeError: Si las entradas no son de los tipos esperados.
        ValueError: Si las entradas no cumplen con los requisitos dimensionales/longitud,
                    o si 'banda' no está correctamente formateada o f_min > f_max.
    """
    EPSILON_Q_ONE = 1e-6  # More reasonable tolerance for q ≈ 1
    EPSILON_POWER = 1e-12  # Stricter tolerance for zero power

    # --- Input Validation ---
    if not isinstance(psd, np.ndarray):
        raise TypeError("El argumento 'psd' debe ser un array de NumPy.")
    if not isinstance(f, np.ndarray):
        raise TypeError("El argumento 'f' debe ser un array de NumPy.")
    if not isinstance(q_param, (int, float)):
        raise TypeError("El argumento 'q_param' debe ser un número.")

    if psd.ndim != 1:
        raise ValueError("El array 'psd' debe ser 1D.")
    if f.ndim != 1:
        raise ValueError("El array 'f' debe ser 1D.")
    if len(psd) != len(f):
        raise ValueError("Los arrays 'psd' y 'f' deben tener la misma longitud.")

    # More flexible banda validation
    if not isinstance(banda, (list, tuple)) or len(banda) != 2:
        raise TypeError("El argumento 'banda' debe ser una lista de dos elementos [f_min, f_max].")

    try:
        f_min = float(banda[0])
        f_max = float(banda[1])
    except (TypeError, ValueError) as e:
        raise TypeError(f"Los elementos de 'banda' deben ser números. Error: {e}")

    # Allow f_min == f_max for single frequency point
    if f_min > f_max:
        raise ValueError(f"f_min ({f_min}) no puede ser mayor que f_max ({f_max}) en 'banda'.")

    # Check q parameter constraints
    if abs(q_param - 1.0) < EPSILON_Q_ONE:
        return None
    if q_param < 0:
        return None

    # --- Selección de la banda de frecuencia ---
    indbanda = np.where((f >= f_min) & (f <= f_max))[0]
    if indbanda.size == 0:
        return None

    psd_banda_raw = psd[indbanda]
    # Remove NaN and negative values
    valid_mask = ~np.isnan(psd_banda_raw) & (psd_banda_raw >= 0)
    psd_banda_valid = psd_banda_raw[valid_mask]

    if psd_banda_valid.size == 0:
        return None 

    # --- Cálculo de la Potencia Total y PDF ---
    potencia_total = np.sum(psd_banda_valid)
    if potencia_total <= EPSILON_POWER:
        return None

    pdf = psd_banda_valid / potencia_total
    
    # Keep all positive PDF values (not just those above epsilon)
    pdf_positive = pdf[pdf > 0]
    N_nz = pdf_positive.size

    if N_nz == 0: 
        return None
    
    if N_nz == 1:
        # Single point has zero entropy when normalized
        return 0.0

    # --- Cálculo de la Entropía de Rényi Normalizada ---
    try:
        # Handle special cases for q
        if q_param == 0:
            # H_0 = log(N) - just count of non-zero elements
            renyi_entropy = np.log(N_nz)
        elif np.isinf(q_param):
            # H_∞ = -log(max(p_i))
            renyi_entropy = -np.log(np.max(pdf_positive))
        else:
            # General case: H_q = (1/(1-q)) * log(sum(p_i^q))
            sum_pi_q = np.sum(pdf_positive**q_param)
            
            if sum_pi_q <= 0:
                return None
                
            renyi_entropy = (1.0 / (1.0 - q_param)) * np.log(sum_pi_q)
        
        # Normalize by maximum possible entropy (uniform distribution)
        max_entropy = np.log(N_nz)  # Maximum entropy for N_nz equally likely states
        
        if max_entropy <= 0:
            return 0.0
            
        # Ensure proper normalization
        if q_param > 1:
            # For q > 1, entropy is negative, so we need to handle normalization carefully
            renyi_entropy_normalized = renyi_entropy / max_entropy
            # Map to [0,1] range where 0 = minimum entropy, 1 = maximum entropy
            renyi_entropy_normalized = 1.0 + renyi_entropy_normalized
            renyi_entropy_normalized = np.clip(renyi_entropy_normalized, 0.0, 1.0)
        else:
            # For 0 < q < 1, entropy is positive
            renyi_entropy_normalized = renyi_entropy / max_entropy
            renyi_entropy_normalized = np.clip(renyi_entropy_normalized, 0.0, 1.0)
        
        return float(renyi_entropy_normalized)
        
    except (OverflowError, ZeroDivisionError, ValueError):
        # Handle numerical issues gracefully
        return None


def calcular_re_original(psd: np.ndarray, f: np.ndarray, banda: List[float], q_param: float) -> Optional[float]:
    """
    Original implementation for comparison - kept for reference.
    """
    # ... (your original implementation here)
    pass

def calcular_re_vector(psd: np.ndarray, f: np.ndarray, banda: List[float], q_param: float) -> np.ndarray:
    """
    Vectorized version of calcular_re that processes multiple segments at once.

    Args:
        psd: Array of shape (n_segments, n_freqs) containing PSD values for multiple segments
        f: Array of shape (n_freqs,) containing frequency values
        banda: List with two numeric elements [f_min, f_max] specifying the frequency band
        q_param: The q parameter (order) for Rényi entropy. Must be a float, q >= 0 and q != 1.

    Returns:
        Array of shape (n_segments,) containing normalized Rényi entropy values.
        NaN values indicate invalid calculations.
    """
    EPSILON_Q_ONE = 1e-6
    EPSILON_POWER = 1e-12

    # --- Input Validation ---
    if not isinstance(psd, np.ndarray):
        raise TypeError("El argumento 'psd' debe ser un array de NumPy.")
    if not isinstance(f, np.ndarray):
        raise TypeError("El argumento 'f' debe ser un array de NumPy.")
    if not isinstance(q_param, (int, float)):
        raise TypeError("El argumento 'q_param' debe ser un número.")
    if psd.ndim != 2:
        raise ValueError("El array 'psd' debe ser 2D.")
    if f.ndim != 1:
        raise ValueError("El array 'f' debe ser 1D.")
    if psd.shape[1] != len(f):
        raise ValueError("La dimensión de frecuencias de psd debe coincidir con la longitud de f.")

    if not isinstance(banda, (list, tuple)) or len(banda) != 2:
        raise ValueError("El argumento 'banda' debe ser una lista de dos elementos [f_min, f_max].")

    try:
        f_min = float(banda[0])
        f_max = float(banda[1])
    except (TypeError, ValueError) as e:
        raise TypeError(f"Los elementos de 'banda' deben ser números. Error: {e}")

    if f_min > f_max:
        raise ValueError(f"f_min ({f_min}) no puede ser mayor que f_max ({f_max}) en 'banda'.")

    if abs(q_param - 1.0) < EPSILON_Q_ONE or q_param < 0:
        return np.full(psd.shape[0], np.nan)

    # Initialize results array with NaN
    n_segments = psd.shape[0]
    results = np.full(n_segments, np.nan)

    # --- Selección de la banda de frecuencia ---
    mask = (f >= f_min) & (f <= f_max)
    if not np.any(mask):
        return results

    # Extract data within band
    psd_banda = psd[:, mask]
    
    # Process each segment
    for i in range(n_segments):
        # Remove NaNs and negative values
        valid_mask = ~np.isnan(psd_banda[i]) & (psd_banda[i] >= 0)
        psd_valid = psd_banda[i, valid_mask]
        
        if psd_valid.size == 0:
            continue
            
        # Calculate total power
        potencia_total = np.sum(psd_valid)
        if potencia_total <= EPSILON_POWER:
            continue
            
        # Calculate PDF
        pdf = psd_valid / potencia_total
        pdf_positive = pdf[pdf > 0]
        N_nz = pdf_positive.size
        
        if N_nz == 0:
            continue
            
        if N_nz == 1:
            results[i] = 0.0
            continue
            
        try:
            # Handle special cases for q
            if q_param == 0:
                # H_0 = log(N)
                renyi_entropy = np.log(N_nz)
            elif np.isinf(q_param):
                # H_∞ = -log(max(p_i))
                renyi_entropy = -np.log(np.max(pdf_positive))
            else:
                # General case: H_q = (1/(1-q)) * log(sum(p_i^q))
                sum_pi_q = np.sum(pdf_positive**q_param)
                if sum_pi_q <= 0:
                    continue
                renyi_entropy = (1.0 / (1.0 - q_param)) * np.log(sum_pi_q)
            
            # Normalize
            max_entropy = np.log(N_nz)
            if max_entropy <= 0:
                results[i] = 0.0
                continue
                
            if q_param > 1:
                # For q > 1, entropy is negative
                renyi_entropy_normalized = 1.0 + (renyi_entropy / max_entropy)
                renyi_entropy_normalized = np.clip(renyi_entropy_normalized, 0.0, 1.0)
            else:
                # For 0 < q < 1, entropy is positive
                renyi_entropy_normalized = renyi_entropy / max_entropy
                renyi_entropy_normalized = np.clip(renyi_entropy_normalized, 0.0, 1.0)
                
            results[i] = float(renyi_entropy_normalized)
            
        except (OverflowError, ZeroDivisionError, ValueError):
            continue
            
    return results

if __name__ == "__main__":
    import time
    
    # Create dummy test data
    n_segments = 1000
    n_freqs = 1000
    f = np.linspace(0, 100, n_freqs)
    psd = np.random.rand(n_segments, n_freqs)
    banda = [20.0, 80.0]
    q_param = 2.0  # Example q parameter
    
    # Test non-vectorized version
    start_time = time.time()
    re_results = np.array([calcular_re(psd[i], f, banda, q_param) for i in range(n_segments)], dtype=float)
    non_vector_time = time.time() - start_time
    
    # Test vectorized version
    start_time = time.time()
    re_results_vector = calcular_re_vector(psd, f, banda, q_param)
    vector_time = time.time() - start_time
    
    # Compare results
    is_close = np.allclose(re_results, re_results_vector, equal_nan=True, rtol=1e-10, atol=1e-10)
    print(f"Results match: {is_close}")
    if not is_close:
        # Print some statistics about the differences
        mask = ~np.isnan(re_results) & ~np.isnan(re_results_vector)
        if np.any(mask):
            diffs = np.abs(re_results[mask] - re_results_vector[mask])
            print(f"Max difference: {np.max(diffs)}")
            print(f"Mean difference: {np.mean(diffs)}")
            print(f"Number of different values: {np.sum(diffs > 0)}")
    
    print(f"Non-vectorized time: {non_vector_time:.3f} seconds")
    print(f"Vectorized time: {vector_time:.3f} seconds")
    print(f"Speedup: {non_vector_time/vector_time:.1f}x")