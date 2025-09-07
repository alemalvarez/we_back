import numpy as np
from typing import List, Optional
from loguru import logger

def calcular_te(psd: np.ndarray, f: np.ndarray, banda: List[float], q_param: float) -> Optional[float]:
    """
    Calcula la Entropía de Tsallis normalizada de la PSD dentro de una banda de frecuencia específica.

    Args:
        psd: Array de NumPy 1D que representa la Densidad Espectral de Potencia.
             Se asume que los valores de PSD son no negativos.
        f: Array de NumPy 1D que representa el vector de frecuencias correspondiente a psd.
        banda: Lista con dos elementos numéricos [f_min, f_max] especificando
               la banda de frecuencia de interés.
        q_param: El parámetro q (índice entrópico) para la entropía de Tsallis. 
                 Debe ser un float, q != 1.

    Returns:
        La Entropía de Tsallis normalizada calculada como un float.
        Devuelve None si:
        - No se encuentran frecuencias en la banda especificada.
        - La banda (después de quitar NaNs) está vacía.
        - El parámetro q es demasiado cercano a 1.
        Devuelve 0.0 si:
        - La potencia total en la banda (después de quitar NaNs) es cero o negativa.
        - Hay 1 o menos puntos válidos en la PDF (N_nz <= 1), ya que la entropía es 0.
        
    Raises:
        TypeError: Si las entradas no son de los tipos esperados.
        ValueError: Si las entradas no cumplen con los requisitos dimensionales/longitud,
                    o si 'banda' no está correctamente formateada o f_min > f_max.
    """
    EPSILON_Q_ONE = 1e-9
    EPSILON_POWER = 1e-9
    EPSILON_PDF_ZERO = 1e-9

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

    if not isinstance(banda, list) or len(banda) != 2:
        raise ValueError("El argumento 'banda' debe ser una lista de dos elementos [f_min, f_max].")

    try:
        f_min = float(banda[0])
        f_max = float(banda[1])
    except (TypeError, ValueError) as e:
        raise TypeError(f"Los elementos de 'banda' deben ser números. Error: {e}")

    if f_min > f_max:
        raise ValueError(f"f_min ({f_min}) no puede ser mayor que f_max ({f_max}) en 'banda'.")

    if abs(q_param - 1.0) < EPSILON_Q_ONE:
        print(f"Advertencia: q_param ({q_param}) está demasiado cerca de 1. "
              f"La entropía de Tsallis no está definida para q=1 con esta fórmula. "
              f"Considere usar la entropía de Shannon.")
        return None

    # --- Selección de la banda de frecuencia ---
    indbanda = np.where((f >= f_min) & (f <= f_max))[0]
    if indbanda.size == 0:
        return None

    psd_banda_raw = psd[indbanda]
    psd_banda_valid = psd_banda_raw[~np.isnan(psd_banda_raw)]

    logger.info(f"psd_banda_valid length: {psd_banda_valid.size}")

    if psd_banda_valid.size == 0:
        return None 

    # --- Cálculo de la Potencia Total y PDF ---
    potencia_total = np.sum(psd_banda_valid)
    if potencia_total <= EPSILON_POWER:
        return 0.0

    pdf = psd_banda_valid / potencia_total
    pdf_positive = pdf[pdf > EPSILON_PDF_ZERO]
    N_nz = pdf_positive.size
    N_total = psd_banda_valid.size

    logger.info(f"N_nz: {N_nz}")
    logger.info(f"N_total: {N_total}")

    # --- Cálculo de la Entropía de Tsallis Normalizada ---
    if N_nz <= 1:
        # Si hay 0 o 1 estados con probabilidad > 0, la entropía (incertidumbre) es 0.
        return 0.0

    # Numerador: (1 - sum(p_i^q))
    sum_pi_q = np.sum(pdf_positive**q_param)
    numerator = 1.0 - sum_pi_q

    # Denominador de normalización: (1 - N_nz^(1-q))
    # Esto asegura que para una distribución uniforme p_i = 1/N_nz, S_q_norm = 1 (si q > 0).
    # O S_q_norm = 0 para N_nz=1 (caso ya cubierto).
    denominator_norm = 1.0 - N_total**(1.0 - q_param)

    logger.info(f"denominator_norm: {denominator_norm}")

    if abs(denominator_norm) < EPSILON_POWER: # Denominador es prácticamente cero
        # Esto debería ocurrir solo si N_nz^(1-q) es 1. 
        # Si N_nz > 1, esto implica 1-q = 0 => q=1, que ya está filtrado.
        # Si el numerador también es cero (como en el caso N_nz=1), el resultado es 0 (ya cubierto).
        # Si el numerador no es cero y el denominador es cero, es una singularidad.
        # Sin embargo, con N_nz > 1 y q != 1, denominator_norm no debería ser cero.
        # Si aun así sucede por problemas numéricos y el numerador es no cero, podría ser Inf.
        # Por ahora, si N_nz > 1 y q != 1, esto es un caso anómalo. Devolver None.
        print(f"Advertencia: Denominador de normalización para Tsallis es cero con N_nz={N_nz}, q={q_param}. Esto es inesperado.")
        return None 
        
    normalized_tsallis_entropy = numerator / denominator_norm

    logger.info(f"normalized_tsallis_entropy: {normalized_tsallis_entropy}")
    
    return float(normalized_tsallis_entropy) 

def calcular_te_vector(psd: np.ndarray, f: np.ndarray, banda: List[float], q_param: float) -> np.ndarray:
    """
    Vectorized version of calcular_te that processes multiple segments at once.

    Args:
        psd: Array of shape (n_segments, n_freqs) containing PSD values for multiple segments
        f: Array of shape (n_freqs,) containing frequency values
        banda: List with two numeric elements [f_min, f_max] specifying the frequency band
        q_param: The q parameter (entropic index) for Tsallis entropy. Must be a float, q != 1.

    Returns:
        Array of shape (n_segments,) containing normalized Tsallis entropy values.
        NaN values indicate invalid calculations.
    """
    EPSILON_Q_ONE = 1e-9
    EPSILON_POWER = 1e-9
    EPSILON_PDF_ZERO = 1e-9

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

    if not isinstance(banda, list) or len(banda) != 2:
        raise ValueError("El argumento 'banda' debe ser una lista de dos elementos [f_min, f_max].")

    try:
        f_min = float(banda[0])
        f_max = float(banda[1])
    except (TypeError, ValueError) as e:
        raise TypeError(f"Los elementos de 'banda' deben ser números. Error: {e}")

    if f_min > f_max:
        raise ValueError(f"f_min ({f_min}) no puede ser mayor que f_max ({f_max}) en 'banda'.")

    if abs(q_param - 1.0) < EPSILON_Q_ONE:
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
        # Remove NaNs
        valid_mask = ~np.isnan(psd_banda[i])
        psd_valid = psd_banda[i, valid_mask]
        
        if psd_valid.size == 0:
            continue
            
        # Calculate total power
        potencia_total = np.sum(psd_valid)
        if potencia_total <= EPSILON_POWER:
            results[i] = 0.0
            continue
            
        # Calculate PDF
        pdf = psd_valid / potencia_total
        pdf_positive = pdf[pdf > EPSILON_PDF_ZERO]
        N_nz = pdf_positive.size
        
        if N_nz <= 1:
            results[i] = 0.0
            continue
            
        try:
            # Calculate Tsallis entropy
            sum_pi_q = np.sum(pdf_positive**q_param)
            numerator = 1.0 - sum_pi_q
            denominator_norm = 1.0 - N_nz**(1.0 - q_param)
            
            if abs(denominator_norm) < EPSILON_POWER:
                continue
                
            normalized_tsallis_entropy = numerator / denominator_norm
            results[i] = float(normalized_tsallis_entropy)
            
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
    te_results = np.array([calcular_te(psd[i], f, banda, q_param) for i in range(n_segments)], dtype=float)
    non_vector_time = time.time() - start_time
    
    # Test vectorized version
    start_time = time.time()
    te_results_vector = calcular_te_vector(psd, f, banda, q_param)
    vector_time = time.time() - start_time
    
    # Compare results
    is_close = np.allclose(te_results, te_results_vector, equal_nan=True, rtol=1e-10, atol=1e-10)
    print(f"Results match: {is_close}")
    if not is_close:
        # Print some statistics about the differences
        mask = ~np.isnan(te_results) & ~np.isnan(te_results_vector)
        if np.any(mask):
            diffs = np.abs(te_results[mask] - te_results_vector[mask])
            print(f"Max difference: {np.max(diffs)}")
            print(f"Mean difference: {np.mean(diffs)}")
            print(f"Number of different values: {np.sum(diffs > 0)}")
    
    print(f"Non-vectorized time: {non_vector_time:.3f} seconds")
    print(f"Vectorized time: {vector_time:.3f} seconds")
    print(f"Speedup: {non_vector_time/vector_time:.1f}x") 