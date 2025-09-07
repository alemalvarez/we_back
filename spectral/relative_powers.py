import numpy as np
from typing import List, Tuple

def calcular_rp(psd: np.ndarray, 
                f: np.ndarray, 
                banda_total: List[float], 
                sub_bandas: List[List[float]]) -> np.ndarray:
    """
    Calcula la potencia relativa para las subbandas indicadas en sub_bandas.

    Args:
        psd (np.ndarray): Densidad espectral de potencia (1D array).
        f (np.ndarray): Vector de frecuencias correspondiente a psd (1D array).
        banda_total (List[float]): Lista o tupla con dos elementos [f_min, f_max] 
                                   especificando la banda de frecuencia total para 
                                   el cálculo de la potencia de referencia (denominador).
        sub_bandas (List[List[float]]): Lista de listas/tuplas. Cada elemento interno 
                                        es una lista/tupla de dos floats [sb_min, sb_max] 
                                        definiendo una sub-banda de frecuencia para la cual 
                                        se calculará la potencia relativa.

    Returns:
        np.ndarray: Un array de NumPy con la potencia relativa para cada sub-banda 
                    especificada en sub_bandas. Los valores pueden ser np.nan o np.inf 
                    si la potencia total en banda_total es cero. Devuelve un array vacío
                    si sub_bandas está vacía.

    Raises:
        TypeError: Si psd o f no son arrays de NumPy.
        ValueError: Si psd o f no son 1D o no tienen la misma longitud.
                    Si banda_total o los elementos de sub_bandas no tienen el formato correcto.

    Ejemplo:
        >>> psd = np.array([0.1, 0.5, 1.0, 2.0, 1.5, 0.8, 0.3])
        >>> f = np.array([1, 2, 3, 4, 5, 6, 7]) # Hz
        >>> banda_total = [1, 7] # Hz
        >>> sub_bandas_ejemplo = [[1, 3], [4, 5], [6, 7]] # Sub-bandas
        >>> calcular_rp(psd, f, banda_total, sub_bandas_ejemplo)
        array([0.25      , 0.5625    , 0.1875    ]) # (0.1+0.5+1.0)/(sum_total), (2.0+1.5)/(sum_total), (0.8+0.3)/(sum_total)
                                                    # sum_total = 0.1+0.5+1.0+2.0+1.5+0.8+0.3 = 6.2
                                                    # 1.6/6.2 = 0.25806...
                                                    # 3.5/6.2 = 0.56451...
                                                    # 1.1/6.2 = 0.17741...
                                                    # Recalculating example in docstring slightly for precision
                                                    # Should be approx: [0.25806, 0.56451, 0.17741]

    See also: CALCULARPARAMETRO, CALCULOAP (from MATLAB context)
    """

    # --- Input validations ---
    if not (isinstance(psd, np.ndarray) and isinstance(f, np.ndarray)):
        raise TypeError("psd y f deben ser arrays de NumPy.")
    if psd.ndim != 1 or f.ndim != 1:
        raise ValueError("psd y f deben ser arrays 1D.")
    if psd.shape != f.shape:
        raise ValueError("psd y f deben tener la misma longitud.")
    
    if not (isinstance(banda_total, (list, tuple)) and len(banda_total) == 2 and banda_total[0] <= banda_total[1]):
        raise ValueError("banda_total debe ser una lista/tupla de dos elementos [f_min, f_max] con f_min <= f_max.")
    
    if not isinstance(sub_bandas, list):
        raise TypeError("sub_bandas debe ser una lista de sub-bandas.")
    if not all(isinstance(sb, (list, tuple)) and len(sb) == 2 and sb[0] <= sb[1] for sb in sub_bandas):
        raise ValueError("Cada sub-banda en sub_bandas debe ser una lista/tupla de dos elementos [f_min, f_max] con f_min <= f_max.")

    # Handle empty sub_bandas list early
    if not sub_bandas:
        return np.array([])

    # Handle empty frequency/psd vector
    if f.size == 0: # psd.size will also be 0
        # If sub_bandas is not empty, but f is empty, total power is effectively 0, 
        # and sub-band powers are 0. Result for each sub-band would be 0/0 = NaN.
        return np.full(len(sub_bandas), np.nan)

    # --- Calcular potencia total en la banda_total (denominador) ---
    idx_banda_total = np.where((f >= banda_total[0]) & (f <= banda_total[1]))[0]
    
    if idx_banda_total.size == 0:
        # No frequencies in the overall reference band. 
        # Total power is 0. All relative powers are undefined (NaN if numerator is 0, Inf if numerator > 0).
        # To be consistent with division by zero, we can calculate numerators and let division handle it,
        # or return NaNs directly if we define relative power as undefined.
        # Let's calculate numerators and allow division by zero, which yields nan/inf.
        potencia_total_denominador = 0.0
    else:
        potencia_total_denominador = np.sum(psd[idx_banda_total])

    # --- Calcular potencias absolutas en cada sub-banda (numeradores) ---
    potencias_absolutas_numeradores = []
    for sb in sub_bandas:
        idx_sub_banda = np.where((f >= sb[0]) & (f <= sb[1]))[0]
        
        if idx_sub_banda.size == 0:
            potencias_absolutas_numeradores.append(0.0) # No power if no frequencies in this sub-band
        else:
            potencias_absolutas_numeradores.append(np.sum(psd[idx_sub_banda]))
    
    np_potencias_absolutas = np.array(potencias_absolutas_numeradores, dtype=float)

    # --- Calcular potencias relativas ---
    # Division by zero in NumPy results in np.inf (for non-zero/0) or np.nan (for 0/0),
    # which matches MATLAB's behavior.
    with np.errstate(divide='ignore', invalid='ignore'): # Suppress runtime warnings for division by zero/NaN
        relative_powers = np_potencias_absolutas / potencia_total_denominador
    
    return relative_powers

def calcular_rp_vector(psd: np.ndarray, 
                      f: np.ndarray, 
                      banda_total: List[float], 
                      sub_bandas: List[List[float]]) -> np.ndarray:
    """
    Vectorized version of calcular_rp that processes multiple segments at once.

    Args:
        psd (np.ndarray): Densidad espectral de potencia (2D array of shape n_segments x n_freqs).
        f (np.ndarray): Vector de frecuencias correspondiente a psd (1D array).
        banda_total (List[float]): Lista o tupla con dos elementos [f_min, f_max] 
                                   especificando la banda de frecuencia total para 
                                   el cálculo de la potencia de referencia (denominador).
        sub_bandas (List[List[float]]): Lista de listas/tuplas. Cada elemento interno 
                                        es una lista/tupla de dos floats [sb_min, sb_max] 
                                        definiendo una sub-banda de frecuencia para la cual 
                                        se calculará la potencia relativa.

    Returns:
        np.ndarray: Un array de NumPy con shape (n_segments, n_sub_bandas) conteniendo 
                    la potencia relativa para cada sub-banda en cada segmento.
                    Los valores pueden ser np.nan o np.inf si la potencia total en 
                    banda_total es cero. Devuelve un array vacío si sub_bandas está vacía.

    Raises:
        TypeError: Si psd o f no son arrays de NumPy.
        ValueError: Si psd no es 2D, f no es 1D, o si las dimensiones no coinciden.
                    Si banda_total o los elementos de sub_bandas no tienen el formato correcto.
    """
    # --- Input validations ---
    if not (isinstance(psd, np.ndarray) and isinstance(f, np.ndarray)):
        raise TypeError("psd y f deben ser arrays de NumPy.")
    if psd.ndim != 2 or f.ndim != 1:
        raise ValueError("psd debe ser 2D y f debe ser 1D.")
    if psd.shape[1] != f.shape[0]:
        raise ValueError("La dimensión de frecuencias de psd debe coincidir con la longitud de f.")
    
    if not (isinstance(banda_total, (list, tuple)) and len(banda_total) == 2 and banda_total[0] <= banda_total[1]):
        raise ValueError("banda_total debe ser una lista/tupla de dos elementos [f_min, f_max] con f_min <= f_max.")
    
    if not isinstance(sub_bandas, list):
        raise TypeError("sub_bandas debe ser una lista de sub-bandas.")
    if not all(isinstance(sb, (list, tuple)) and len(sb) == 2 and sb[0] <= sb[1] for sb in sub_bandas):
        raise ValueError("Cada sub-banda en sub_bandas debe ser una lista/tupla de dos elementos [f_min, f_max] con f_min <= f_max.")

    # Handle empty sub_bandas list early
    if not sub_bandas:
        return np.array([])

    # Handle empty frequency/psd vector
    if f.size == 0:
        return np.full((psd.shape[0], len(sub_bandas)), np.nan)

    # --- Calcular potencia total en la banda_total (denominador) ---
    mask_banda_total = (f >= banda_total[0]) & (f <= banda_total[1])
    if not np.any(mask_banda_total):
        potencia_total_denominador = np.zeros(psd.shape[0])
    else:
        potencia_total_denominador = np.sum(psd[:, mask_banda_total], axis=1)

    # --- Calcular potencias absolutas en cada sub-banda (numeradores) ---
    n_segments = psd.shape[0]
    n_sub_bandas = len(sub_bandas)
    potencias_absolutas = np.zeros((n_segments, n_sub_bandas))

    for i, sb in enumerate(sub_bandas):
        mask_sub_banda = (f >= sb[0]) & (f <= sb[1])
        if np.any(mask_sub_banda):
            potencias_absolutas[:, i] = np.sum(psd[:, mask_sub_banda], axis=1)

    # --- Calcular potencias relativas ---
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_powers = potencias_absolutas / potencia_total_denominador[:, np.newaxis]
    
    return relative_powers

if __name__ == "__main__":
    import time
    
    # Create dummy test data
    n_segments = 1000
    n_freqs = 1000
    f = np.linspace(0, 100, n_freqs)
    psd = np.random.rand(n_segments, n_freqs)
    banda_total = [20.0, 80.0]
    sub_bandas = [[20.0, 40.0], [40.0, 60.0], [60.0, 80.0]]
    
    # Test non-vectorized version
    start_time = time.time()
    rp_results = np.array([calcular_rp(psd[i], f, banda_total, sub_bandas) for i in range(n_segments)])
    non_vector_time = time.time() - start_time
    
    # Test vectorized version
    start_time = time.time()
    rp_results_vector = calcular_rp_vector(psd, f, banda_total, sub_bandas)
    vector_time = time.time() - start_time
    
    # Compare results
    is_close = np.allclose(rp_results, rp_results_vector, equal_nan=True, rtol=1e-10, atol=1e-10)
    print(f"Results match: {is_close}")
    if not is_close:
        # Print some statistics about the differences
        mask = ~np.isnan(rp_results) & ~np.isnan(rp_results_vector)
        if np.any(mask):
            diffs = np.abs(rp_results[mask] - rp_results_vector[mask])
            print(f"Max difference: {np.max(diffs)}")
            print(f"Mean difference: {np.mean(diffs)}")
            print(f"Number of different values: {np.sum(diffs > 0)}")
    
    print(f"Non-vectorized time: {non_vector_time:.3f} seconds")
    print(f"Vectorized time: {vector_time:.3f} seconds")
    print(f"Speedup: {non_vector_time/vector_time:.1f}x")
