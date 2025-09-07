import numpy as np
from typing import List

def calcular_sef95(psd: np.ndarray, f: np.ndarray, banda: list[float]) -> float | None:
    """
    Calcula la Frecuencia Espectral Límite al 95% (SEF95) de la distribución 
    de densidad espectral de potencia (PSD) dentro de una banda de frecuencia específica.

    Args:
        psd: Array de NumPy que representa la Densidad Espectral de Potencia.
        f: Array de NumPy que representa el vector de frecuencias correspondiente a psd.
        banda: Lista o tupla con dos elementos [f_min, f_max] especificando 
               la banda de frecuencia de interés.

    Returns:
        La frecuencia espectral límite al 95% calculada dentro de la banda especificada.
        Devuelve None si no hay datos en la banda, si la potencia total es cero o negativa,
        o si ocurre un error.
    """

    if len(psd) != len(f):
        raise ValueError("psd y f deben tener la misma longitud.")
    if not isinstance(banda, (list, tuple)) or len(banda) != 2:
        raise ValueError("banda debe ser una lista o tupla de dos elementos [f_min, f_max].")
    if banda[0] > banda[1]:
        raise ValueError("El primer elemento de banda (f_min) no puede ser mayor que el segundo (f_max).")

    # Encontrar índices dentro de la banda de frecuencia
    indbanda = np.where((f >= banda[0]) & (f <= banda[1]))[0]

    if indbanda.size == 0:
        print(f"Advertencia: No se encontraron frecuencias en la banda especificada [{banda[0]}, {banda[1]}].")
        return None

    psd_banda = psd[indbanda]
    f_banda = f[indbanda]

    # Potencia total en la banda
    potencia_total = np.sum(psd_banda)

    if potencia_total <= 0:
        print(f"Advertencia: La potencia total en la banda [{banda[0]}, {banda[1]}] es cero o negativa.")
        return None

    # Suma acumulada de la potencia en la banda
    vector_suma = np.cumsum(psd_banda)

    # Encontrar el índice donde la suma acumulada alcanza el 95% de la potencia total
    indices_95 = np.where(vector_suma <= (0.95 * potencia_total))[0]

    if indices_95.size == 0:
        # Si ningún índice cumple (es decir, vector_suma[0] > 0.95 * potencia_total),
        # la SEF95 es la primera frecuencia de la banda.
        ind_sef95_en_banda = 0
    else:
        # El índice deseado es el último que cumple la condición (equivalente a max(find(...)))
        ind_sef95_en_banda = indices_95[-1]

    # La SEF95 es la frecuencia correspondiente a ese índice
    spectral_edge_frequency_95 = f_banda[ind_sef95_en_banda]

    return float(spectral_edge_frequency_95)

def calcular_sef95_vector(psd: np.ndarray, f: np.ndarray, banda: List[float]) -> np.ndarray:
    """
    Vectorized version of calcular_sef95 that processes multiple segments at once.

    Args:
        psd: Array of shape (n_segments, n_freqs) containing PSD values for multiple segments
        f: Array of shape (n_freqs,) containing frequency values
        banda: List with two numeric elements [f_min, f_max] specifying the frequency band

    Returns:
        Array of shape (n_segments,) containing spectral edge frequency values.
        NaN values indicate invalid calculations.
    """
    # --- Input Validation ---
    if not isinstance(psd, np.ndarray):
        raise TypeError("El argumento 'psd' debe ser un array de NumPy.")
    if not isinstance(f, np.ndarray):
        raise TypeError("El argumento 'f' debe ser un array de NumPy.")
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

    # Initialize results array with NaN
    n_segments = psd.shape[0]
    results = np.full(n_segments, np.nan)

    # --- Selección de la banda de frecuencia ---
    mask = (f >= f_min) & (f <= f_max)
    if not np.any(mask):
        return results

    # Extract data within band
    psd_banda = psd[:, mask]
    f_banda = f[mask]
    
    # Process each segment
    for i in range(n_segments):
        # Calculate total power in band
        potencia_total = np.sum(psd_banda[i])
        
        if potencia_total <= 0:
            continue
            
        # Calculate cumulative sum of power
        vector_suma = np.cumsum(psd_banda[i])
        
        # Find index where cumulative sum reaches 95% of total power
        indices_95 = np.where(vector_suma <= (0.95 * potencia_total))[0]
        
        if indices_95.size == 0:
            # If no index satisfies the condition, SEF95 is the first frequency
            ind_sef95_en_banda = 0
        else:
            # The desired index is the last one that satisfies the condition
            ind_sef95_en_banda = indices_95[-1]
            
        # SEF95 is the frequency corresponding to that index
        results[i] = f_banda[ind_sef95_en_banda]
            
    return results

if __name__ == "__main__":
    import time
    
    # Create dummy test data
    n_segments = 1000
    n_freqs = 1000
    f = np.linspace(0, 100, n_freqs)
    psd = np.random.rand(n_segments, n_freqs)
    banda = [20.0, 80.0]
    
    # Test non-vectorized version
    start_time = time.time()
    sef95_results = np.array([calcular_sef95(psd[i], f, banda) for i in range(n_segments)], dtype=float)
    non_vector_time = time.time() - start_time
    
    # Test vectorized version
    start_time = time.time()
    sef95_results_vector = calcular_sef95_vector(psd, f, banda)
    vector_time = time.time() - start_time
    
    # Compare results
    is_close = np.allclose(sef95_results, sef95_results_vector, equal_nan=True, rtol=1e-10, atol=1e-10)
    print(f"Results match: {is_close}")
    if not is_close:
        # Print some statistics about the differences
        mask = ~np.isnan(sef95_results) & ~np.isnan(sef95_results_vector)
        if np.any(mask):
            diffs = np.abs(sef95_results[mask] - sef95_results_vector[mask])
            print(f"Max difference: {np.max(diffs)}")
            print(f"Mean difference: {np.mean(diffs)}")
            print(f"Number of different values: {np.sum(diffs > 0)}")
    
    print(f"Non-vectorized time: {non_vector_time:.3f} seconds")
    print(f"Vectorized time: {vector_time:.3f} seconds")
    print(f"Speedup: {non_vector_time/vector_time:.1f}x") 