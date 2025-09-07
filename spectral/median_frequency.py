import numpy as np
from typing import Union, Sequence

def calcular_mf(psd: np.ndarray, f: np.ndarray, banda: Sequence[Union[int, float]]) -> float | None:
    """
    Calcula la frecuencia mediana (MF) de la distribución de densidad espectral 
    de potencia (PSD) dentro de una banda de frecuencia específica.

    Args:
        psd: Array de NumPy que representa la Densidad Espectral de Potencia.
        f: Array de NumPy que representa el vector de frecuencias correspondiente a psd.
        banda: Lista o tupla con dos elementos [f_min, f_max] especificando 
               la banda de frecuencia de interés.

    Returns:
        La frecuencia mediana calculada dentro de la banda especificada. 
        Devuelve None si no hay datos en la banda o si ocurre un error.
        
    See also: calcular_parametro, calcular_sef, calcular_iaftf 
    (Assuming these are related functions in the original MATLAB context)
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
         return None # O manejar como se considere apropiado, e.g., devolver f_banda[0]

    # Suma acumulada de la potencia en la banda
    vector_suma = np.cumsum(psd_banda)

    # Encontrar el índice donde la suma acumulada alcanza la mitad de la potencia total
    # np.where(...)[0] devuelve los índices que cumplen la condición.
    # [-1] selecciona el último índice, similar a max(find(...)) en MATLAB
    indices_mitad = np.where(vector_suma <= (potencia_total / 2))[0]

    if indices_mitad.size == 0:
        # Si ningún índice cumple (vector_suma[0] > potencia_total / 2),
        # la MF es la primera frecuencia de la banda.
        ind_mf_en_banda = 0
    else:
        # El índice deseado es el último que cumple la condición
        ind_mf_en_banda = indices_mitad[-1]
        # Si la potencia está exactamente dividida, podríamos necesitar interpolar
        # o decidir si tomar el índice actual o el siguiente.
        # La implementación MATLAB original toma este índice.

    # La frecuencia mediana es la frecuencia correspondiente a ese índice
    # Assuming f_banda is a 1D array, f_banda[ind_mf_en_banda] will be a scalar float.
    frecuencia_mediana = f_banda[ind_mf_en_banda]

    return float(frecuencia_mediana) # Explicitly cast to float to satisfy linter

def calcular_mf_vector(psd: np.ndarray, f: np.ndarray, banda: list[float]) -> np.ndarray:
    """
    Vectorized version of calcular_mf that processes multiple segments at once.

    Args:
        psd: Array of shape (n_segments, n_freqs) containing PSD values for multiple segments
        f: Array of shape (n_freqs,) containing frequency values
        banda: List or tuple with two elements [f_min, f_max] specifying the frequency band

    Returns:
        Array of shape (n_segments,) containing median frequencies for each segment.
        None values are represented as np.nan.
    """
    if psd.ndim != 2:
        raise ValueError("psd must be 2D array with shape (n_segments, n_freqs)")
    if f.ndim != 1:
        raise ValueError("f must be 1D array")
    if psd.shape[1] != len(f):
        raise ValueError("psd and f must have matching frequency dimensions")
    if not isinstance(banda, (list, tuple)) or len(banda) != 2:
        raise ValueError("banda must be a list or tuple of two elements [f_min, f_max]")
    if banda[0] > banda[1]:
        raise ValueError("f_min cannot be greater than f_max")

    # Find indices within frequency band
    mask = (f >= banda[0]) & (f <= banda[1])
    if not np.any(mask):
        print(f"Warning: No frequencies found in specified band [{banda[0]}, {banda[1]}]")
        return np.full(psd.shape[0], np.nan)

    # Extract data within band
    psd_banda = psd[:, mask]
    f_banda = f[mask]

    # Calculate total power for each segment
    potencia_total = np.sum(psd_banda, axis=1)
    
    # Initialize result array with NaN
    mf_results = np.full(psd.shape[0], np.nan)
    
    # Process segments with non-zero power
    valid_segments = potencia_total > 0
    if not np.any(valid_segments):
        print(f"Warning: Total power in band [{banda[0]}, {banda[1]}] is zero or negative for all segments")
        return mf_results

    # Calculate cumulative sum for valid segments
    cumsum = np.cumsum(psd_banda[valid_segments], axis=1)
    half_power = potencia_total[valid_segments, np.newaxis] / 2
    
    # Find indices where cumsum exceeds half power
    # Use <= instead of > to match non-vectorized version exactly
    indices = np.zeros(len(valid_segments), dtype=int)
    for i, (cs, hp) in enumerate(zip(cumsum, half_power)):
        idx = np.where(cs <= hp)[0]
        indices[i] = idx[-1] if len(idx) > 0 else 0
    
    # Get median frequencies for valid segments
    mf_results[valid_segments] = f_banda[indices]
    
    return mf_results

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
    mf_results = np.array([calcular_mf(psd[i], f, banda) for i in range(n_segments)], dtype=float)
    non_vector_time = time.time() - start_time
    
    # Test vectorized version
    start_time = time.time()
    mf_results_vector = calcular_mf_vector(psd, f, banda)
    vector_time = time.time() - start_time
    
    # Compare results
    is_close = np.allclose(mf_results, mf_results_vector, equal_nan=True, rtol=1e-10, atol=1e-10)
    print(f"Results match: {is_close}")
    if not is_close:
        # Print some statistics about the differences
        mask = ~np.isnan(mf_results) & ~np.isnan(mf_results_vector)
        if np.any(mask):
            diffs = np.abs(mf_results[mask] - mf_results_vector[mask])
            print(f"Max difference: {np.max(diffs)}")
            print(f"Mean difference: {np.mean(diffs)}")
            print(f"Number of different values: {np.sum(diffs > 0)}")
    print(f"Non-vectorized time: {non_vector_time:.3f} seconds")
    print(f"Vectorized time: {vector_time:.3f} seconds")
    print(f"Speedup: {non_vector_time/vector_time:.1f}x")
