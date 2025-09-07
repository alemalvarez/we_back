import numpy as np
from typing import List, Optional

def calcular_scf(psd: np.ndarray, f: np.ndarray, banda: List[float]) -> Optional[float]:
    """
    Calcula el Factor de Cresta Espectral (SCF) de la PSD dentro de una banda de frecuencia específica.

    El Factor de Cresta Espectral (SCF) es una medida de la tonalidad de la señal.
    Proporciona una estimación de la concentración del espectro de potencia en torno a
    unas pocas componentes. Un SCF elevado indica que el espectro tiene una componente
    claramente dominante, mientras que un SCF pequeño se obtiene cuando las componentes
    del espectro son similares entre sí. El SCF se define como el máximo del espectro
    de potencia dividido por la potencia media en la banda de análisis.

    Args:
        psd: Array de NumPy 1D que representa la Densidad Espectral de Potencia.
             Se asume que los valores de PSD son no negativos.
        f: Array de NumPy 1D que representa el vector de frecuencias correspondiente a psd.
        banda: Lista con dos elementos numéricos [f_min, f_max] especificando
               la banda de frecuencia de interés (en las mismas unidades que f).

    Returns:
        El Factor de Cresta Espectral calculado como un float.
        Devuelve None si:
        - No se encuentran frecuencias en la banda especificada.
        - La potencia total (y por ende, media) en la banda es cero o negativa.
        
    Raises:
        TypeError: Si 'psd' o 'f' no son arrays de NumPy, o si los elementos de 'banda' no son numéricos.
        ValueError: Si las entradas no cumplen con los requisitos dimensionales, de longitud,
                    o si 'banda' no está correctamente formateada o f_min > f_max.
    """

    # --- Input Validation ---
    if not isinstance(psd, np.ndarray):
        raise TypeError("El argumento 'psd' debe ser un array de NumPy.")
    if not isinstance(f, np.ndarray):
        raise TypeError("El argumento 'f' debe ser un array de NumPy.")

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
        raise TypeError(
            f"Los elementos de 'banda' ('{banda[0]}', '{banda[1]}') deben ser números (convertibles a float). Error original: {e}"
        )

    if f_min > f_max:
        raise ValueError(
            f"El primer elemento de 'banda' (f_min={f_min}) no puede ser mayor que el segundo (f_max={f_max})."
        )

    # --- Selección de la banda de frecuencia ---
    # Encuentra los índices de las frecuencias que caen dentro de la banda especificada.
    indbanda = np.where((f >= f_min) & (f <= f_max))[0]

    if indbanda.size == 0:
        print(f"Advertencia: No se encontraron frecuencias en la banda especificada [{f_min}, {f_max}].")
        return None

    psd_banda = psd[indbanda]
    
    # A estas alturas, psd_banda.size > 0 porque indbanda.size > 0.

    # --- Cálculo del SCF ---
    max_psd_banda = np.max(psd_banda)
    
    # Suma de PSD en banda (potencia total en la banda)
    sum_psd_banda = np.sum(psd_banda)

    # Si la potencia total en la banda es cero o negativa.
    # (PSD debería ser no-negativa, por lo que sum_psd_banda < 0 es anómalo).
    if sum_psd_banda <= 1e-9: # Usar una pequeña tolerancia para comparar con cero
        # Si sum_psd_banda es efectivamente 0 (y PSDs son non-negativos), todos los psd_banda son 0.
        # max_psd_banda es 0. mean_psd_banda es 0. SCF es 0/0 -> indefinido (NaN).
        print(
            f"Advertencia: La potencia total en la banda [{f_min}, {f_max}] es {sum_psd_banda:.4e} "
            f"(efectivamente cero o negativa). SCF no puede calcularse o es indefinido."
        )
        return None

    # A este punto, sum_psd_banda > 0 (y psd_banda.size > 0).
    mean_psd_banda = sum_psd_banda / psd_banda.size
    
    # Dado que mean_psd_banda > 0, la división es segura y el resultado será finito.
    scf = max_psd_banda / mean_psd_banda
    
    return float(scf)

def calcular_scf_vector(psd: np.ndarray, f: np.ndarray, banda: List[float]) -> np.ndarray:
    """
    Vectorized version of calcular_scf that processes multiple segments at once.

    Args:
        psd: Array of shape (n_segments, n_freqs) containing PSD values for multiple segments
        f: Array of shape (n_freqs,) containing frequency values
        banda: List with two numeric elements [f_min, f_max] specifying the frequency band

    Returns:
        Array of shape (n_segments,) containing spectral crest factor values.
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
    
    # Process each segment
    for i in range(n_segments):
        # Calculate max PSD in band
        max_psd_banda = np.max(psd_banda[i])
        
        # Calculate total power in band
        sum_psd_banda = np.sum(psd_banda[i])
        
        if sum_psd_banda <= 1e-9:  # Small threshold to avoid division by zero/noise
            continue
            
        # Calculate mean PSD in band
        mean_psd_banda = sum_psd_banda / psd_banda[i].size
        
        # Calculate SCF
        results[i] = max_psd_banda / mean_psd_banda
            
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
    scf_results = np.array([calcular_scf(psd[i], f, banda) for i in range(n_segments)], dtype=float)
    non_vector_time = time.time() - start_time
    
    # Test vectorized version
    start_time = time.time()
    scf_results_vector = calcular_scf_vector(psd, f, banda)
    vector_time = time.time() - start_time
    
    # Compare results
    is_close = np.allclose(scf_results, scf_results_vector, equal_nan=True, rtol=1e-10, atol=1e-10)
    print(f"Results match: {is_close}")
    if not is_close:
        # Print some statistics about the differences
        mask = ~np.isnan(scf_results) & ~np.isnan(scf_results_vector)
        if np.any(mask):
            diffs = np.abs(scf_results[mask] - scf_results_vector[mask])
            print(f"Max difference: {np.max(diffs)}")
            print(f"Mean difference: {np.mean(diffs)}")
            print(f"Number of different values: {np.sum(diffs > 0)}")
    
    print(f"Non-vectorized time: {non_vector_time:.3f} seconds")
    print(f"Vectorized time: {vector_time:.3f} seconds")
    print(f"Speedup: {non_vector_time/vector_time:.1f}x") 