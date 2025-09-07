import numpy as np
from typing import Tuple, List, Optional

def calcular_iaftf(psd: np.ndarray, f: np.ndarray, banda: List[float], q: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """
    Calcula los parámetros 'individual alpha frequency' (IAF) y
    'transition frequency' (TF) para la distribución contenida en PSD.

    Args:
        psd: Array de NumPy que representa la Densidad Espectral de Potencia.
        f: Array de NumPy que representa el vector de frecuencias correspondiente a psd.
        banda: Lista o tupla con dos elementos [f_min, f_max] especificando
               la banda de frecuencia de interés (actualmente no se usa explícitamente
               en los cálculos de IAF/TF en el código original, pero se mantiene por consistencia).
        q: Lista o tupla con dos elementos [q_min, q_max] que controla los
           intervalos de frecuencia a considerar para calcular la IAF (típicamente [4, 15] Hz).

    Returns:
        Una tupla conteniendo (FrecuenciaAlfa, FrecuenciaTransision).
        Ambos pueden ser None si los cálculos no son posibles (e.g., PSD con NaNs).

    See also: calcular_parametro, calcular_mf, calcular_sef
    """

    if len(psd) != len(f):
        raise ValueError("psd y f deben tener la misma longitud.")
    if not isinstance(banda, (list, tuple)) or len(banda) != 2:
        raise ValueError("banda debe ser una lista o tupla de dos elementos [f_min, f_max].")
    if banda[0] > banda[1]:
        raise ValueError("El primer elemento de banda (f_min) no puede ser mayor que el segundo (f_max).")
    if not isinstance(q, (list, tuple)) or len(q) != 2:
        raise ValueError("q debe ser una lista o tupla de dos elementos [q_min, q_max].")
    if q[0] > q[1]:
        raise ValueError("El primer elemento de q (q_min) no puede ser mayor que el segundo (q_max).")

    # Se buscan los indices positivos entre los valores indicados en q.
    indbanda_q_indices = np.where((f >= q[0]) & (f <= q[1]))[0]

    if indbanda_q_indices.size < 3: # MATLAB original accede a indbanda(3)
        print(f"Advertencia: No hay suficientes puntos de datos en el rango q [{q[0]}, {q[1]}] para calcular IAF/TF.")
        return None, None

    # Se comprueba si la PSD está formada por NaNs (artefacto)
    # En MATLAB indbanda(3) es el tercer elemento. En Python, es indbanda_q_indices[2]
    if np.isnan(psd[indbanda_q_indices[2]]):
        return None, None

    frecuencia_alfa: Optional[float] = None
    frecuencia_transision: Optional[float] = None

    psd_q_banda = psd[indbanda_q_indices]
    f_q_banda = f[indbanda_q_indices]

    # Se calcula el valor de la potencia total para el espectro en la banda q.
    potencia_total_q = np.sum(psd_q_banda)

    if potencia_total_q <= 0:
        print(f"Advertencia: La potencia total en la banda q [{q[0]}, {q[1]}] es cero o negativa.")
        # Podríamos devolver f_q_banda[0] o None dependiendo del comportamiento deseado.
        # El código MATLAB procedería y FrecuenciaAlfa sería f(indbanda(1))
        # Sin embargo, si potencia_total_q es 0, vectorsuma también lo será.
        # np.where(vectorsuma <= potenciatotal/2) podría devolver todos los índices
        # o ninguno si psd_q_banda es todo ceros.
        # Para evitar división por cero o comportamiento indefinido, se retorna None aquí.
        # Si se prefiere emular el comportamiento de MATLAB incluso con potencia cero,
        # esta lógica necesitaría ajustarse.
        return None, None


    # Se suman los valores de potencia relativa para el espectro en la banda q.
    vectorsuma_q = np.cumsum(psd_q_banda)

    # Se coge el índice para el cual se tiene la mitad de la potencia total.
    # indices_mitad_q contendrá los índices dentro de psd_q_banda
    indices_mitad_q = np.where(vectorsuma_q <= potencia_total_q / 2)[0]

    if indices_mitad_q.size == 0:
        # Si no se ha seleccionado ningún índice es porque en el primer valor esta
        # mas del 50% de la potencia total.
        ind_media_en_q_banda = 0
    else:
        ind_media_en_q_banda = indices_mitad_q[-1] # Equivalente a max(find(...))

    frecuencia_alfa = f_q_banda[ind_media_en_q_banda]
    # indmedia original se refería al índice en el vector f global
    indmedia_global = indbanda_q_indices[ind_media_en_q_banda]


    ######################################################################
    # Se calcula el parametro TF.
    ######################################################################
    # Se buscan los índices entre 0.5 Hz y la IAF (usando el índice global de IAF).
    # indinferiorTF: primer índice donde f >= 0.5 Hz
    # indsuperiorTF: índice global de FrecuenciaAlfa
    
    indinferiorTF_candidates = np.where(f >= 0.5)[0]
    if indinferiorTF_candidates.size == 0:
        print("Advertencia: No se encontraron frecuencias >= 0.5 Hz para el cálculo de TF.")
        return frecuencia_alfa, None # Devolver IAF si se calculó, TF como None
    
    indinferiorTF_global = indinferiorTF_candidates[0] # min(find(f >= 0.5))
    indsuperiorTF_global = indmedia_global # Índice global de FrecuenciaAlfa

    if indinferiorTF_global > indsuperiorTF_global:
        print(f"Advertencia: El límite inferior para TF ({f[indinferiorTF_global]} Hz) es mayor que IAF ({frecuencia_alfa} Hz). TF no se puede calcular.")
        return frecuencia_alfa, None

    # Se buscan los índices entre los valores indicados (índices globales).
    indTF_global = np.arange(indinferiorTF_global, indsuperiorTF_global + 1)

    if indTF_global.size == 0:
        print("Advertencia: No hay rango de frecuencias para calcular TF.")
        return frecuencia_alfa, None

    # Se toma el trozo de la PSD entre las frecuencias especificadas.
    psd_recortada_tf = psd[indTF_global]
    f_recortada_tf = f[indTF_global]

    if psd_recortada_tf.size == 0:
        print("Advertencia: psd_recortada_tf está vacía. No se puede calcular TF.")
        return frecuencia_alfa, None


    # Se calcula el valor de la potencia total para el espectro recortado.
    potenciatotal_tf = np.sum(psd_recortada_tf)

    if potenciatotal_tf <= 0:
        print(f"Advertencia: La potencia total en la banda TF [{f_recortada_tf[0]}, {f_recortada_tf[-1]}] es cero o negativa.")
        # De manera similar a IAF, si la potencia es cero, el MATLAB original asignaría la primera frecuencia.
        # Podríamos asignar f_recortada_tf[0] o retornar None.
        # Optaremos por retornar None para mayor claridad de que el cálculo no fue robusto.
        if f_recortada_tf.size > 0:
             frecuencia_transision = f_recortada_tf[0] # Emula comportamiento MATLAB
        else:
             return frecuencia_alfa, None # No hay frecuencias en el rango TF
        return frecuencia_alfa, frecuencia_transision


    # Se suman los valores de potencia relativa para el espectro positivo.
    vectorsuma_tf = np.cumsum(psd_recortada_tf)

    # Se coge el índice para el cual se tiene la mitad de la potencia total.
    # indices_mitad_tf contendrá los índices dentro de psd_recortada_tf
    indices_mitad_tf = np.where(vectorsuma_tf <= potenciatotal_tf / 2)[0]

    if indices_mitad_tf.size == 0:
        # Si no se ha seleccionado ningun indice es porque en el primer valor esta
        # mas del 50% de la potencia total.
        ind_media_tf_en_recortada = 0
    else:
        ind_media_tf_en_recortada = indices_mitad_tf[-1]

    if f_recortada_tf.size > ind_media_tf_en_recortada :
        frecuencia_transision = f_recortada_tf[ind_media_tf_en_recortada]
    elif f_recortada_tf.size > 0 : # Si ind_media_tf_en_recortada está fuera de rango pero hay elementos
        frecuencia_transision = f_recortada_tf[0] # Comportamiento por defecto del MATLAB si indTF(1)
    else: # No hay elementos en f_recortada_tf
        return frecuencia_alfa, None


    return frecuencia_alfa, frecuencia_transision

def calcular_iaftf_vector(psd: np.ndarray, f: np.ndarray, banda: List[float], q: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized version of calcular_iaftf that processes multiple segments at once.

    Args:
        psd: Array of shape (n_segments, n_freqs) containing PSD values for multiple segments
        f: Array of shape (n_freqs,) containing frequency values
        banda: List or tuple with two elements [f_min, f_max] specifying the frequency band
        q: List or tuple with two elements [q_min, q_max] controlling frequency intervals for IAF

    Returns:
        Tuple of two arrays of shape (n_segments,):
        - Array of IAF values (NaN for invalid calculations)
        - Array of TF values (NaN for invalid calculations)
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
    if not isinstance(q, (list, tuple)) or len(q) != 2:
        raise ValueError("q must be a list or tuple of two elements [q_min, q_max]")
    if q[0] > q[1]:
        raise ValueError("q_min cannot be greater than q_max")

    n_segments = psd.shape[0]
    iaf_results = np.full(n_segments, np.nan)
    tf_results = np.full(n_segments, np.nan)

    # Find indices within q band
    mask_q = (f >= q[0]) & (f <= q[1])
    if not np.any(mask_q) or np.sum(mask_q) < 3:
        print(f"Warning: Not enough data points in q range [{q[0]}, {q[1]}] for IAF/TF calculation")
        return iaf_results, tf_results

    # Extract data within q band
    psd_q_banda = psd[:, mask_q]
    f_q_banda = f[mask_q]

    # Calculate total power for each segment in q band
    potencia_total_q = np.sum(psd_q_banda, axis=1)
    
    # Process segments with non-zero power
    valid_segments = potencia_total_q > 0
    if not np.any(valid_segments):
        print(f"Warning: Total power in q band [{q[0]}, {q[1]}] is zero or negative for all segments")
        return iaf_results, tf_results

    # Calculate cumulative sum for valid segments
    cumsum_q = np.cumsum(psd_q_banda[valid_segments], axis=1)
    half_power_q = potencia_total_q[valid_segments, np.newaxis] / 2
    
    # Find indices where cumsum exceeds half power
    # Use <= instead of > to match non-vectorized version exactly
    indices_q = np.zeros(np.sum(valid_segments), dtype=int)
    for i, (cs, hp) in enumerate(zip(cumsum_q, half_power_q)):
        idx = np.where(cs <= hp)[0]
        indices_q[i] = idx[-1] if len(idx) > 0 else 0
    
    # Get IAF values for valid segments
    iaf_results[valid_segments] = f_q_banda[indices_q]

    # Calculate TF for segments with valid IAF
    valid_iaf_segments = ~np.isnan(iaf_results)
    if not np.any(valid_iaf_segments):
        return iaf_results, tf_results

    # Find indices for TF calculation (0.5 Hz to IAF)
    indinferiorTF = np.searchsorted(f, 0.5)
    if indinferiorTF >= len(f):
        print("Warning: No frequencies >= 0.5 Hz found for TF calculation")
        return iaf_results, tf_results

    # For each valid segment, calculate TF
    for i in np.where(valid_iaf_segments)[0]:
        iaf = iaf_results[i]
        indsuperiorTF = np.searchsorted(f, iaf)
        
        if indinferiorTF > indsuperiorTF:
            continue  # Skip if lower bound > upper bound
            
        # Extract PSD and frequencies for TF calculation
        psd_tf = psd[i, indinferiorTF:indsuperiorTF + 1]
        f_tf = f[indinferiorTF:indsuperiorTF + 1]
        
        if len(psd_tf) == 0:
            continue
            
        potencia_total_tf = np.sum(psd_tf)
        if potencia_total_tf <= 0:
            tf_results[i] = f_tf[0]  # Emulate MATLAB behavior
            continue
            
        # Calculate TF
        cumsum_tf = np.cumsum(psd_tf)
        indices_tf = np.where(cumsum_tf <= potencia_total_tf / 2)[0]
        
        if len(indices_tf) == 0:
            tf_results[i] = f_tf[0]
        else:
            tf_results[i] = f_tf[indices_tf[-1]]

    return iaf_results, tf_results

if __name__ == "__main__":
    import time
    
    # Create dummy test data
    n_segments = 1000
    n_freqs = 1000
    f = np.linspace(0, 100, n_freqs)
    psd = np.random.rand(n_segments, n_freqs)
    banda = [20.0, 80.0]
    q = [4.0, 15.0]
    
    # Test non-vectorized version
    start_time = time.time()
    iaf_results = np.zeros(n_segments)
    tf_results = np.zeros(n_segments)
    for i in range(n_segments):
        iaf, tf = calcular_iaftf(psd[i], f, banda, q)
        iaf_results[i] = iaf if iaf is not None else np.nan
        tf_results[i] = tf if tf is not None else np.nan
    non_vector_time = time.time() - start_time
    
    # Test vectorized version
    start_time = time.time()
    iaf_results_vector, tf_results_vector = calcular_iaftf_vector(psd, f, banda, q)
    vector_time = time.time() - start_time
    
    # Compare results
    iaf_match = np.allclose(iaf_results, iaf_results_vector, equal_nan=True, rtol=1e-10, atol=1e-10)
    tf_match = np.allclose(tf_results, tf_results_vector, equal_nan=True, rtol=1e-10, atol=1e-10)
    print(f"IAF results match: {iaf_match}")
    print(f"TF results match: {tf_match}")
    
    if not iaf_match or not tf_match:
        # Print some statistics about the differences
        for name, results, results_vector in [("IAF", iaf_results, iaf_results_vector), 
                                            ("TF", tf_results, tf_results_vector)]:
            mask = ~np.isnan(results) & ~np.isnan(results_vector)
            if np.any(mask):
                diffs = np.abs(results[mask] - results_vector[mask])
                print(f"\n{name} differences:")
                print(f"Max difference: {np.max(diffs)}")
                print(f"Mean difference: {np.mean(diffs)}")
                print(f"Number of different values: {np.sum(diffs > 0)}")
    
    print(f"\nNon-vectorized time: {non_vector_time:.3f} seconds")
    print(f"Vectorized time: {vector_time:.3f} seconds")
    print(f"Speedup: {non_vector_time/vector_time:.1f}x")
