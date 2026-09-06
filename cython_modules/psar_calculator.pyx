import numpy as np
cimport numpy as np

cdef double max_acceleration
cdef double acceleration

def calculate_psar_cython(np.ndarray[double, ndim=1] highs, 
                           np.ndarray[double, ndim=1] lows, 
                           double acceleration_factor, 
                           double max_af):
    cdef int n = highs.shape[0]
    
    # Allocate memory for arrays
    cdef np.ndarray[double, ndim=1] psar = np.empty(n)
    cdef np.ndarray[int, ndim=1] trend = np.ones(n, dtype=np.int32)
    cdef np.ndarray[double, ndim=1] extreme_point = np.empty(n)
    cdef np.ndarray[double, ndim=1] af = np.full(n, acceleration_factor)

    # Initial conditions
    psar[0] = highs[0]  # Start with the high as the initial PSAR
    extreme_point[0] = highs[0]

    # Iterate through the data
    cdef int i
    for i in range(1, n):
        psar[i] = psar[i - 1] + af[i - 1] * (extreme_point[i - 1] - psar[i - 1])

        if trend[i - 1] == 1:  # Uptrend
            extreme_point[i] = max(extreme_point[i - 1], highs[i])

            if lows[i] < psar[i]:  # Trend reversal
                trend[i] = -1
                psar[i] = extreme_point[i - 1]
                extreme_point[i] = lows[i]
                af[i] = acceleration_factor
            else:
                trend[i] = 1
                af[i] = min(af[i - 1] + acceleration_factor, max_af)
        else:  # Downtrend
            extreme_point[i] = min(extreme_point[i - 1], lows[i])

            if highs[i] > psar[i]:  # Trend reversal
                trend[i] = 1
                psar[i] = extreme_point[i - 1]
                extreme_point[i] = highs[i]
                af[i] = acceleration_factor
            else:
                trend[i] = -1
                af[i] = min(af[i - 1] + acceleration_factor, max_af)

    return psar[-1]  # Return last PSAR value
