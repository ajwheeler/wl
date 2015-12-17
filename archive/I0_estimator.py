import numpy as np

def estimated_I0(data, center, r0):
    numerator = 0
    denominator = 0
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            dist = np.sqrt((i-center[0])**2 + (j-center[1])**2)
            numerator += np.exp(-dist/r0)*val
            denominator += np.exp(-2*dist/r0)
    return numerator/denominator
