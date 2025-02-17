import numpy as np
import matplotlib.pyplot as plt
def compute_capacitance_per_unit_length(h, r0, M, N, epsilon_0=8.854e-12):
    """
    Compute the capacitance per unit length of a metallic cylinder using moment method.
    :param h: Height of the cylinder
    :param r0: Radius of the cylinder
    :param M: Number of subsections along circumference
    :param N: Number of subsections along height
    :param epsilon_0: Permittivity of free space (default: 8.854e-12 F/m)
    :return: Capacitance per unit length (F/m)
    """
    MN = M * N  # Total number of subsections
    L_matrix = np.zeros((MN, MN))
    
    for m in range(MN):
        for n in range(MN):
            if m != n:
                theta_m, theta_n = (2 * np.pi * (m % M) / M, 2 * np.pi * (n % M) / M)
                z_m, z_n = ((m // M) * h / N, (n // M) * h / N)
                
                distance = np.sqrt((2 * r0 * np.sin((theta_m - theta_n) / 2))**2 + (z_m - z_n)**2)
                L_matrix[m, n] = 1 / (4 * np.pi * epsilon_0 * distance)
            else:
                L_matrix[m, n] = (1 / (2 * np.pi * epsilon_0)) * np.log((h / N) / (r0 / M))
    
    L_inv = np.linalg.inv(L_matrix)
    capacitance = np.sum(L_inv)
    
    return capacitance / h  # Capacitance per unit length

# Compute and print capacitance per unit length
cap_per_unit_length1 = compute_capacitance_per_unit_length(2.0, 1.0, 11, 9)
cap_per_unit_length2 = compute_capacitance_per_unit_length(4.0, 1.0,9, 11)
cap_per_unit_length3 = compute_capacitance_per_unit_length(12.0, 1.0,9, 13)
cap_per_unit_length4 = compute_capacitance_per_unit_length(40.0, 1.0,10, 30)
cap_per_unit_length5 = compute_capacitance_per_unit_length(120.0, 1.0,10, 40)
"""print(f"Capacitance per unit length: {cap_per_unit_length1:.4e} F/m")
print(f"Capacitance per unit length: {cap_per_unit_length2:.4e} F/m")
print(f"Capacitance per unit length: {cap_per_unit_length3:.4e} F/m")
print(f"Capacitance per unit length: {cap_per_unit_length4:.4e} F/m")
print(f"Capacitance per unit length: {cap_per_unit_length5:.4e} F/m")
"""
X=[2,4,12,40,120]
Y=[cap_per_unit_length1,cap_per_unit_length2,cap_per_unit_length3,cap_per_unit_length4,cap_per_unit_length5]
plt.plot(X,Y)