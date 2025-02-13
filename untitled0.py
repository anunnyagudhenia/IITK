# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 15:40:38 2025

@author: anunn
"""

import numpy as np
import matplotlib.pyplot as plt

def compute_capacitance_cylinder(h, r0, M, N, V=1):
    """
    Compute the capacitance of an isolated metallic cylinder using the moment method.
    :param h: Height of the cylinder
    :param r0: Radius of the cylinder
    :param M: Number of circumferential divisions
    :param N: Number of axial divisions
    :param V: Applied potential
    :return: Capacitance per unit length (C/h) and charge distribution
    """
    MN = M * N
    L = np.zeros((MN, MN))  # Matrix for solving charge distribution
    
    # Compute matrix elements
    for m in range(MN):
        for n in range(MN):
            if m == n:
                L[m, n] = (1 / (2 * np.pi * r0)) * np.log((h / 2 + np.sqrt((h / 2) ** 2 + r0 ** 2)) / r0)
            else:
                zm = (m // M) * (h / N)
                zn = (n // M) * (h / N)
                theta_m = 2 * np.pi * (m % M) / M
                theta_n = 2 * np.pi * (n % M) / M
                L[m, n] = 1 / np.sqrt(r0 ** 2 * (np.cos(theta_m) - np.cos(theta_n)) ** 2 + (zm - zn) ** 2)
    
    # Solve for charge distribution
    V_vector = np.full((MN,), V)
    Q_vector = np.linalg.solve(L, V_vector)
    
    # Compute total charge
    Q_total = np.sum(Q_vector) * (h / N) * (2 * np.pi * r0 / M)
    capacitance = Q_total / V
    
    return capacitance / h, Q_vector.reshape((N, M))

def compute_capacitance_truncated_cone(h, r1, r2, M, N, V=1):
    """
    Compute the capacitance of an isolated metallic truncated cone using the moment method.
    :param h: Height of the truncated cone
    :param r1: Top radius of the truncated cone
    :param r2: Bottom radius of the truncated cone
    :param M: Number of circumferential divisions
    :param N: Number of axial divisions
    :param V: Applied potential
    :return: Capacitance per unit length (C/h) and charge distribution
    """
    MN = M * N
    L = np.zeros((MN, MN))  # Matrix for solving charge distribution
    
    # Compute matrix elements
    for m in range(MN):
        for n in range(MN):
            zm = (m // M) * (h / N)
            zn = (n // M) * (h / N)
            r_m = r1 + (r2 - r1) * (zm / h)
            r_n = r1 + (r2 - r1) * (zn / h)
            theta_m = 2 * np.pi * (m % M) / M
            theta_n = 2 * np.pi * (n % M) / M
            if m == n:
                L[m, n] = (1 / (2 * np.pi * r_m)) * np.log((h / 2 + np.sqrt((h / 2) ** 2 + r_m ** 2)) / r_m)
            else:
                L[m, n] = 1 / np.sqrt(r_m ** 2 * (np.cos(theta_m) - np.cos(theta_n)) ** 2 + (zm - zn) ** 2)
    
    # Solve for charge distribution
    V_vector = np.full((MN,), V)
    Q_vector = np.linalg.solve(L, V_vector)
    
    # Compute total charge
    Q_total = np.sum(Q_vector) * (h / N) * (2 * np.pi * np.mean([r1, r2]) / M)
    capacitance = Q_total / V
    
    return capacitance / h, Q_vector.reshape((N, M))

# Parameters for Cylinder
h = 2  # Height of cylinder
r0 = 1  # Radius of cylinder
M = 11  # Circumferential divisions
N = 9  # Axial divisions

C_per_h_cyl, charge_distribution_cyl = compute_capacitance_cylinder(h, r0, M, N)
print(C_per_h_cyl, charge_distribution_cyl)
"""
# Parameters for Truncated Cone
r1 = 0.3  # Top radius of truncated cone
r2 = 0.6  # Bottom radius of truncated cone

C_per_h_cone, charge_distribution_cone = compute_capacitance_truncated_cone(h, r1, r2, M, N)

# Plot charge distribution for cylinder
plt.imshow(charge_distribution_cyl, cmap='hot', aspect='auto')
plt.colorbar(label='Charge Density')
plt.title('Charge Distribution on the Cylinder')
plt.xlabel('Circumferential Segments')
plt.ylabel('Axial Segments')
plt.show()

# Plot charge distribution for truncated cone
plt.imshow(charge_distribution_cone, cmap='hot', aspect='auto')
plt.colorbar(label='Charge Density')
plt.title('Charge Distribution on the Truncated Cone')
plt.xlabel('Circumferential Segments')
plt.ylabel('Axial Segments')
plt.show()

print(f"Capacitance per unit length (Cylinder): {C_per_h_cyl:.4e} F/m")
print(f"Capacitance per unit length (Truncated Cone): {C_per_h_cone:.4e} F/m")"""
