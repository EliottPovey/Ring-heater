import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def w(z, w0, z0):
    return w0 * np.sqrt(1 + ((z - z0) / (np.pi * w0**2 / 0.000001559))**2)

xdata = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
ydata = [0.00191962/2, 0.00183975/2, 0.0017588/2, 0.00169308/2, 0.0016268/2, 0.00158108/2, 0.00154911/2, 0.00151754/2,
         0.00150782/2, 0.00150818/2, 0.0015209/2, 0.00154915/2, 0.0015894/2, 0.00163844/2, 0.00169545/2, 0.00175646/2]

p0 = [0.1, 1]

popt, pcov = curve_fit(w, xdata, ydata, p0=p0, maxfev=10000)
perr = np.sqrt(np.diag(pcov))

w0_fit, z0_fit = popt
print(f"w0 = {w0_fit:.6f} m  ±  {perr[0]:.6f}")
print(f"z0 = {z0_fit:.6f} m  ±  {perr[1]:.6f}")

z_fine = np.linspace(xdata[0], xdata[-1], 500)
plt.figure(figsize=(8, 5))
plt.scatter(xdata, ydata, color='red', zorder=5, label='Data')
plt.plot(z_fine, w(z_fine, *popt), label=f'Fit: w0={w0_fit:.5f} m, z0={z0_fit:.4f} m')
plt.xlabel('z (m)')
plt.ylabel('Beam radius w(z) (m)')
plt.title('Gaussian Beam Waist Fit')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()