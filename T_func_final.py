import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1, i1, i0
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── Physical parameters ────────────────────────────────────────────────────────
h      = 0.0012
a      = 0.00635
rho    = 2202
C      = 772
K      = 1.38
sig    = 0.9 * 5.67e-8
T_ext  = 300
gammaP = 1
b      = -0.00216
c      = 0.00216
dndT   = 0.86e-5
tau    = (4 * sig * T_ext**3 * a) / K
kappa  = K / (C * rho * a**2)   

# ── Eigenvalue equations ───────────────────────────────────────────────────────
def equ(u):   return u - (tau / np.tan((u * h) / (2 * a)))
def eqv(v):   return v + (tau * np.tan((v * h) / (2 * a)))
def eqz(zeta): return j1(zeta) * zeta - tau * j0(zeta)

def find_roots_trig(func, singularity_type, u_min, u_max, n_singularities=20):
    if singularity_type == 'cot':
        sing = [2 * n * np.pi * a / h for n in range(-n_singularities, n_singularities + 1)]
    else:
        sing = [(2*n + 1) * np.pi * a / h for n in range(-n_singularities, n_singularities + 1)]
    sing = sorted(s for s in sing if u_min <= s <= u_max)
    intervals = [u_min] + sing + [u_max]
    eps = 1e-8
    roots = []
    for i in range(len(intervals) - 1):
        lo, hi = intervals[i] + eps, intervals[i + 1] - eps
        if lo >= hi:
            continue
        try:
            f_lo, f_hi = func(lo), func(hi)
        except Exception:
            continue
        if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi < 0:
            roots.append(brentq(func, lo, hi, xtol=1e-10))
    return roots

def find_roots_bessel(func, zeta_min, zeta_max, n_points=10000):
    zeta = np.linspace(zeta_min, zeta_max, n_points)
    f = func(zeta)
    roots = []
    for i in range(len(zeta) - 1):
        if np.isfinite(f[i]) and np.isfinite(f[i+1]) and f[i] * f[i+1] < 0:
            roots.append(brentq(func, zeta[i], zeta[i+1], xtol=1e-10))
    return roots

u = np.array(find_roots_trig(equ, 'cot', 0, 1000))
v = np.array([r for r in find_roots_trig(eqv, 'tan', 0, 1000) if abs(r) > 1e-8])
zeta = np.array(find_roots_bessel(eqz, 0, 1000))

A = ((2*gammaP) / (2*np.pi*a*(c-b)*K)
     * (np.sin(u*c/a) - np.sin(u*b/a))
     / ((u*h/a + np.sin(u*h/a)) * (i1(u)*u/a + tau*i0(u)/a)))

B = ((2*gammaP) / (2*np.pi*a*(c-b)*K)
     * (-np.cos(v*c/a) + np.cos(v*b/a))
     / ((v*h/a - np.sin(v*h/a)) * (i1(v)*v/a + tau*i0(v)/a)))

u2d = u[:, None]; z2d = zeta[None, :]       
v2d = v[:, None]

cu = (2 * z2d**2 * (u2d*i1(u2d)*j0(z2d) + z2d*i0(u2d)*j1(z2d))
      / ((u2d**2 + z2d**2) * (tau**2 + z2d**2) * j0(z2d)**2))   

cv = (2 * z2d**2 * (v2d*i1(v2d)*j0(z2d) + z2d*i0(v2d)*j1(z2d))
      / ((v2d**2 + z2d**2) * (tau**2 + z2d**2) * j0(z2d)**2))  


ACu = A[:, None] * cu    
BCv = B[:, None] * cv   



def _exp_factors(t_arr):
    """Return exp(-u^2*kappa*t), exp(-v^2*kappa*t), exp(-zeta^2*kappa*t).
    Shapes: (Mu, Nt), (Mv, Nt), (P, Nt)  [or 1-D if t_arr is scalar]."""
    t = np.atleast_1d(np.asarray(t_arr, float))
    exp_u = np.exp(-np.outer(u**2 * kappa, t))   
    exp_v = np.exp(-np.outer(v**2 * kappa, t))   
    exp_zeta = np.exp(-np.outer(zeta**2 * kappa, t))
    return exp_u, exp_v, exp_zeta


def T_field(r, z, t):
    """
    Temperature rise at (r, z) for time(s) t.
    r, x may be scalars or 1-D arrays; t may be a scalar or 1-D array.
    Returns array of shape (Nx, Nr, Nt) squeezed of length-1 dimensions.

    Algorithm
    ---------
    Exploits the separability  alpha[m,p] = (u_m^2 + z_p^2)*kappa  to write

        exp(-alpha[m,p]*t) = exp_u[m,t] * exp_z[p,t]

    so the double sum over (m,p) factors as two matrix multiplications:

        T(r,x,t) = steady(r,x) - Σ_m exp_u[m,t]*cos_x[m]*
                                    (Σ_p ACu[m,p]*J0_r[p]*exp_z[p,t])
                 + (sine branch)
    """
    r = np.atleast_1d(np.asarray(r, float))
    x = np.atleast_1d(np.asarray(z, float))
    t = np.atleast_1d(np.asarray(t, float))

    J0_r  = j0(np.outer(zeta, r) / a)   
    cos_z = np.cos(np.outer(u, z) / a)    
    sin_z = np.sin(np.outer(v, z) / a)    

    exp_u, exp_v, exp_zeta = _exp_factors(t) 

    steady_cos = (cos_z.T @ ACu) @ J0_r    
    steady_sin = (sin_z.T @ BCv) @ J0_r

    J0_expzeta = J0_r[:, :, None] * exp_zeta[:, None, :]

    H_cos = np.tensordot(ACu, J0_expzeta, axes=([1], [0]))

    cos_exp_u = cos_z * exp_u[:, None, :]             
    decay_cos = np.tensordot(cos_exp_u, H_cos,
                             axes=([0, 2], [0, 2])).T    

    decay_cos = np.einsum('mx,mt,mrt->xrt', cos_z, exp_u, H_cos)

    H_sin = np.tensordot(BCv, J0_expzeta, axes=([1], [0]))
    decay_sin = np.einsum('mx,mt,mrt->xrt', sin_z, exp_v, H_sin)

    result = (steady_cos + steady_sin)[:, :, None] - (decay_cos + decay_sin)
    return result.squeeze()


def psi_mod(t, r):

    r = np.atleast_1d(np.asarray(r, float))
    t = np.atleast_1d(np.asarray(t, float))

    J0_r  = j0(np.outer(zeta, r) / a)          
    exp_u, _, exp_zeta = _exp_factors(t)        

    spatial = (np.sin(u * h / (2*a)) * a / u)   
    Wmp = ACu * spatial[:, None]                  

    steady = Wmp.sum(axis=0) @ J0_r


    J0_expzeta = J0_r[:, :, None] * exp_zeta[:, None, :]       
    H = np.tensordot(Wmp, J0_expzeta, axes=([1], [0]))       
    decay = np.einsum('mt,mrt->rt', exp_u, H)             

    result = 2 * dndT * (steady[:, None] - decay)  
    return result.squeeze()


# --- Plot 1: T(t) ---
t_vals = np.logspace(0, 6, 1000)
T_vals_1 = T_field(0.0,  0.0, t_vals)
T_vals_2 = T_field(a, 0.0, t_vals)
plt.plot(t_vals, T_vals_1, label=r'Center')
plt.plot(t_vals, T_vals_2, label=r'Edge')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.title(r'Change in temperature $T(t)-T_{ext}$ under the ring heater')
plt.xscale('log')
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7, color='gray')
plt.gca().set_axisbelow(True)
plt.show()

# --- Plot 2: ψ(r) ---
r_vals = np.linspace(0, a, 500)
plt.figure()
for t_fixed in [1e1, 1e2, 1e3]:
    psi_vals = psi_mod(t_fixed, r_vals)
    psi_vals = psi_vals - psi_vals[0]
    exp = int(np.log10(t_fixed))
    plt.plot(r_vals, psi_vals * 1e6, label=rf'$t = 10^{{{exp}}}$ s')
plt.xlabel('r (m)')
plt.ylabel(r'$z$ (µm)')
plt.title(r'Aberration relative to the center $\psi(r)-\psi(0)$ at fixed times')
plt.legend()
plt.tick_params(axis='both', direction='in', top=True, right=True)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray')
plt.gca().set_axisbelow(True)
plt.show()


# --- RoC fit ---
def circle(x, r):
    return x**2 / (r + np.sqrt(r**2 - x**2)) + psi_mod(1e6, 0)
 
x_data = np.linspace(0, a, 50000)
y_data = psi_mod(1e6, x_data)
 

r0 = 70
popt_c, pcov_c = curve_fit(circle, x_data, y_data, p0=[r0], maxfev=100000)
r_fit, = popt_c
print(f"r = {r_fit:.4f}")
 
 

x_fit = np.linspace(x_data[0], x_data[-1], 1000)

fig, ax = plt.subplots(figsize=(8, 5))


ax.plot(x_data, y_data * 1e6, color='steelblue', lw=2.5, zorder=2,
        label=r'Theoretical aberration')


ax.plot(x_fit, circle(x_fit, *popt_c) * 1e6,
        color='red', lw=2, linestyle=(0, (4, 3)),
        label=f'Circle fit  (R = {r_fit:.4f} m)')

ax.set_xlabel('r (m)')
ax.set_ylabel(r'$\psi(r)$ (µm)')
ax.set_title(r'Theoretical aberration at $t = 10^6$ s with circle fit')
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray')
ax.set_axisbelow(True)  
ax.legend()
ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

plt.tight_layout()
plt.show()

ss_res_c = np.sum((y_data - circle(x_data, *popt_c))**2)
ss_tot   = np.sum((y_data - np.mean(y_data))**2)
print(f"R² Circle:    {1 - ss_res_c/ss_tot:.6f}")
