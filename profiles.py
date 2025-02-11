import numpy as np
import xarray as xr
import thermo

# Linear profile, no fixed shear, and African Easterly Jet model
def linv_aej(z, u0, dudz, ujet, href=4000, hsca=6500):
    u = u0 + dudz*z
    aej = -ujet*(np.cos(np.pi*(z - href)/hsca))**2 # z = href - 0.5*hsca => (href - 0.5*hsca - href) / hsca = -0.5
    aej[z<=href-0.5*hsca] = 0
    aej[z>=href+0.5*hsca] = 0
    return u + aej

# Linear profile, fixed shear, and African Easterly Jet model => Use for u
def linv_aej_fs(z, u0, ujet, dudz=-0.001, href=4000, hsca=6500):
    u = u0 + dudz*z
    aej = -ujet*(np.cos(np.pi*(z - href)/hsca))**2 # z = href - 0.5*hsca => (href - 0.5*hsca - href) / hsca = -0.5
    aej[z<=href-0.5*hsca] = 0
    aej[z>=href+0.5*hsca] = 0
    return u + aej

# Linear profile, no fixed shear => Use for v
def linv(z, u0, dudz):
    return u0 + dudz*z

# Linear profile with mixed layer, with mixed layer value based on surface value (u0) minus offset => Use for thl
def linml(z, u0, dudz, du0=1.65, zml=500):
    u = np.zeros(z.shape)
    u[z<=zml] = u0 - du0 # Positive offsets are reductions w.r.t surface
    u[z>zml] = u0 - du0 + (z[z>zml] - zml)*dudz
    return u

# linml, plus lower-tropospheric deviation allowing for trade inversion-like stable layer
def linml_sl(z, u0, dult, dudz=0.005163, zref=2500, du0=1.25, zml=500):
    u = linml(z, u0, dudz, du0=du0, zml=zml)
    usl = dult*(np.cos(np.pi*(z - zref)/(2*zref)))**2
    usl[z<zml] = 0
    usl[z>=2*zref] = 0
    return u + usl

# Exponential decay with mixed layer, where u0 is set at zml => Use for qt
def exp(z, u0, u_lambda, zml=500):
    u = u0 * np.exp(-(z-zml) / u_lambda) * np.exp(-((z-zml)/10000)**2)
    if type(u0) == float:
        u[z<=zml] = u0
    elif u0.ndim < 2:
        u[z<=zml] = u0
    elif u0.ndim == 2:
        z  = z .reshape(z.size)
        u0 = u0.reshape(u0.size)
        for i in range(u0.size):
            u[z<=zml,i] = u0[i]
    else:
        print('Dimension of input u0 not supported, mixed layer not added')
    return u

# Identical, except the BL moisture is fixed
def exp_h(z, u_lambda, u0=0.0157, zml=500):
    u = u0 * np.exp(-(z-zml) / u_lambda) * np.exp(-((z-zml)/10000)**2)
    if type(u0) == float:
        u[z<=zml] = u0
    elif u0.ndim < 2:
        u[z<=zml] = u0
    elif u0.ndim == 2:
        z  = z .reshape(z.size)
        u0 = u0.reshape(u0.size)
        for i in range(u0.size):
            u[z<=zml,i] = u0[i]
    else:
        print('Dimension of input u0 not supported, mixed layer not added')
    return u

# Letting lower tropospheric moisture scale with u_lambda
def exp_h_lt(z, u_lambda, u0=0.0157, dq=-15, zref=2000, u_lambda_ref=4500, zml=500):
    u = u0 * np.exp(-(z-zml) / u_lambda) * np.exp(-((z-zml)/10000)**2)
    if type(u0) == float:
        u[z<=zml] = u0
    elif u0.ndim < 2:
        u[z<=zml] = u0
    elif u0.ndim == 2:
        z  = z .reshape(z.size)
        u0 = u0.reshape(u0.size)
        for i in range(u0.size):
            u[z<=zml,i] = u0[i]
    else:
        print('Dimension of input u0 not supported, mixed layer not added')
    dult = dq*(1/u_lambda - 1/u_lambda_ref)
    usl = dult*(np.cos(np.pi*(z - zref)/(2*zref)))**2
    usl[z<zml] = 0
    usl[z>=2*zref] = 0
    return u + usl

# Compute rh for profile
def rhProf(thl, qt, pres):
    T, ql = thermo.T_and_ql(thl, qt, pres)    
    qs = thermo.qsatur(T,pres)
    qv = qt-ql
    return 100*qv/qs

# Relaxation towards a reference profile over a certain height
def relax(z, prof_ideal, prof_ref, href=9e3, hsca=3e3):
    # Assumes 1D input
    if prof_ideal.size != prof_ref.size:
        prof_ref = prof_ref.interp(zm=z,kwargs={"fill_value": "extrapolate"})

    du = prof_ideal - prof_ref

    # Subtract difference from the original profile, scaled by a function which goes
    # from 0->1 over a height hsca centered around href
    fac = 0.5 - 0.5*np.cos(np.pi*(z - (href-0.5*hsca))/hsca)
    fac[z<=href-0.5*hsca] = 0
    fac[z>=href+0.5*hsca] = 1

    return prof_ideal - du*fac

def relax_all(z, thl, qt, u, era5_ref, href_relax, hsca_relax, href_relax_u, hsca_relax_u):
    thl = relax(z, thl, era5_ref['theta_l'], href_relax, hsca_relax)
    qt = relax(z, qt, era5_ref['q'], href_relax, hsca_relax)
    u = relax(z, u, era5_ref['u']*0, href_relax_u, hsca_relax_u)
    return thl, qt, u