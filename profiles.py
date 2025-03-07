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
def linml_sl(z, u0, dult, dudz=0.005, zref=3750, zh=3250, du0=1.25, zml=500):
    u = linml(z, u0, dudz, du0=du0, zml=zml)
    usl = dult*(np.cos(np.pi*(z - zref)/(2*zh)))**2
    usl[z<zref-zh] = 0
    usl[z<zml] = 0
    usl[z>=zref+zh] = 0
    return u + usl

# Given a fixed sst, offset at the surface (du) and mixed layer height (zml), fit a moist adiabat on top
# With freedom in the mid-tropospheric stability.
# Works in absolute T, but takes thl as input
def maml_sl(z, u0, dult, ps=101300, zref=5000, zh=2500, du0=1.25, zml=650):
    # Matches what DALES does for the base state
    pres = thermo.pressure(z, ps=ps, thls=u0)
    
    u = np.zeros(z.shape)
    u[z<=zml] = u0 - du0 - thermo.grav/thermo.cp*z[z<=zml] # Positive offsets are reductions w.r.t surface; dry adiabat
    iml = np.where(z<=zml)[0][-1]
    u0 = u[iml]

    # iterate upwards for moist adiabat
    for i in range(len(z[iml:])):
        # Calculate the saturation specific humidity at the previous height's temperature and the atmosphere's pressure
        qsi = thermo.qsatur(u[iml+i-1], pres[[iml+i-1]])

        # Moist adiabatic temperature gradient at this level, assuming T=Tv
        dTdz = - (thermo.grav * (1 + qsi*thermo.rlv/(thermo.rd*u[iml+i-1])) ) / (thermo.cp + qsi*thermo.rlv**2/(thermo.rv*u[iml+i-1]**2) )

        # Linear extrapolation
        u[iml+i] = u[iml+i-1] + dTdz*(z[iml+i] - z[iml+i-1])
    
    usl = dult*(np.cos(np.pi*(z - zref)/(2*zh)))**2
    usl[z<zref-zh] = 0
    usl[z>=zref+zh] = 0
    return u*(1e5/pres)**(thermo.rd/thermo.cp) + usl

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

# RH-based control
# Given
# - RH mixed layer top (fixed)
# - RH ~7500m (mid-troposphere) => FREE
# - RH 15000m (tropopause), fixed function of RH-mid
def rh_c(z, rh7500, rh0=0.75, rh500=0.95):
    rh = np.zeros(z.size)
    
    rh15000 = np.minimum(rh7500*0.8+0.3,0.95)
    
    # From rough fits:
    # rh7500 = np.linspace(0,1,100)
    # plt.plot(rh7500, np.sqrt(rh7500*0.5)+0.3)
    # plt.plot(rh7500, np.minimum(rh7500*0.8+0.3,0.95))
    # plt.scatter([0.8,0.25,0.5],[0.95,0.5,00.75])
    # plt.grid()

    # For FT
    # Fit quadratic through these points
    c_ft = np.polyfit([500.,7500.,14500.], [rh500,rh7500,rh15000], 2)
    rh_ft = c_ft[0]*z*z + c_ft[1]*z + c_ft[2]
    rh[np.logical_and(z >= 500, z < 14500.)] = rh_ft[np.logical_and(z >= 500, z < 14500.)]

    # For mixed layer, linearly down to surface
    rh[z<500] = rh0 + (rh500-rh0)/500.*z[z<500]

    # For stratosphere, maintain the high-point RH (we will blend it to zero above the tropopause)
    rh[z>=14500] = rh[z<14500.][-1]
    return rh

def qt_from_rh(z, rh, thl, ps=101300):
    # Matches what DALES does for the base state
    p = thermo.pressure(z, ps=ps, thls=float(thl[0]+1.25))
    T = thl*(p/1e5)**(thermo.rd/thermo.cp)
    qs = thermo.qsatur(T, p)
    return rh*qs

# thl = linml_sl(zf, pars_n['thls'], pars_n['dthllt'])
# rh = rh_c(zf, 0.5)
# qt = qt_from_rh(zf, rh, thl)

# Compute rh for profile from thl and qt
def rhProf(zf, thl, qt, ps):
    pres = thermo.pressure(zf, ps=ps, thls=thl[0]+1.25)
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

    return prof_ideal - du.to_numpy()*fac

# Extend upwards with gradients from reference profile
def extend_thl(z, prof_ideal, prof_ref, z0):
    # Assumes 1D input
    if prof_ideal.size != prof_ref.size:
        prof_ref = prof_ref.interp(zm=z,kwargs={"fill_value": "extrapolate"})

    prof_ref_dz = prof_ref.differentiate('zm').rolling(zm=3).mean().to_numpy()

    iz0 = np.where(z>=z0)[0][0]
    for i in range(iz0,len(z)):
        prof_ideal[i] = prof_ideal[i-1] + prof_ref_dz[i]*(z[i] - z[i-1])
    # du = prof_ideal - prof_ref
    

    # # Subtract difference from the original profile, scaled by a function which goes
    # # from 0->1 over a height hsca centered around href
    # fac = 0.5 - 0.5*np.cos(np.pi*(z - (href-0.5*hsca))/hsca)
    # fac[z<=href-0.5*hsca] = 0
    # fac[z>=href+0.5*hsca] = 1

    return prof_ideal

def relax_all(z, thl, qt, u, era5_ref, href_relax_thl, hsca_relax_thl, href_relax_qt, hsca_relax_qt, href_relax_u, hsca_relax_u):
    thl = relax(z, thl, era5_ref['theta_l'], href_relax_thl, hsca_relax_thl)
    qt = relax(z, qt, era5_ref['q'], href_relax_qt, hsca_relax_qt)
    u = relax(z, u, era5_ref['u']*0, href_relax_u, hsca_relax_u)
    return thl, qt, u