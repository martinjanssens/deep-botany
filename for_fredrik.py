import xarray as xr
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Linear profile, no fixed shear, and African Easterly Jet model => Use for u?
def linv_aej(z, u0, dudz, ujet, href=4000, hsca=6000):
    u = u0 + dudz*z
    aej = -ujet*np.cos(np.pi*(z - href)/hsca)
    aej[z<=href-0.5*hsca] = 0
    aej[z>=href+0.5*hsca] = 0
    return u + aej

# Linear profile, no fixed shear => Use for v
def linv(z, u0, dudz):
    return u0 + dudz*z

# Linear profile with mixed layer, with mixed layer value based on surface value (u0) minus offset => Use for thl
def linml(z, u0, dudz, du0=1.25, zml=500):
    u = np.zeros(z.shape)
    u[z<=zml] = u0 - du0 # Positive offsets are reductions w.r.t surface
    u[z>zml] = u0 - du0 + (z[z>zml] - zml)*dudz
    return u

# Exponential decay with mixed layer, where u0 is set at zml => Use for qt
def exp(z, u0, u_lambda, zml=500):
    u = u0 * np.exp(-(z-zml) / u_lambda)
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

# Relaxation towards a reference profile over a certain height => Use for all profiles
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


# Example profiles
if __name__ == '__main__':

    # load era5 profiles, averaged over a mesoscale box and ~1200 summer days
    era5_env_mn = xr.open_dataset('era5_env_mn.nc')
    
    # Surface pressure and surface theta_l
    thls = era5_env_mn['sst']*(1e5/era5_env_mn['sp'])**(2./7)
    print('Mean ps:', era5_env_mn['sp'].mean().data)
    print('Mean thls:', thls.mean().data)   

    # fitting height range
    zfitmin = 0
    zfitmax = 8e3

    era5_env_mn_fit = era5_env_mn.sel(zm=slice(zfitmax,zfitmin)).astype('float64')
    zfit = era5_env_mn['zm'].sel(zm=slice(zfitmax,zfitmin)).astype('float64')

    # fit profiles
    [thls,dthldz], pcth = curve_fit(linml,
                                    zfit, 
                                    era5_env_mn_fit['theta_l'], 
                                    p0=[300,0.004])

    [qt0,hqt], pcqt = curve_fit(exp,
                                zfit,
                                era5_env_mn_fit['q'],
                                p0=[0.016,1500])

    [u0,dudz,ujet], pcu = curve_fit(linv_aej,
                                    zfit,
                                    era5_env_mn_fit['u'],
                                    p0=[-1, 0.00222,6])

    [v0,dvdz], pcv = curve_fit(linv,
                               zfit,
                               era5_env_mn_fit['v'],
                               p0=[0,0])
    
    # evaluate profiles
    zgrid = np.linspace(20., 12e3, 250) # should be higher, using 12km for plotting
    
    thl = linml(zgrid, thls, dthldz) # FIXME is that the right thls to use for the SST?
    qt = exp(zgrid, qt0, hqt)
    u = linv_aej(zgrid, u0, dudz, ujet)
    v = linv(zgrid, v0, dvdz)

    # relax them back to the reference state
    qt = relax(zgrid, qt, era5_env_mn['q'])
    thl = relax(zgrid, thl, era5_env_mn['theta_l'])
    u = relax(zgrid, u, era5_env_mn['u'])
    v = relax(zgrid, v, era5_env_mn['v'])
    
    # check that thls>thl[0]
    dthls0 = thls.mean().data-thl[0]
    print('thls - thl[0]:', dthls0) 

    cs = ['black', 'C1']

    era5_env_mn_plt = era5_env_mn.sel(zm=slice(np.max(zgrid),np.min(zgrid)))

    fig, axs = plt.subplots(ncols=4,figsize=(15,5),sharey=True)

    era5_env_mn_plt['theta_l'].plot(y='zm', ax=axs[0], color=cs[0])
    axs[0].plot(thl, zgrid, color=cs[1])
    axs[0].set_xlabel(r'$\theta_l$ [K]')

    era5_env_mn_plt['q'].plot(y='zm', ax=axs[1], color=cs[0])
    axs[1].plot(qt, zgrid, color=cs[1])
    axs[1].set_xlabel(r'$q_v$ [kg/kg]')

    era5_env_mn_plt['u'].plot(y='zm', ax=axs[2], color=cs[0])
    axs[2].plot(u, zgrid, color=cs[1])
    axs[2].set_xlabel(r'$u$ [m/s]')

    era5_env_mn_plt['v'].plot(y='zm', ax=axs[3], color=cs[0], label='ERA-5')
    axs[3].plot(v, zgrid, color=cs[1], label='Fits')
    axs[3].set_xlabel(r'$v$ [m/s]')
    
    axs[3].legend(bbox_to_anchor=(1,1), loc='best')
    
    plt.savefig('prof-test.pdf',bbox_inches='tight')
    
    # TODO:
    # Settle on horizontal/vertical advection
    # Base state for radiation above the TOD   
