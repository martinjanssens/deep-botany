import numpy as np
import xarray as xr
import netCDF4 as nc
from scipy.optimize import curve_fit
import pandas as pd
import sys
import os
from profiles import *
import utils
import thermo
import f90nml

Lv = 2.5e6
cp = 1004.
Rd = 287.05
grav = 9.81

ps_fixed = 101300. # Pa. Surface pressure to use for all runs

data_path = './data'
data_path_fp = data_path+'/data_fp'
ensemble_path = './ensemble'

os.makedirs(ensemble_path, exist_ok=True)

# Grid for DALES simulations, inspired by RCEMIP grid and EUREC4A grid
zlowmax = 3e3
dzlow   = 40
nztot   = 200
r0      = 1.02
zlow1max= 15e3
r1      = 1.1
dzmax   = 500.

# Parameter ranges, manually adjusted from values determined in Hypercube-designer.ipynb
ranges = {
    'lat':      [7.5,   12.5],
    'thls':     [298.5,  301],
    'dthllt':   [0.0,    4.0],
    'hqt':      [2700,  4750],
    'u0':       [-5,      10],
    'ujet':     [0,       10],
}
Nc_default = 100e6 # cloud drops per m3

sweeps = ranges.copy()
sweeps['Nc'] =  [20e6,    1000e6] # per m3

# for the top of the domain, relax profiles back to the reference state using ERA5, which is
# - the southern domain mean upper troposphere for q/theta_l
# - zero for u and v
href_relax=9e3 # Relax to same profile in the upper troposphere, which we do not control
hsca_relax=3e3
href_relax_u=19e3 # Above the tropopause, relax to zero
hsca_relax_u=7e3


## Nudging (SECOND ATTEMPT - WEAKER NUDGING)
tnudge_ft = 6 # hours
lev_max_change = 15000 # m
nudge_params = (10,4.622,5,lev_max_change,tnudge_ft*3600) # ~7 days over FT

nml_template = f90nml.read('namoptions.template')

# Nudging functions used in https://doi. org/10.1029/2023MS003796
def _nudge_atan(x, a=5, b=2, c=20, lev_max_change=5000, end=3600*6, test_plot=False):
            y = b * (np.pi/2+np.arctan(a* np.pi/2*(1-x/lev_max_change)))
            y = end + y**c
            # plot
            if test_plot:
                plt.figure(figsize=(6,9))
                plt.plot(y,x)
                plt.xlim([10e3,10e8])
                plt.ylim([0,5000])
                plt.xscale('log')
            return y

def create_nudging(zf, thl, qt, u, v, nudge_params):
    """
    Makes a nudging input file, based on
    profiles to nudge towards of
    - zf
    - thl
    - qt
    - u
    - v
    nudge_params is a tuple that contains the input parameters to Alessandro's
    arctangent nudging function.
    """

    zero = np.zeros(zf.shape)
    (a,b,c,z_max_change,tnudge_ft) = nudge_params

    # Nudging factor with height;
    # is multiplied with nudging time (tnudgefac) from namelist;
    # here we set tnudgefac=1 -> Then this is the nudging time in seconds
    nudgefac = _nudge_atan(zf,a,b,c,z_max_change,tnudge_ft)
    out_profs = np.stack((zf,nudgefac,u,v,zero,thl,qt)).T

    return out_profs


def create_backrad(data_path, out_dir_rad, experiment='001'):
    """
    Create backrad.inp.001.nc
    Since upper troposphere in all simulations is from the same ERA5 profile, we can use the same profile to extrapolate
    immediately above the TOM, without creating large spurious gradients, in all Botany simulations.

    Use from RCEMIP:
    - that the TOA is at 50 Pa
    - the function for o3(pres)
    - the humidity
    """

    backrad_rce = xr.open_dataset(data_path+'/backrad_rce.nc')
    #era5_ref = xr.open_dataset(data_path+'/era5_ref.nc') # now computed above
    brl = era5_ref.isel(zm=slice(None, None, -1)).swap_dims({'zm':'level'})
    brl['level'] = brl['level']*100

    ## pressure (from ERA5, extrapolated to 50 Pa)
    pres_br = np.hstack([brl['level'].to_numpy(), [50.]])

    ## temperature
    # Extrapolate above ERA5 to 50 Pa level using the international standard atmosphere (https://www.digitaldutch.com/atmoscalc/)
    # offset to fit at the highest ERA5 level
    Tf = brl['t'].isel(level=-1)
    dT_ISA = Tf - 270.650 # ISA offset
    T50 = 256.488 # Offset ISA T at 50 Pa

    T_br = np.hstack([brl['t'].to_numpy(), [T50]])

    ## q
    # Extrapolate by assuming it stays constant at the final ERA5 value
    q_br = np.hstack([brl['q'], [brl['q'][-1]]])

    ## o3
    # Reconstruct RCEMIP profile (from Wing et al., 2018 - https://doi.org/10.5194/gmd-11-793-2018)
    def o3_wing(p, g1, g2, g3):
        p = p/100 # Pa -> hPa
        return g1*p**g2*np.exp(-p/g3)/1e6

    g1 = 3.6478
    g2 = 0.83209
    g3 = 11.3513
    o3_rcemip = o3_wing(backrad_rce['lev'], g1, g2, g3)

    # Use the same function to fit our o3 to TOA
    [g1b, g2b, g3b], pco3 = curve_fit(o3_wing, brl['level'], brl['o3'], p0=[g1*1.6,g2,g3])
    o3_br = o3_wing(pres_br, g1b, g2b, g3b)

    # Create backrad
    backrad_out = os.path.join(out_dir_rad,'backrad.inp.'+experiment+'.nc')

    nc_file = nc.Dataset(backrad_out, 'w')
    nc_file.title = 'Background radiation input for deep-botany simulations, from ERA5'

    dims = nc_file.createDimension('lev', pres_br.size)

    p_var = nc_file.createVariable('lev', 'f4', ('lev'))
    T_var = nc_file.createVariable('T',   'f4', ('lev'))
    q_var = nc_file.createVariable('q',   'f4', ('lev'))
    o_var = nc_file.createVariable('o3',  'f4', ('lev'))

    p_var.units = 'Pa'
    T_var.units = 'K'
    q_var.units = 'kg/kg'
    o_var.units = '-'

    p_var[:] = pres_br
    T_var[:] = T_br
    q_var[:] = q_br
    o_var[:] = o3_br

    nc_file.close()



# Set the grid
zh, zf, = utils.make_grid(zlowmax, dzlow, nztot, r0, zlow1max, r1, dzmax=dzmax)

# Manually retrieved ERA5 from Copernicus (used for higher levels, which do not exist in HERA5 from DKRZ)
era5_allplev = xr.open_dataset(data_path+'/era5_month_allplev_s.nc').sel(expver=1)
era5_allplev['zm'] = (era5_allplev['z']/grav).mean(['time','latitude','longitude'])
era5_allplev = era5_allplev.set_coords(['zm']).swap_dims({'level':'zm'})
era5_allplev['pres'] = era5_allplev['level'] * 100
era5_allplev['theta_l']  = (1e5/era5_allplev['pres'])**(2/7)*(era5_allplev['t'])
era5_allplev_mn = era5_allplev.mean(['time','latitude','longitude'])
era5_ref = era5_allplev_mn.copy(deep=True)

# Interpolate pressure to model grid for RH calculations
pres = (era5_allplev_mn['level']*100)
pres_np = np.flip(np.append(pres.to_numpy(),ps_fixed))
zm_np = np.flip(np.append(pres['zm'].to_numpy(), 0.))
pres = xr.DataArray(data=pres_np, coords={'zm':zm_np})
pres_zf = pres.interp(zm=zf,kwargs={"fill_value": "extrapolate"})


create_backrad(data_path, ensemble_path)


def compute_profiles(pars):
    thl = linml_sl(zf, pars['thls'], pars['dthllt'])
    qt = exp_h_lt(zf, pars['hqt'])
    u = linv_aej_fs(zf, pars['u0'], pars['ujet'])
    thl, qt, u = relax_all(zf, thl, qt, u, era5_ref, href_relax, hsca_relax, href_relax_u, hsca_relax_u)
    tke = 1 - zf/3000; tke[zf>=3000] = 0.
    return thl, qt, u, tke

def setup_run(ind, pars, experiment='001'):
    run_dir = ensemble_path + f'/run_{ind}'
    os.makedirs(run_dir, exist_ok=True)

    zero = np.zeros(zf.shape)

    thl, qt, u, tke = compute_profiles(pars)
    v = np.zeros_like(u)
    prof = np.stack((zf,thl.to_numpy(),qt.to_numpy(),u.to_numpy(),zero,tke)).T
    profile_out = os.path.join(run_dir, 'prof.inp.'+experiment)
    np.savetxt(profile_out, prof, fmt='%12.6g',
               header='\n    height         thl          qt            u            v          TKE')

    # lscale.inp - no large-scale forcing other than nudging
    lscale = np.stack((zf,u,v,zero,zero,zero,zero,zero)).T
    lscale_out = os.path.join(run_dir, 'lscale.inp.'+experiment)
    np.savetxt(lscale_out, lscale, fmt='%12.6g',
               header='\n    height           ug           vg         wfls      dqtdxls      dqtdyls      dqtdtls      dthlrad')

    # nudge.inp
    nudge_profs = create_nudging(zf, thl, qt, u, v, nudge_params)

    nudge_out = os.path.join(run_dir, 'nudge.inp.'+experiment)
    #f = open(nudge_out, 'w')
    #f.close()

    # Append two time instances - one at start, one after end of simulation
    with open(nudge_out, 'wb') as f:
        np.savetxt(f, nudge_profs, fmt='%+10.10e', comments='',
                   header='\n      z (m)          factor (-)         u (m s-1)         v (m s-1)         w (m s-1)          thl (K)        qt (kg kg-1)    \n# 0.00000000E+00')
        np.savetxt(f, nudge_profs, fmt='%+10.10e', comments='',
                   header='\n      z (m)          factor (-)         u (m s-1)         v (m s-1)         w (m s-1)          thl (K)        qt (kg kg-1)    \n# 1.00000000E+07')


    nml = nml_template.copy()
    nml['PHYSICS']['thls'] = pars['thls']
    nml['PHYSICS']['ps'] = ps_fixed
    nml['NAMMICROPHYSICS']['Nc_0'] = pars['Nc']
    nml['DOMAIN']['xlat'] = pars['lat']
    nml['DEEPBOTANY'] = dict(pars)   # save a copy of the parameter dictionary in the namelist as an extra section
    nml['DEEPBOTANY']['index'] = ind  # ...and add the index

    nml.write(os.path.join(run_dir, 'namoptions.'+experiment), force=True)



ensemble = []

# cube center - note also intermediate latitude
center = {v : (ranges[v][0] + ranges[v][1])/2 for v in ranges.keys()}
center['Nc'] = Nc_default
ensemble.append(center)

center_s = {
            'thls':   299.926558,
            'dthllt':   0.496936,
            'hqt':    4059.26059,
            'u0':       3.414611,
            'ujet':     2.896884,
            'Nc' :    Nc_default,
}
ensemble.append(center_s)

center_n = {
            'thls':   299.511256,
            'dthllt':   2.354826,
            'hqt':   3598.007748,
            'u0':       1.796379,
            'ujet':      6.52969,
            'Nc' :    Nc_default,
}
ensemble.append(center_n)

# add corners
for lat in ranges['lat']:
    for thls in ranges['thls']:
        for dthllt in ranges['dthllt']:
            for hqt in ranges['hqt']:
                for u0 in ranges['u0']:
                    for ujet in ranges['ujet']:
                        pars = {'lat' : lat,
                                'thls' : thls,
                                'dthllt' : dthllt,
                                'hqt' : hqt,
                                'u0': u0,
                                'ujet' : ujet,
                                'Nc' : Nc_default,
                                }
                        ensemble.append(pars)


# Add sweeps. For now only two points for each variable - min and max from the ranges.
# lat is handled separately: each sweep point is added for the N and S domain
for c in center_s, center_n: # for every sweep point, do it for N and S domains
    for var in (sweeps.keys()):
        if var != 'lat': # don't sweep latitude here
            for val in sweeps[var]:
                m = c.copy()
                m[var] = val
                ensemble.append(m)

df = pd.DataFrame(ensemble)

# if some sweeps added duplicate points, remove them now
df.drop_duplicates(inplace=True)

#print(df.to_string())
df.to_csv(ensemble_path+'/parameters.csv')

#generate profiles
for ind,m in df.iterrows():
    setup_run(ind, m)
