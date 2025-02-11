import numpy as np
import xarray as xr

def interpolate_to_latlon(era5_env):

    nlon = np.unique(era5_env['longitude']).size
    nlat = np.unique(era5_env['latitude']).size

    lon_min = np.min(era5_env['longitude'])
    lon_max = np.max(era5_env['longitude'])

    lat_min = np.min(era5_env['latitude'])
    lat_max = np.max(era5_env['latitude'])

    lons = np.linspace(lon_min, lon_max, nlon)
    lats = np.linspace(lat_min, lat_max, nlat)

    # Reset all variables to have the dimensions lat/lon (and be zero)
    coords_ll = {'time':era5_env['time'],
                 'zm':era5_env['zm'],
                 'lon':lons,
                 'lat':lats}
    era5_env_ll = xr.Dataset(coords=coords_ll)
    varnames = list(era5_env.keys())
    for i in range(len(varnames)):
        nmi = varnames[i]
        if nmi == 'pres':
            era5_env_ll[nmi] = xr.DataArray(np.zeros([era5_env['time'].size,
                                                      era5_env['zm'].size])*np.nan,
                                            dims=('time','zm'))
        elif nmi == 'sst':
            era5_env_ll[nmi] = xr.DataArray(np.zeros([era5_env['time'].size,
                                                      lons.size,
                                                      lats.size])*np.nan,
                                            dims=('time','lon','lat'))
        else:
            era5_env_ll[nmi] = xr.DataArray(np.zeros([era5_env['time'].size,
                                                      era5_env['zm'].size,
                                                      lons.size,
                                                      lats.size])*np.nan,
                                            dims=('time', 'zm', 'lon', 'lat'))

    # Try looping over cell again and just getting what is there
    idx = era5_env['cell']
    for i in range(idx.size):
        loni = idx[i]['longitude']
        lati = idx[i]['latitude']

        # Find the closest point in the meshgrid spanned by lats / lons to this (loni, lati)
        ilon = np.argmin(np.abs(loni.data - lons))
        ilat = np.argmin(np.abs(lati.data - lats))

        for j in range(len(varnames)):
            nmj = varnames[j]
            if nmj == 'pres':
                era5_env_ll[nmj][:,:] = era5_env[nmj]
            elif nmj == 'sst':
                era5_env_ll[nmj][:,ilon,ilat] = era5_env[nmj].isel(cell=i)
            else:
                era5_env_ll[nmj][:,:,ilon,ilat] = era5_env[nmj].isel(cell=i)
    
    # Coarsen by factor 2
    era5_env_ll = era5_env_ll.coarsen({'lon':2,'lat':2},boundary='trim').mean()

    return era5_env_ll

# In this function, you have a constant dz up to zlowmax, and then you stretch by a factor r0 each level
# until you have reacher dzmax. You continue with this dz until you have used nztot points.
# You can't control the top level height.
def make_grid(zlowmax, dzlow, nztot, r0, zlow1max, r1, dzmax=200):
    zlow = np.arange(0,zlowmax+dzlow,dzlow)
    nzup = nztot - len(zlow)
    dzup = dzlow*r0**np.arange(nzup+1)
    dzup[dzup>dzmax] = dzmax
    zup = zlowmax + np.cumsum(dzup)  
    zh = np.concatenate((zlow,zup))
    
    # Above zlow1max, stretch with r1 instead
    zhlow1 = zh[zh<zlow1max]
    dzm1 = np.diff(zhlow1)[-1]
    nzup = nztot - len(zhlow1)
    dzup = dzm1*r1**np.arange(1,nzup+2)
    dzup[dzup>dzmax] = dzmax
    zup = zhlow1[-1] + np.cumsum(dzup)
    zh = np.concatenate((zhlow1,zup))
    
    zf = (zh[1:]+zh[:-1])/2
#     izmax_inp = np.where(z < zf[-1])[0][-1]

    return zh, zf, #izmax_inp

def compute_rms(prof_fit, prof_ref, var):
    return np.sqrt(((prof_ref[var] - prof_fit)**2).mean())

