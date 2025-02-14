## Setup

### Anaconda

Packages required are listed in environment.yml. Use (not tested):
```
conda env create --name deep-botany -f environment.yml
```

### venv + pip

```
python -m venv deep-botany-env
. deep-botany-env/bin/activate
```

```
pip install astropy bokeh cartopy cmocean easygems healpy intake intake-xarray jinja2 matplotlib netcdf4 numcodecs numpy orcestra pyproj pyshp pytz pyyaml requests scipy tqdm tzdata urllib3 xarray zarr ipykernel seaborn f90nml
```
or

```
pip install -r requirements.txt 
```
containing the versions installed by the command above (Feb 2025).

Add the environment to Jupyter:
```
python -m ipykernel install --user --name=deep-botany-env
```


