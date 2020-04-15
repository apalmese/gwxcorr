import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import fitsio
from argparse import ArgumentParser
from scipy.stats import binned_statistic
import treecorr as tc
import os
import astropy.table
from kmeans_radec import KMeans, kmeans_sample

parser = ArgumentParser()
parser.add_argument('--input-galaxies', '-i', dest='input_galaxies', type=str, default=None,
    help='Path of the galaxy catalog file')
parser.add_argument('--input-gw-path', dest='gw_path', type=str, default='/home/s1/jsanch87/gwxcorr/out',
    help='Path to the directory containing the GW maps')
parser.add_argument('--output-file', dest='out_path', type=str, default=None,
    help='Path of the output file')
parser.add_argument('--use-full-map', dest='use_ns', action='store_true', default=False,
    help='Get both North and South catalogs')
parser.add_argument('--zbins', dest='zbins', type=list, default=[0,0.15,0.3,1.], help='Redshift bin edges')
parser.add_argument('--debug', dest='debug', action='store_true', help='Show debugging plots', default=False)
parser.add_argument('--randoms', dest='rnd_path', default=None, type=str)
parser.add_argument('--use-map', dest='use_map', action='store_true', default=False,
    help='Use Healpy maps')
parser.add_argument('--n-jk', dest='n_jk', type=int, default=50, help='Number of JK regions')
parser.add_argument('--linear', dest='linear', action='store_true', default=False, help='Use linear binning')
parser.add_argument('--theta-bins', dest='tbins',type=int, default=30, help='Number of angular bins')
args = parser.parse_args()

def make_hp_map(nside, input_data, weights=None):
    _px_nums = hp.ang2pix(nside, input_data['RA'], input_data['DEC'], lonlat=True)
    _counts = np.bincount(_px_nums, weights=weights, minlength=hp.nside2npix(nside)).astype(float)
    return _counts

def setup_tc_catalogs(input_data, map_gw, input_rnd, weights=None, use_map=False):
    nside = hp.get_nside(map_gw)
    ra ,dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    cat_gw = tc.Catalog(ra=ra, dec=dec, k=map_gw, ra_units='deg', dec_units='deg')
    if use_map:
        nside = hp.get_nside(map_gw)
        map_galaxy = make_hp_map(nside, input_data, weights=weights)
        map_rnd = make_hp_map(nside, input_rnd)
        cat_galaxy = tc.Catalog(ra=ra, dec=dec, w=map_galaxy, ra_units='deg', dec_units='deg')
        cat_rnd = tc.Catalog(ra=ra, dec=dec, w=map_rnd, ra_units='deg', dec_units='deg')
    else:
        cat_galaxy = tc.Catalog(ra=input_data['RA'], dec=input_data['DEC'], w=weights, ra_units='deg', dec_units='deg')
        cat_rnd = tc.Catalog(ra=input_rnd['RA'], dec=input_rnd['DEC'], ra_units='deg', dec_units='deg') 
    return cat_galaxy, cat_gw, cat_rnd

def compute_corr(cat_galaxy, cat_gw, cat_rnd, min_sep=0.5, max_sep=60, nbins=30, linear=False):
    if linear:
        binning='Linear'
    else:
        binning='Log'
    dd = tc.NKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='deg', bin_type=binning)
    dr = tc.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='deg', bin_type=binning)
    rd = tc.NKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='deg', bin_type=binning)
    dd2 = tc.KKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='deg', bin_type=binning)
    dd.process(cat_galaxy, cat_gw)
    dd2.process(cat_gw)
    rd.process(cat_rnd, cat_gw)
    dr.process(cat_galaxy, cat_rnd)
    xi_auto = dd2.xi
    xi, varxi = dd.calculateXi(rd)
    return np.exp(dd.meanlogr), xi, xi_auto

def get_kmeans_labels(data, njk):
    X = np.zeros((len(data['RA']), 2))
    X[:,0] = data['RA']
    X[:,1] = data['DEC']
    km = kmeans_sample(X, njk, maxiter=100, tol=1e-4)
    print('KMeans info:', km.converged, np.bincount(km.labels))
    return km

data = fitsio.read(args.input_galaxies)
if args.rnd_path is None:
    randoms = fitsio.read(args.input_galaxies.replace('galaxy','random0')) # Use the appropriate randoms
else:
    randoms = fitsio.read(args.rnd_path)
#mask = hp.read_map(args.input_galaxies.replace('galaxy', 'mask'))

if args.use_ns: # Join both North and South galaxies?
    if 'North' in args.input_galaxies:
        data = np.concatenate([data, fitsio.read(args.input_galaxies.replace('North', 'South'))])
        randoms = np.concatenate([randoms, fitsio.read(args.input_galaxies.replace('North', 'South').replace('galaxy','random0'))])
    else:
        data = np.concatenate([data, fitsio.read(args.input_galaxies.replace('South', 'North'))])
        randoms = np.concatenate([randoms, fitsio.read(args.input_galaxies.replace('South', 'North').replace('galaxy','random0'))])
weights = data['WEIGHT_SYSTOT']*(data['WEIGHT_NOZ'] + data['WEIGHT_CP'] - 1)
nside = 1024
mask = make_hp_map(nside, randoms) > 0
data_out = dict()
data_cov = dict()
for i in range(len(args.zbins)-1):
    print('Bin', i)
    zmin = args.zbins[i]
    zmax = args.zbins[i+1]
    map_gw = np.loadtxt(os.path.join(args.gw_path, f'gw_skymaps_bin{i}.dat'))
    map_gw[~mask] = 0.
    map_gw[mask] = map_gw[mask]/np.mean(map_gw[mask])-1.
    zmask = (data['Z'] >= zmin) & (data['Z'] < zmax)
    zmask_rnd = (randoms['Z'] >= zmin) & (randoms['Z'] < zmax)
    # Mask unseen regions
    if args.debug:
        hp.mollview(make_hp_map(1024, data[zmask]))
        hp.mollview(map_gw)
        plt.show()
    cat_galaxy, cat_gw, cat_rnd = setup_tc_catalogs(data[zmask], map_gw, randoms[zmask_rnd], weights=weights[zmask], use_map=args.use_map)
    theta, w, w_auto = compute_corr(cat_galaxy, cat_gw, cat_rnd, linear=args.linear, nbins=args.tbins)
    if args.debug:
        plt.figure()
        plt.loglog(theta, w)
        plt.show()
    data_out['theta'] = theta
    data_out[f'w_{i}'] = w
    data_out[f'w_auto_{i}'] = w_auto
    if args.n_jk > 0:
        _km = get_kmeans_labels(data[zmask], args.n_jk)
        labels = _km.labels
        X2 = np.zeros((np.count_nonzero(mask), 2))
        _ra, _dec = hp.pix2ang(hp.get_nside(mask), np.where(mask>0)[0], lonlat=True)
        X2[:,0] = _ra
        X2[:,1] = _dec
        X3 = np.zeros((np.count_nonzero(zmask_rnd), 2))
        X3[:,0] = randoms[zmask_rnd]['RA']
        X3[:,1] = randoms[zmask_rnd]['DEC']
        labels2 = _km.find_nearest(X2)
        labels3 = _km.find_nearest(X3)
        if args.debug:
            print('Labels', np.unique(labels))
            print('Labels 2', np.bincount(labels2))
            print('Labels 3', np.bincount(labels3))
        for i_jk in range(0, args.n_jk):
            print(f'JK: {i_jk}, {np.count_nonzero(labels!=i_jk)} galaxies left')
            map_gw_aux = np.zeros_like(map_gw)
            jk_reg_px = np.unique(hp.ang2pix(hp.get_nside(mask), data['RA'][zmask][labels==i_jk], data['DEC'][zmask][labels==i_jk], lonlat=True))
            map_gw_aux[mask>0] = map_gw[mask>0]
            map_gw_aux[jk_reg_px] = 0.
            if args.debug:
                print('Pixels matching', len(jk_reg_px))
                hp.mollview(map_gw_aux)
                #hp.mollview(mask_galaxies_aux-mask_galaxies)
                plt.show()
            cat_galaxy, cat_y, cat_rnd = setup_tc_catalogs(data[zmask][labels!=i_jk], map_gw_aux, randoms[zmask_rnd][labels3!=i_jk],
                weights=weights[zmask][labels!=i_jk], use_map=args.use_map)
            theta, w, w_auto = compute_corr(cat_galaxy, cat_gw, cat_rnd, linear=args.linear, nbins=args.tbins)
            data_cov[f'w_{i}_{i_jk}'] = w
            data_cov[f'w_{i}_{i_jk}_auto'] = w_auto
tab = astropy.table.Table(data_out)
tab.write(args.out_path, overwrite=True)
tab2 = astropy.table.Table(data_cov)
tab2.write(args.out_path.replace('.fits', '_cov.fits'), overwrite=True) 
