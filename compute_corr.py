import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import fitsio
from argparse import ArgumentParser
from scipy.stats import binned_statistic
import treecorr as tc
import os
import astropy.table

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
args = parser.parse_args()

def make_hp_map(nside, input_data, weights=None):
    _px_nums = hp.ang2pix(nside, input_data['RA'], input_data['DEC'], lonlat=True)
    _counts = np.bincount(_px_nums, weights=weights, minlength=hp.nside2npix(nside)).astype(float)
    return _counts

def setup_tc_catalogs(map_galaxy, map_gw, map_rnd):
    if len(map_galaxy)!=len(map_gw):
        raise ValueError(f'Galaxy maps should have the same dimensions as GW maps {hp.get_nside(map_galaxy)}, {hp.get_nside(map_gw)}')
    else:
        nside = hp.get_nside(map_galaxy)
        ra ,dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
        cat_galaxy = tc.Catalog(ra=ra, dec=dec, w=map_galaxy, ra_units='deg', dec_units='deg')
        cat_gw = tc.Catalog(ra=ra, dec=dec, w=map_gw, ra_units='deg', dec_units='deg')
        cat_rnd = tc.Catalog(ra=ra, dec=dec, w=map_rnd, ra_units='deg', dec_units='deg')
        px_rnd2 = np.random.choice(np.where(map_rnd>0)[0], size=len(np.sum(map_rnd)))
        map_rnd2 = np.bincount(px_rnd2, minlength=hp.nside2npix(nside))
        cat_rnd2 = tc.Catalog(ra=ra, dec=dec, w=map_rnd2, ra_units='deg', dec_units='deg')
        return cat_galaxy, cat_gw, cat_rnd, cat_rnd2

def compute_corr(cat_galaxy, cat_gw, cat_rnd, cat_rnd2, min_sep=0.5, max_sep=100, nbins=100):
    dd = tc.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='deg')
    dr = tc.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='deg')
    rd = tc.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='deg')
    rr = tc.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='deg')
    dd_gal = tc.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='deg')
    dd.process(cat_galaxy, cat_gw)
    dr.process(cat_galaxy, cat_rnd)
    rd.process(cat_rnd2, cat_gw) # Probably will have to change this...
    rr.process(cat_rnd, cat_rnd2)
    xi, varxi = dd.calculateXi(rr, dr, rd)
    return np.exp(dd.meanlogr), xi

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
data_out = dict()
for i in range(len(args.zbins)-1):
    print('Bin', i)
    zmin = args.zbins[i]
    zmax = args.zbins[i+1]
    map_gw = np.loadtxt(os.path.join(args.gw_path, f'gw_skymaps_bin{i}.dat'))
    zmask = (data['Z'] >= zmin) & (data['Z'] < zmax)
    map_galaxies = make_hp_map(hp.get_nside(map_gw),  data[zmask], weights=weights[zmask])
    zmask_rnd = (randoms['Z'] >= zmin) & (randoms['Z'] < zmax)
    map_rnd = make_hp_map(hp.get_nside(map_gw), randoms[zmask_rnd], weights=np.ones(np.count_nonzero(zmask_rnd)))
    # Mask unseen regions
    map_galaxies[map_rnd==0] = 0.
    map_gw[map_rnd==0] = 0.
    if args.debug:
        hp.mollview(map_galaxies)
        hp.mollview(map_gw)
        hp.mollview(map_rnd)
        plt.show()
    cat_galaxy, cat_gw, cat_rnd = setup_tc_catalogs(map_galaxies, map_gw, map_rnd)
    theta, w = compute_corr(cat_galaxy, cat_gw, cat_rnd)
    if args.debug:
        plt.figure()
        plt.loglog(theta, w)
        plt.show()
    data_out['theta'] = theta
    data_out[f'w_{i}'] = w
tab = astropy.table.Table(data_out)
tab.write(args.out_path, overwrite=True) 
