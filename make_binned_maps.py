#Downloads and plots all the maps
import os
import numpy  as np
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import meander #pip install this to draw contours around skymap

from glob import glob
from urllib import request
from matplotlib import cm
from astropy.io import fits
from argparse import ArgumentParser
from astropy.cosmology import FlatLambdaCDM,z_at_value
import astropy.units as u 
import os

cosmo = FlatLambdaCDM(H0=70.,Om0=0.3)



def compute_contours(proportions,samples):
    """Plot containment contour around desired level. E.g 90% containment of a
    PDF on a healpix map.

    Parameters:
    -----------
    proportions: list
        list of containment level to make contours for.
        E.g [0.68,0.9]
    samples: array
        array of values read in from healpix map
        E.g samples = hp.read_map(file)

    Returns:
    --------
    theta_list: list
        List of arrays containing theta values for desired contours
    phi_list: list
        List of arrays containing phi values for desired contours
    """

    levels = []
    sorted_samples = list(reversed(list(sorted(samples))))
    nside = hp.pixelfunc.get_nside(samples)
    sample_points = np.array(hp.pix2ang(nside,np.arange(len(samples)))).T
    for proportion in proportions:
        level_index = (np.cumsum(sorted_samples) > proportion).tolist().index(True)
        level = (sorted_samples[level_index] + (sorted_samples[level_index+1] if level_index+1 < len(samples) else 0)) / 2.0
        levels.append(level)
    contours_by_level = meander.spherical_contours(sample_points, samples, levels)

    theta_list = []; phi_list=[]
    for contours in contours_by_level:
        for contour in contours:
            theta, phi = contour.T
            phi[phi<0] += 2.0*np.pi
            theta_list.append(theta)
            phi_list.append(phi)

    return theta_list, phi_list


#############



parser = ArgumentParser(description='GW Event Plotter')
parser.add_argument('--events_file', nargs=1,type=str,
                    help='LVC list of events to use', default="no_BNS_events_6Feb20.tab")
parser.add_argument('-d', '--download', default=False,
                    help='Download skymaps or use those on disk')
parser.add_argument('-m', '--maps', default="maps_no_BNS_events_6Feb20.tab",
                    help='File conatining filenames of skymaps to use')
parser.add_argument('-b', '--zbins',
                    default=[0.,0.15,0.3,1.], type=float, nargs='+',
                    help='Redshift bins edges')
args = parser.parse_args()

#########

events_file = args.events_file #"allevents.tab"
Download = args.download
maps_filenames=args.maps
zbins = args.zbins

mapdir='./skymaps'
events = np.genfromtxt(events_file,delimiter=",",dtype=object)
ids = events[:,0]


#Download files

if Download:
    fitsFiles=[]
    for event_id in ids:
        fitsFile = f'LALInference_{event_id}.fits.gz'
        print(type(fitsFile), type(mapdir))
        if not os.path.isfile(os.path.join(mapdir,fitsFile)):
            url = f'https://gracedb.ligo.org/api/superevents/{event_id}/files/LALInference.fits.gz'
            print(f'Downloading {url}')
            out, outdict = request(url, os.path.join(mapdir, fitsFile))
            try:
                test = outdict['content-disposition']
                fitsFiles.append(fitsFile)
            except:
                url = f'https://gracedb.ligo.org/api/superevents/{event_id}/files/bayestar.fits.gz'
                fitsFile = f'bayestar_{event_id}.fits.gz'
                print(f'Downloading {url}')
                out, outdict = request(url, os.path.join(mapdir, fitsFile))
                fitsFiles.append(fitsFile)
    np.savetxt(maps_filenames,fitsFiles,fmt='%s')
else:
    fitsFiles = np.genfromtxt(maps_filenames,dtype=str)


#Read files

#Assume nside of 1024 for now as a base of comparison between different map resolutions
nside_all=1024
npix = hp.nside2npix(1024)
sum_map = np.zeros(npix)

nmaps=fitsFiles.shape[0]
probs=np.zeros((npix,nmaps))
zs=np.zeros(nmaps)
for i in range(nmaps):
    fitsFile = fitsFiles[i]
    #try:
    print(mapdir, fitsFile, type(mapdir), type(fitsFile))
    hdus = fits.open(os.path.join(mapdir,fitsFile))
    header = hdus[1].header
    dist = header['DISTMEAN']
    zs[i] = z_at_value(cosmo.luminosity_distance, dist * u.Mpc, zmax=1.0)
    prob = hp.read_map(os.path.join(mapdir,fitsFile))
    nside = hp.pixelfunc.get_nside(prob)
    print("Read ",fitsFile," with NSIDE ",nside)

    if (nside==1024):
        probs[:,i] = prob
    else:
        probs[:,i] = hp.pixelfunc.ud_grade(prob, 1024)

		#sum_map=sum_map+prob
        #except:
# print "Corrupted file ", fitsFile

#Bin in z

idxs = np.digitize(zs,zbins)
nzbins = len(zbins)-1
binned_maps=np.zeros((npix,nzbins))

for k in range(nzbins):
    binned_maps[:,k] = np.sum(probs[:,(idxs==(k+1))], axis=1)

    #Now plot them

    plt.clf()

    # Choose color map and set background to white
    cmap = cm.OrRd
    cmap.set_under("w")

    # Compute GW contours.
    prob64 = hp.pixelfunc.ud_grade(binned_maps[:,k], 64) #reduce nside to make it faster
    prob64 = prob64/np.sum(prob64)
    pixels = np.arange(prob64.size)
    #sample_points = np.array(hp.pix2ang(nside,pixels)).T
    levels = [0.50, 0.90]
    theta_contour, phi_contour = compute_contours(levels, prob64)

    # Access DESI contours.
    #desi_mask = hp.read_map('desi_mask_nside{:04d}.fits'.format(nside_all))
    #sum_map[desi_mask == 0] = hp.UNSEEN

    # Plot GW skymap in Mollweide projection
    hp.mollview(binned_maps[:,k], cbar=True, unit=r'probability', min=0, max=3e-4,rot=180, cmap=cmap)
    hp.graticule(ls=':', alpha=0.5, dpar=30, dmer=45) # Set grid lines

    # Draw containment contour around GW skymap
    nregion = len(theta_contour) // len(levels)
    for i, (tc, pc) in enumerate(zip(theta_contour, phi_contour)):
        hp.projplot(tc, pc, linewidth=1, c='k')
    ax = plt.gca()

    # Label latitude lines.
    ax.text( 2.00,  0.10, r'$0^\circ$', horizontalalignment='left')
    ax.text( 1.80,  0.45, r'$30^\circ$', horizontalalignment='left')
    ax.text( 1.30,  0.80, r'$60^\circ$', horizontalalignment='left')
    ax.text( 1.83, -0.45, r'$-30^\circ$', horizontalalignment='left')
    ax.text( 1.33, -0.80, r'$-60^\circ$', horizontalalignment='left')
    ax.text(-2.00,  0.10, r'$0^\circ$', horizontalalignment='right')
    ax.text(-1.80,  0.45, r'$30^\circ$', horizontalalignment='right')
    ax.text(-1.30,  0.80, r'$60^\circ$', horizontalalignment='right')
    ax.text(-1.85, -0.45, r'$-30^\circ$', horizontalalignment='right')
    ax.text(-1.35, -0.80, r'$-60^\circ$', horizontalalignment='right')

    # Label longitude lines.
    ax.text( 2.0, -0.15, r'0$^\mathrm{h}$', horizontalalignment='center')
    ax.text( 1.5, -0.15, r'3$^\mathrm{h}$', horizontalalignment='center')
    ax.text( 1.0, -0.15, r'6$^\mathrm{h}$', horizontalalignment='center')
    ax.text( 0.5, -0.15, r'9$^\mathrm{h}$', horizontalalignment='center')
    ax.text( 0.0, -0.15, r'12$^\mathrm{h}$', horizontalalignment='center')
    ax.text(-0.5, -0.15, r'15$^\mathrm{h}$', horizontalalignment='center')
    ax.text(-1.0, -0.15, r'18$^\mathrm{h}$', horizontalalignment='center')
    ax.text(-1.5, -0.15, r'21$^\mathrm{h}$', horizontalalignment='center')
    ax.text(-2.0, -0.15, r'24$^\mathrm{h}$', horizontalalignment='center')

    plt.title(f'Bin {k}')

    hp.write_map(f'out/gw_skymaps_bin_{zbins[k]}_{zbins[k+1]}.fits',binned_maps[:,k])

    plt.savefig(f'out/gw_bin_{zbins[k]}_{zbins[k+1]}.pdf')
    #plt.show()
