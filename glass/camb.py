# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''GLASS module for CAMB interoperability'''

__version__ = '2022.8.10'


import logging
from collections import deque
import numpy as np

import camb

from glass.core import generator

logger = logging.getLogger(__name__)


@generator('zmin, zmax -> cl')
def camb_matter_cl(pars, lmax, ncorr=1, *, k_eta_fac=2.5, nonlinear=True,
                   limber=False, limber_lmin=100):
    '''generate the matter angular power spectrum using CAMB'''

    # make a copy of input parameters so we can set the things we need
    pars = pars.copy()

    # set up parameters for angular power spectra
    if nonlinear:
        pars.NonLinear = camb.model.NonLinear_both
    else:
        pars.NonLinear = camb.model.NonLinear_none
    pars.Want_CMB = False
    pars.Want_Cls = True
    pars.min_l = 1
    pars.max_l = lmax + 1
    pars.max_eta_k = lmax*k_eta_fac

    # set up parameters to only compute the intrinsic matter cls
    pars.SourceTerms.limber_window = limber
    pars.SourceTerms.limber_phi_lmin = limber_lmin
    pars.SourceTerms.counts_density = True
    pars.SourceTerms.counts_evolve = True
    pars.SourceTerms.counts_redshift = False
    pars.SourceTerms.counts_lensing = False
    pars.SourceTerms.counts_velocity = False
    pars.SourceTerms.counts_radial = False
    pars.SourceTerms.counts_timedelay = False
    pars.SourceTerms.counts_ISW = False
    pars.SourceTerms.counts_potential = False

    logger.info('compute angular power spectra up to lmax=%d', lmax)
    logger.info('correlating %d matter shells', ncorr)
    logger.info('nonlinear power spectrum: %s', nonlinear)
    logger.info('Limber\'s approximation above l=%d: %s', limber_lmin, limber)
    logger.info('computing initial background results')

    # compute the initial power spectra etc.
    bg = camb.get_background(pars, no_thermo=True)

    # keep a stack of ncorr previous source windows for correlation
    windows = deque()

    # initial yield
    cl = None

    # yield computed cls and get a new redshift interval, or stop on exit
    while True:
        try:
            zmin, zmax = yield cl
        except GeneratorExit:
            break

        # create a new source window for redshift interval
        z = np.linspace(zmin, zmax, 100)
        w = ((1 + z)*bg.angular_diameter_distance(z))**2/bg.h_of_z(z)
        w /= np.trapz(w, z)
        s = camb.sources.SplinedSourceWindow(source_type='counts', z=z, W=w)

        # keep only ncorr previous source windows on the stack
        # then add the new one to the front of the stack
        if len(windows) > ncorr:
            windows.pop()
        windows.appendleft(s)

        # pass a copy of the list of windows to CAMB
        pars.SourceWindows = list(windows)

        # compute the cls from the updated source windows
        results = camb.get_results(pars)
        all_cls = results.get_source_cls_dict(lmax=lmax, raw_cl=True)

        # yield the cls for current window and all past ones, in order
        cl = [all_cls.get(f'W1xW{i+1}', None) for i in range(ncorr+1)]
