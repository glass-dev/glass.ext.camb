# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''GLASS module for CAMB interoperability'''

__version__ = '2022.10.11'


import logging
from collections import deque
import numpy as np

import camb

from glass.generator import generator
from glass.matter import WZ, CL

logger = logging.getLogger(__name__)


@generator(receives=WZ, yields=CL)
def camb_matter_cl(pars, lmax, ncorr=1, *, limber=False, limber_lmin=100):
    '''generate the matter angular power spectrum using CAMB'''

    # make a copy of input parameters so we can set the things we need
    pars = pars.copy()

    # set up parameters for angular power spectra
    pars.Want_CMB = False
    pars.Want_Cls = True
    pars.min_l = 1
    pars.set_for_lmax(lmax)

    # set up parameters to only compute the intrinsic matter cls
    pars.SourceTerms.limber_windows = limber
    pars.SourceTerms.limber_phi_lmin = limber_lmin
    pars.SourceTerms.counts_density = True
    pars.SourceTerms.counts_redshift = False
    pars.SourceTerms.counts_lensing = False
    pars.SourceTerms.counts_velocity = False
    pars.SourceTerms.counts_radial = False
    pars.SourceTerms.counts_timedelay = False
    pars.SourceTerms.counts_ISW = False
    pars.SourceTerms.counts_potential = False
    pars.SourceTerms.counts_evolve = False

    logger.info('computing angular power spectra up to lmax=%d', lmax)
    logger.info('correlating %d matter shells', ncorr)
    if limber:
        logger.info('using Limber\'s approximation for l >= %d', limber_lmin)
    else:
        logger.info('not using Limber\'s approximation')

    # keep a stack of ncorr previous shells for correlation
    shells = deque([], ncorr+1)

    # initial yield
    cl = None

    while True:
        # yield computed cls and get a new redshift interval
        z, w = yield cl

        # create a new source window for shell
        s = camb.sources.SplinedSourceWindow(source_type='counts', z=z, W=w)

        # add the new shell to the front of the stack
        shells.appendleft(s)

        # pass the stack of shells to CAMB
        pars.SourceWindows = shells

        # compute the cls from the updated shells
        results = camb.get_results(pars)
        cls = results.get_source_cls_dict(lmax=lmax, raw_cl=True)

        # yield the cls for current window and all past ones, in order
        cl = [cls.get(f'W1xW{i+1}', None) for i in range(ncorr+1)]
