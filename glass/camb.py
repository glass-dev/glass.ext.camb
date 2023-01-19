# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''GLASS module for CAMB interoperability'''

__version__ = '2023.1'

import warnings
import numpy as np
import camb

from glass.math import restrict_interval


def matter_cls(pars, lmax, weights, *, limber=False, limber_lmin=100):
    '''Compute angular matter power spectra using CAMB.'''

    # make a copy of input parameters so we can set the things we need
    pars = pars.copy()

    # set up parameters for angular power spectra
    pars.WantTransfer = False
    pars.WantCls = True
    pars.Want_CMB = False
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

    sources = []
    for z, w in zip(weights.z, weights.w):
        s = camb.sources.SplinedSourceWindow(z=z, W=w)
        sources.append(s)
    pars.SourceWindows = sources

    n = len(sources)
    cls = camb.get_results(pars).get_source_cls_dict(lmax=lmax, raw_cl=True)

    for i in range(1, n+1):
        if np.any(cls[f'W{i}xW{i}'] < 0):
            warnings.warn('negative auto-correlation in shell {i}; improve accuracy?')

    return [cls[f'W{i}xW{j}'] for i in range(1, n+1) for j in range(i, 0, -1)]
