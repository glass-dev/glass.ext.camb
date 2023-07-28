# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''GLASS module for CAMB interoperability'''

import warnings
import numpy as np
import camb


def camb_tophat_weight(z):
    '''Weight function for tophat window functions and CAMB.

    This weight function linearly ramps up the redshift at low values,
    from :math:`w(z = 0) = 0` to :math:`w(z = 0.1) = 1`.

    '''
    return np.clip(z/0.1, None, 1.)


def matter_cls(pars, lmax, ws, *, limber=False, limber_lmin=100):
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
    for za, wa, _ in ws:
        s = camb.sources.SplinedSourceWindow(z=za, W=wa)
        sources.append(s)
    pars.SourceWindows = sources

    n = len(sources)
    cls = camb.get_results(pars).get_source_cls_dict(lmax=lmax, raw_cl=True)

    for i in range(1, n+1):
        if np.any(cls[f'W{i}xW{i}'] < 0):
            warnings.warn('negative auto-correlation in shell {i}; improve accuracy?')

    return [cls[f'W{i}xW{j}'] for i in range(1, n+1) for j in range(i, 0, -1)]


def camb_cosmo_from_config(config):
    """Load a CAMB cosmology from a GLASS configuration."""

    from cosmology import Cosmology

    if config.getbool("cosmo.nonlinear", True):
        nonlinear = camb.model.NonLinear_both
    else:
        nonlinear = camb.model.NonLinear_none

    pars = camb.set_params(NonLinear=nonlinear)
    cosmology = {}

    try:
        h = config.getfloat("cosmo.h")
    except KeyError:
        h = pars.H0/100
    else:
        cosmology["H0"] = 100*h

    try:
        Oc = config.getfloat("cosmo.Oc")
    except KeyError:
        pass
    else:
        cosmology["omch2"] = Oc*h**2

    try:
        Ob = config.getfloat("cosmo.Ob")
    except KeyError:
        pass
    else:
        cosmology["ombh2"] = Ob*h**2

    pars.set_cosmology(**cosmology)

    return Cosmology.from_camb(pars)


def camb_tophat_weight_from_config(config, cosmo):
    """Load a CAMB tophat weight from a GLASS configuration."""
    return camb_tophat_weight


def camb_cls_from_config(config, shells, cosmo):
    """Compute CAMB angular matter power spectra from a GLASS configuration."""
    if cosmo.__class__.__name__ != "CambCosmology":
        raise TypeError("CAMB spectra require CAMB cosmology")
    # boost the accuracy to get consistent lensing and galaxies
    pars = cosmo._p.copy()
    pars.Accuracy.TimeStepBoost = 5
    lmax = config.getint("fields.lmax")
    return matter_cls(pars, lmax, shells)
