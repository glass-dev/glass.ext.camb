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


def matter_cls(pars, lmax, ws, *, limber=False, limber_lmin=100,
               cdm_only=False):
    '''Compute angular matter power spectra using CAMB.

    CAMB's number counts source traces cold dark matter.  By default, each
    window is given the scale-dependent bias :math:`b(k, z) = \\sqrt{P_{\\rm
    tot}(k, z)/P_{\\rm cdm}(k, z)}`, which turns the source into total matter
    mode by mode.  Set ``cdm_only=True`` to compute the cold dark matter
    spectra instead.

    '''

    if cdm_only:
        biases = [{} for _ in ws]
    else:
        biases = _total_matter_bias(pars, ws)

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
    for (za, wa, _), bias in zip(ws, biases):
        s = camb.sources.SplinedSourceWindow(z=za, W=wa, **bias)
        sources.append(s)
    pars.SourceWindows = sources

    n = len(sources)
    cls = camb.get_results(pars).get_source_cls_dict(lmax=lmax, raw_cl=True)

    for i in range(1, n+1):
        if np.any(cls[f'W{i}xW{i}'] < 0):
            warnings.warn(f'negative auto-correlation in shell {i}; improve accuracy?')

    return [cls[f'W{i}xW{j}'] for i in range(1, n+1) for j in range(i, 0, -1)]


# wavenumbers of the bias tables; CAMB sets the bias to zero outside the
# table, so the range must cover every wavenumber that CAMB requests
_BIAS_K = np.geomspace(1e-7, 1e4, 331)


def _total_matter_bias(pars, ws):
    '''Bias of total matter against cold dark matter for each window.'''

    nonlinear = pars.NonLinear != camb.model.NonLinear_none
    zmax = max(float(np.max(za)) for za, _, _ in ws)
    kw = dict(zmin=0.0, zmax=zmax, nz_step=150, kmax=12.0,
              nonlinear=nonlinear, hubble_units=False, k_hunit=False)
    pk_tot = camb.get_matter_power_interpolator(
        pars, var1='delta_tot', var2='delta_tot', **kw)
    pk_cdm = camb.get_matter_power_interpolator(
        pars, var1='delta_cdm', var2='delta_cdm', **kw)

    # constant continuation outside the range of the interpolators
    k = np.clip(_BIAS_K, 2e-4, 11.5)

    biases = []
    for za, _, _ in ws:
        b = np.sqrt(pk_tot.P(za, k)/pk_cdm.P(za, k))
        biases.append(dict(k_bias=_BIAS_K, bias_kz=np.ascontiguousarray(b.T)))
    return biases
