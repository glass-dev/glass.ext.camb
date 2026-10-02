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

    CAMB's number counts source traces cold dark matter.  By default, the
    spectra are converted to total matter, with each pair rescaled by
    :math:`\\sqrt{R_i R_j}`, where :math:`R_i` is the ratio of total matter to
    cold dark matter power at the effective redshift of shell :math:`i` and
    the Limber wavenumber.  Set ``cdm_only=True`` to skip the conversion.

    '''

    pars_in = pars

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
            warnings.warn(f'negative auto-correlation in shell {i}; improve accuracy?')

    cls = [cls[f'W{i}xW{j}'] for i in range(1, n+1) for j in range(i, 0, -1)]

    if not cdm_only:
        cls = _total_matter_cls(pars_in, lmax, ws, cls)

    return cls


def _total_matter_cls(pars, lmax, ws, cls):
    '''Rescale cold dark matter shell spectra to total matter.'''

    nonlinear = pars.NonLinear != camb.model.NonLinear_none
    zmax = max(float(np.max(za)) for za, _, _ in ws)
    kw = dict(zmin=0.0, zmax=zmax, nz_step=150, kmax=12.0,
              nonlinear=nonlinear, hubble_units=False, k_hunit=False)
    pk_tot = camb.get_matter_power_interpolator(
        pars, var1='delta_tot', var2='delta_tot', **kw)
    pk_cdm = camb.get_matter_power_interpolator(
        pars, var1='delta_cdm', var2='delta_cdm', **kw)
    bg = camb.get_background(pars)

    # ratio at the effective redshift and Limber wavenumber of each shell
    ell = np.arange(lmax+1, dtype=float)
    rs = []
    for _, _, zeff in ws:
        zeff = float(zeff)
        if zeff <= 1e-3:
            rs.append(np.ones(lmax+1))
            continue
        chi = float(bg.comoving_radial_distance(zeff))
        k = np.clip((ell + 0.5)/chi, 2e-4, 11.5)
        rs.append(pk_tot.P(zeff, k, grid=False)/pk_cdm.P(zeff, k, grid=False))

    n = len(rs)
    out = []
    idx = 0
    for i in range(1, n+1):
        for j in range(i, 0, -1):
            cl = np.array(cls[idx], dtype=float)
            m = min(len(cl), lmax+1)
            cl[:m] *= np.sqrt(rs[i-1][:m]*rs[j-1][:m])
            out.append(cl)
            idx += 1
    return out
