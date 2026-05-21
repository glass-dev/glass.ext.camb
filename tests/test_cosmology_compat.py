import numpy as np
import camb
from cosmology.compat.camb import Cosmology

from glass.ext.camb import camb_tophat_weight, matter_cls


def test_cosmology_from_camb_results():
    h = 0.7
    Oc = 0.25
    Ob = 0.05

    pars = camb.set_params(
        H0=100 * h,
        omch2=Oc * h**2,
        ombh2=Ob * h**2,
        NonLinear=False,
    )
    results = camb.get_background(pars)
    cosmo = Cosmology(results)

    assert cosmo is not None
    # basic sanity check: H0 should be close to 70
    assert abs(cosmo.H0 - 70.0) < 1e-3


def test_matter_cls_from_cosmology():
    h = 0.7
    Oc = 0.25
    Ob = 0.05

    pars = camb.set_params(
        H0=100 * h,
        omch2=Oc * h**2,
        ombh2=Ob * h**2,
        NonLinear=False,
    )
    results = camb.get_background(pars)
    cosmo = Cosmology(results)

    lmax = 100
    za = np.linspace(0, 1, 50)
    wa = camb_tophat_weight(za)

    cls = matter_cls(cosmo.params, lmax, [(za, wa, None)])

    # one auto-correlation for one window
    assert len(cls) == 1
    # length is lmax + 1
    assert cls[0].shape == (lmax + 1,)
    # all values must be finite
    assert np.all(np.isfinite(cls[0]))
