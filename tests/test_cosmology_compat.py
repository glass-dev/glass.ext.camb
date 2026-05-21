import camb
from cosmology.compat.camb import Cosmology


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
