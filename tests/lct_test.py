from rsmath import lct_lib
from pykern import pkunit
from pykern import pkio
import numpy as np


def test_lct_abscissae():

    def testFn1(u):
        return np.exp(- np.pi * (1. + 1.j) * u**2)

    def testFn1a(u):
        return np.exp(- np.pi * (1. + 1.j) * (u + 0.5)**2)

    def tri(t):
        if type(t) in (list, np.ndarray):
            t_a = np.array(t)
            oz = [ 1 if b1 and b2 else 0 for b1, b2 in zip(-1 < t_a, t_a < 1) ]
            r = np.array(oz) * (1 - np.sign(t_a) * t_a )
        else:
            r = 0
            if -1 < t and t < 1: r = 1 - np.sign(t) * t
        return r

    def testFn2(u):
        return 1.5 * tri(u / 3.) - 0.5 * tri(u)

    def testFn3(u):
        bit = [0, 1, 1, 0, 1, 0, 1, 0]
        nn = len(bit)
        idx = np.floor(4 + u / 2).astype(int)
        return [ bit[i] if 0 <= i and i < nn else 0 for i in idx ]

    def testFn4(u):
        ua = np.ndarray.flatten(np.asarray(u))
        xv = [ -6, -5, -4,  -3.5, -3, -2, -1, 0, 1, 1.5, 2.5, 2.5, 3.5, 3.5, 4,   5, 6 ]
        yv = [  0,  1,  0.5, 0.5,  1,  1,  2, 1, 1, 0.5, 0.5, 1,   1,   0.5, 0.5, 1, 0 ]
        rs = np.zeros(len(ua))
        indices = [ [ i for i,x in enumerate(xv) if x < ue ] for ue in ua ]
        indices = [ (i, ii[-1]) for i,ii in enumerate(indices) if 0 < len(ii) and len(ii) < len(xv) ]
        for idx in indices:
            x0, x1 = xv[idx[1] : idx[1] + 2]
            y0, y1 = yv[idx[1] : idx[1] + 2]
            dx = x1 - x0
            rs[idx[0]] = y0 * (x1 - ua[idx[0]]) / dx + y1 * (ua[idx[0]] - x0) / dx
        if np.isscalar(u): rs = rs[0]
        return rs

    rh1 = 3.
    np1 = 69
    du1 = rh1 / (np1 // 2)
    uvals1 = lct_lib.lct_abscissae(np1, du1)
    fvals1 = testFn1(uvals1)
    fvals1a = testFn1a(uvals1)

    rh2 = 5.
    np2 = 50
    du2 = rh2 / (np2 // 2)
    uvals2 = lct_lib.lct_abscissae(np2, du2)
    fvals2 = testFn2(uvals2)

    rh3 = 8.
    np3 = 100
    du3 = rh3 / (np3 // 2)
    uvals3 = lct_lib.lct_abscissae(np3, du3)
    fvals3 = testFn3(uvals3)

    rh4 = 8.
    np4 = 100
    du4 = rh4 / (np4 // 2)
    uvals4 = lct_lib.lct_abscissae(np4, du4)
    fvals4 = [ testFn4(u) for u in uvals4 ]

    du1, du2, du3, du4

    assert (du1, du2, du3, du4) == (0.08823529411764706, 0.2, 0.16, 0.16)

    for d in pkunit.case_dirs(group_prefix="1"):
        pkio.write_text(d.join("u_and_f_vals.txt"),
            str(
                [
                    uvals1,
                    uvals2,
                    uvals3,
                    uvals4,
                    fvals1,
                    fvals2,
                    fvals3,
                    fvals4,
                ]
            )
        )