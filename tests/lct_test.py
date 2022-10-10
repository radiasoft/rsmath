from rsmath import lct_lib
from pykern import pkunit
from pykern.pkcollections import PKDict
from pykern import pkio
import numpy as np


# TODO (gurhar1133): floating point not allways deterministic
# change test structure to reflect that fact

def test_lct_abscissae_and_signal():

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

    def d_f_u_vals(rh, np, fn_idx):
        du = rh / (np // 2)
        uvals = lct_lib.lct_abscissae(np, du)
        fvals = [
            testFn1,
            testFn2,
            testFn3,
            lambda x: [ testFn4(u) for u in x ],
        ][fn_idx](uvals)
        return du, uvals, fvals

    dus = []
    all_fvals = []
    all_uvals = []
    for i, inputs in enumerate(((3., 69), (5., 50), (8., 100), (8., 100))):
        du, uvals, fvals = d_f_u_vals(*inputs, i)
        dus.append(du)
        all_fvals.append(fvals)
        all_uvals.append(uvals)

    assert dus == [0.08823529411764706, 0.2, 0.16, 0.16]

    print(f"uvals:{all_uvals}")
    print(f"fvals:{all_fvals}")
    data_dir = pkunit.data_dir()
    expect_path = data_dir.join("1.out/u_and_f_vals.txt")
    actual_path = pkio.write_text("u_and_f_vals_actual.txt", str([*all_uvals, *all_fvals]))

    from pykern import pksubprocess

    pksubprocess.check_call_with_signals([
        "ndiff", actual_path, expect_path, "ndiff_conf.txt",
    ], output="u_and_f_vals_ndiff.out")

    # for d in pkunit.case_dirs(group_prefix="1"):
    #     pkio.write_text(d.join("u_and_f_vals.txt"),
    #         str([*all_uvals, *all_fvals]
    #         )
    #     )

    assert list(lct_lib.lct_abscissae(8, 0.25)) == [-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75]
    assert list(lct_lib.lct_abscissae(7, 0.25)) == [-0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75]
    assert list(lct_lib.lct_abscissae(8, 0.25, ishift = True)) == [ 0.  ,  0.25,  0.5 ,  0.75, -1.  , -0.75, -0.5 , -0.25]
    assert list(lct_lib.lct_abscissae(7, 0.25, ishift = True)) == [ 0.  ,  0.25,  0.5 ,  0.75, -0.75, -0.5 , -0.25]
    assert [round(x, 1) for x in list(lct_lib.lct_abscissae(20, 3 / (20 // 2)))] == [-3. , -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3,  0. , 0.3,  0.6,  0.9,  1.2,  1.5,  1.8,  2.1,  2.4,  2.7]
    assert [round(x, 1) for x in list(lct_lib.lct_abscissae(21, 3 / (21 // 2)))] == [-3. , -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3,  0. , 0.3,  0.6,  0.9,  1.2,  1.5,  1.8,  2.1,  2.4,  2.7,  3. ]



    k_rsmp = 2.0
    signals = list(zip(dus, all_fvals))
    rsmps = [lct_lib.resample_signal(k_rsmp, sig) for sig in signals]

    # for d in pkunit.case_dirs(group_prefix="2"):
    #     pkio.write_text(d.join("signals.txt"), str(rsmps))
