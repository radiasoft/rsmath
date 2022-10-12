from rsmath import lct
from pykern import pkunit
from pykern.pkcollections import PKDict
from pykern import pksubprocess
from pykern import pkio
import numpy as np
import re


_K_RSMP = 2.0
_M_SCL = -2

def test_lct_signal():
    data_dir = pkunit.data_dir()
    work_dir = pkunit.empty_work_dir()

    def testFn1(u):
        return np.exp(- np.pi * (1. + 1.j) * u**2)

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
        uvals = lct.lct_abscissae(np, du)
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
        all_fvals.append([f for f in fvals])
        all_uvals.append([u for u in uvals])

    sigs = list(zip(dus, all_fvals))
    f = _fmt_matrix_string(str([
        [_cast_complex_for_write(d) for d in dus],
        *[[_cast_complex_for_write(u) for u in uval] for uval in all_uvals],
        *[[_cast_complex_for_write(f) for f in fval] for fval in all_fvals]
    ]))
    s = _cast_from_complex_signal(sigs, lct.resample_signal, _K_RSMP)
    c = _cast_from_complex_signal(sigs, lct.scale_signal, _M_SCL)

    for case in (
        PKDict(case="u_and_f_vals", data=f),
        PKDict(case="signals", data=s),
        PKDict(case="scaled_signals", data=c),
    ):
        _ndiff_files(
            data_dir.join(case.case + ".txt"),
            pkio.write_text(work_dir.join(case.case + "_actual.txt"), case.data),
            work_dir.join("ndiff.out"),
            data_dir
        )


def test_lct_abscissae():
    data_dir = pkunit.data_dir()
    work_dir = pkunit.empty_work_dir()
    a = _fmt_matrix_string(str([list(x) for x in [
        lct.lct_abscissae(8, 0.25),
        lct.lct_abscissae(7, 0.25),
        lct.lct_abscissae(8, 0.25, ishift = True),
        lct.lct_abscissae(7, 0.25, ishift = True),
        lct.lct_abscissae(20, 3 / (20 // 2)),
        lct.lct_abscissae(21, 3 / (21 // 2))
        ]
    ]))
    _ndiff_files(
        data_dir.join("lct_abscissae_outputs.txt"),
        pkio.write_text(work_dir.join("lct_abscissae_outputs_actual.txt"),
            a
        ),
        work_dir.join("ndiff.out"),
        data_dir
    )


def test_lct_fourier():
    pass


def test_chirp_multiply():
    pass


def test_lct_decomposition():
    pass


def test_apply_lct():
    pass


def _ndiff_files(expect_path, actual_path, diff_file, data_dir):
        pksubprocess.check_call_with_signals([
            "ndiff", actual_path, expect_path, data_dir.join("ndiff_conf.txt"),
        ], output=str(diff_file))

        d = pkio.read_text(diff_file)
        if re.search("diffs have been detected", d):
            raise AssertionError(f"{d}")


def _cast_complex_for_write(number):
    return (number.real, number.imag)


def _cast_from_complex_signal(signals, signal_function, factor):
        s = [signal_function(factor, sig) for sig in signals]
        for arr in s:
            conv = [_cast_complex_for_write(number) for number in arr[1]]
            arr[1] = conv
        return _fmt_matrix_string(str(s))


def _fmt_matrix_string(matrix_string):
    return matrix_string.replace("],", "]\n")
