# -*- coding: utf-8 -*-
u"""Tests for linear canonical transform functions
"""
from rsmath import lct
from pykern import pkunit
from pykern.pkcollections import PKDict
from pykern import pksubprocess
from pykern import pkio
import numpy as np
import re


_K_RSMP = 2.0
_M_SCL = -2
_Q_CM = 0.667
_EXAMPLE_MATRICES = [
    [[0.5, -0.5], [0.5, 1.5]],
    [[2.0, 1.0], [-2.6, -0.8]],
    [[-0.5, 0.5], [-0.5, -1.5]],
]


class _MultiCases:
    def __init__(self):

        def _sigs():
            return list(zip(self._vals.dus, self._vals.all_fvals))

        def _vals():
            dus = []
            all_fvals = []
            all_uvals = []
            for i, inputs in enumerate(((3.0, 69), (5.0, 50), (8.0, 100), (8.0, 100))):
                du, uvals, fvals = d_f_u_vals(*inputs, i)
                dus.append(du)
                all_fvals.append([f for f in fvals])
                all_uvals.append([u for u in uvals])
            return PKDict(
                dus=dus,
                all_fvals=all_fvals,
                all_uvals=all_uvals,
            )

        self._vals = _vals()
        self._sigs = _sigs()
        for d in pkunit.case_dirs():
            pkio.write_text(
                d.join(f"{d.basename}.ndiff"),
                getattr(self, f"_case_{d.basename}")(),
            )

    def _case_abscissae(self):
        return _lct_abscissae()

    def _case_chirp(self):
        return _cast_from_complex_signal(self._sigs, lct.chirp_multiply, _Q_CM)

    def _case_decomp(self):
        return str([lct.lct_decomposition(m) for m in _EXAMPLE_MATRICES])

    def _case_fourier(self):
        return _cast_from_complex_signal(self._sigs, lct.lct_fourier, None)

    def _case_lct(self):
        return _apply_lct()

    def _case_scaled_signals(self):
        return _cast_from_complex_signal(self._sigs, lct.scale_signal, _M_SCL)

    def _case_signals(self):
        return _cast_from_complex_signal(self._sigs, lct.resample_signal, _K_RSMP)

    def _case_u_f(self):
        return _f_data(
            self._vals.dus,
            self._vals.all_fvals,
            self._vals.all_uvals
        )


def test_multi():
    _MultiCases()


def test_apply_2d_sep():
    data_dir = pkunit.data_dir()
    for case_number in (0, 1):
        i = _case(case_number, data_dir)
        r = lct.apply_lct_2d_sep(i.mx, i.my, i.signal_in)
        pkunit.file_eq(
            expect_path=data_dir.join(f"2d_sep_expect_out{case_number}.ndiff"),
            actual=str(
                [r[0], r[1], [_cast_complex_for_write(number) for number in r[2]]]
            ),
        )


def _apply_lct():
    np1 = 64
    du1 = 3.0 / (np1 // 2)
    np3 = 384
    du3 = 15.0 / (np3 // 2)
    signal1 = [du1, _fn1(lct.lct_abscissae(np1, du1))]
    signal1a = [du1, _fn1a(lct.lct_abscissae(np1, du1))]
    signal3 = [du3, _fn3(lct.lct_abscissae(np3, du3))]
    return str(
        _convert_signal_data(
            [
                lct.apply_lct(_EXAMPLE_MATRICES[0], signal1),
                lct.apply_lct(_EXAMPLE_MATRICES[0], signal1a),
                lct.apply_lct(_EXAMPLE_MATRICES[1], signal1),
                lct.apply_lct(_EXAMPLE_MATRICES[1], signal1a),
                lct.apply_lct(_EXAMPLE_MATRICES[1], signal3),
            ]
        )
    )


def _f_data(dus, all_fvals, all_uvals):
    return _fmt_matrix_string(
        str(
            [
                [_cast_complex_for_write(d) for d in dus],
                *[[_cast_complex_for_write(u) for u in uval] for uval in all_uvals],
                *[[_cast_complex_for_write(f) for f in fval] for fval in all_fvals],
            ]
        )
    )


def _lct_abscissae():
    return _fmt_matrix_string(
        str(
            [
                list(x)
                for x in [
                    lct.lct_abscissae(8, 0.25),
                    lct.lct_abscissae(7, 0.25),
                    lct.lct_abscissae(8, 0.25, ishift=True),
                    lct.lct_abscissae(7, 0.25, ishift=True),
                    lct.lct_abscissae(20, 3 / (20 // 2)),
                    lct.lct_abscissae(21, 3 / (21 // 2)),
                ]
            ]
        )
    )


def _case(case_number, data_dir):
    from pykern import pkjson

    i = pkjson.load_any(data_dir.join(f"json_inputs{case_number}.json"))
    if case_number == 1:
        for idx, arr in enumerate(i.signal_in[2]):
            new_arr = [complex(s) for s in arr]
            i.signal_in[2][idx] = new_arr
    return i


def _cast_complex_for_write(number):
    return (number.real, number.imag)


def _cast_from_complex_signal(signals, signal_function, factor):
    if factor:
        return _convert_signal_data([signal_function(factor, sig) for sig in signals])
    return _convert_signal_data([signal_function(sig) for sig in signals])


def _convert_signal_data(signals):
    for arr in signals:
        conv = [_cast_complex_for_write(number) for number in arr[1]]
        arr[1] = conv
    return _fmt_matrix_string(str(signals))


def _fmt_matrix_string(matrix_string):
    return matrix_string.replace("],", "]\n")


def _fn1(u):
    return np.exp(-np.pi * (1.0 + 1.0j) * u**2)


def _fn1a(u):
    return np.exp(-np.pi * (1.0 + 1.0j) * (u + 0.5) ** 2)


def _tri(t):
    if type(t) in (list, np.ndarray):
        t_a = np.array(t)
        oz = [1 if b1 and b2 else 0 for b1, b2 in zip(-1 < t_a, t_a < 1)]
        r = np.array(oz) * (1 - np.sign(t_a) * t_a)
    else:
        r = 0
        if -1 < t and t < 1:
            r = 1 - np.sign(t) * t
    return r


def _fn2(u):
    return 1.5 * _tri(u / 3.0) - 0.5 * _tri(u)


def _fn3(u):
    bit = [0, 1, 1, 0, 1, 0, 1, 0]
    nn = len(bit)
    idx = np.floor(4 + u / 2).astype(int)
    return [bit[i] if 0 <= i and i < nn else 0 for i in idx]


def _fn4(u):
    ua = np.ndarray.flatten(np.asarray(u))
    xv = [-6, -5, -4, -3.5, -3, -2, -1, 0, 1, 1.5, 2.5, 2.5, 3.5, 3.5, 4, 5, 6]
    yv = [0, 1, 0.5, 0.5, 1, 1, 2, 1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5, 1, 0]
    rs = np.zeros(len(ua))
    indices = [[i for i, x in enumerate(xv) if x < ue] for ue in ua]
    indices = [
        (i, ii[-1]) for i, ii in enumerate(indices) if 0 < len(ii) and len(ii) < len(xv)
    ]
    for idx in indices:
        x0, x1 = xv[idx[1] : idx[1] + 2]
        y0, y1 = yv[idx[1] : idx[1] + 2]
        dx = x1 - x0
        rs[idx[0]] = y0 * (x1 - ua[idx[0]]) / dx + y1 * (ua[idx[0]] - x0) / dx
    if np.isscalar(u):
        rs = rs[0]
    return rs


def d_f_u_vals(rh, np, fn_idx):
    du = rh / (np // 2)
    uvals = lct.lct_abscissae(np, du)
    fvals = [_fn1, _fn2, _fn3, lambda x: [_fn4(u) for u in x],][
        fn_idx
    ](uvals)
    return du, uvals, fvals
