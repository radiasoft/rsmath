import numpy as np

gr = (1 + np.sqrt(5)) / 2
d2r = np.pi / 180.
r2d = 1 / d2r


def minmax(arr):
    """
    Compute and return the min and max of a given array.
    """
    return np.min(arr), np.max(arr)

def even_ceil(x):
    """
    Return smallest even integer greater than or equal to x.
    """
    ec = int(np.ceil(x))
    if ec % 2 != 0: ec += 1
    return ec

def odd_ceil(x):
    """
    Return smallest odd integer greater than or equal to x.
    """
    oc = int(np.ceil(x))
    if oc % 2 == 0: oc += 1
    return oc

def round_to_even(x):
    """
    Return even integer closest to x.
    """
    return 2 * round(x / 2)

def convert_params_3to4(alpha, beta, gamma):
    """
    Given LCT parameters (α,β,γ), return the associated 2x2 ABCD matrix.

    Caveats: Not all authors use the same convention for the parameters
    (α,β,γ): some reverse the rôles of α and γ.
    We follow the notation of [Koç [7]](#ref:Koc-2011-thesis)
    and [Koç, et al. [8]](#ref:Koc-2008-DCLCT),
    also [Healy [4], ch.10](#ref:Healy:2016:LCTs).

    Restrictions: The parameter beta may not take the value zero.

    Arguments:
    α, β, γ -- a parameter triplet defining a 1D LCT

    Return a symplectic 2x2 ABCD martrix.
    """
    if beta == 0.:
        print("The parameter beta may not take the value zero!")
        return -1
    M = np.zeros((2,2))
    M[0,0] = gamma / beta
    M[0,1] = 1. / beta
    M[1,0] = -beta + alpha * gamma / beta
    M[1,1] = alpha / beta

    return M

def convert_params_4to3(M_lct):
    """
    Given a symplectic 2x2 ABCD matrix, return the associated parameter triplet (α,β,γ).

    Caveats: Not all authors use the same convention for the parameters
    (α,β,γ): some reverse the rôles of α and γ.
    We follow the notation of [Koç [7]](#ref:Koc-2011-thesis)
    and [Koç, et al. [8]](#ref:Koc-2008-DCLCT),
    also [Healy [4], ch.10](#ref:Healy:2016:LCTs).

    Restrictions: The (1,2) matrix entry may not take the value zero.

 ** DTA: We need to decide how best to handle b very near zero.

    Arguments:
    M_lct -- a symplectic 2x2 ABCD martrix that defines a 1D LCT

    Return the parameter triplet α, β, γ.
    """
    a, b, c, d = np.asarray(M_lct).flatten()
    if b == 0.:
        print("The (1,2) matrix entry may not take the value zero!")
        return -1
    beta = 1 / b
    alpha = d / b
    gamma = a / b

    return alpha, beta, gamma

def lct_abscissae(nn, du, ishift = False):
    """
    Return abscissae for a signal of length nn with spacing du.

    With the default setting `ishift = False`, the abscissae will
    be either centered on the origin (if an odd number of points),
    or one step off of that (if an even number of points).
    If one sets `ishift = True`, then the computed array of abscissae
    will be shifted, or “rolled”, to place the origin at the beginning.

    Arguments:
    nn -- length of signal for which to construct abscissae,
            i.e. number of samples
    du -- sample spacing
    ishift -- a boolean argument: if True, the array of abcissae will
                be rotated so as to place the origin in entry [0]
    """
    u_vals = du * (np.arange(0, nn) - nn // 2)
    # if ishift == True: u_vals = np.roll(u_vals, (nn + 1) // 2)
    if ishift == True: u_vals = np.fft.ifftshift(u_vals)
    return u_vals

def resample_signal(k, in_signal, debug = False):
    """
    Resample the input signal, and return the resultant signal.

    This function takes an input signal
        U = [dt, [u0, u1, ..., u_{n-1}]]
    and resamples it by a factor of k, returning the new signal
        SMPL(k){U} = [dt', [u0, u1, ..., u_{n'-1}]],
    where n' is _roughly_ k * n. The “roughly” has to do with the
    fact that k need not be an integer.

    This function requires interpolating the data to a new sample
    interval: The initial range is Δt = n * dt, with sample points
    at the centers of n equal-size intervals, and the function is
    taken to have period Δt. The points for the resampled signal
    will occupy the _same_ range Δt.

 ** DTA: This function currently uses 1D _linear_ interpolation.
         We should upgrade this to use a more sophisticated
         interpolation scheme, such as those available in SciPy
         or that described by Pachón, et al.

    Arguments:
        k -- factor by which to resample the data
        in_signal -- [dt, [u_0, ..., u_{n-1}]], where dt denotes
                     the sample interval of the input signal [u]

    Return the resampled signal.
    """
    dt, signal_u = in_signal
    n = len(signal_u)

    # number of samples and sample spacing for resampled signal
    n_bar = int(np.ceil(k * n))
    dt_bar = dt * n / n_bar

    # abscissae
    t_vals = lct_abscissae(n,     dt    )
    t_bar  = lct_abscissae(n_bar, dt_bar)

    # interpolate signal at the new abscissae
    p = n * abs(dt)
    u_bar = np.interp(t_bar, t_vals, signal_u, period = p)

    if debug:
        print("n    = ", n, "   n_bar = ", n_bar)
        print("dt   =", dt, "  dt_bar =", dt_bar, "\n")
        print("t_in =", t_vals, "\n")
        print("t_up =", t_bar)

    return [dt_bar, u_bar]


def scale_signal(m, in_signal):
    """
    Scale the input signal, and return the result.

    This function implements the LCT version of 1D Scaling (SCL),
    with parameter m acting on a given input signal:
        SCL(m): LCT[m 0, 0 1/m]{f}(m.x) <- f(x) / sqrt(|m|).
    The input data must have the form
        [dX, [f_0, ..., f_{N-1}]],
    where dX denotes the sample interval of the incoming signal.

    Arguments:
    m -- scale factor
    in_signal -- the signal to transform, [dX, signal_array],
                 where dX denotes the incoming sample interval

    Return the scaled signal.
    """
    # NB: dX . ΔY = ΔX . dY = 1
    # and Ns = 1 + ΔX / dX = 1 + ΔY / dY
    dX, signalX = in_signal

    dY = abs(m) * dX
    signalY = np.copy(signalX) / np.sqrt(abs(m))
    if m < 0: # reverse signal
        ii = 0 + 1j  # “double-struck” i as unit imaginary
        signalY = ii * signalY[::-1]
        if len(signalY) % 2 == 0:
            # move what was the most negative frequency component,
            # now the most positive, back to the beginning
            signalY = np.roll(signalY, 1)

    return [dY, signalY]


def lct_fourier(in_signal):
    """
    Fourier transform the input signal, and return the result.

    This function implements the LCT version of a 1D Fourier transform (FT),
        FT(): LCT[0 1, -1 0]{f}(y) <- e^{-ii φ} FT(f),
    using numpy’s FFT to do the heavy lifting. As indicated here, the LCT
    version differs by an overall phase.

 ** DTA: KB Wolf remarks that correctly identifying the phase is a delicate
         matter. In light of that, we need to verify the phase used here.

    Argument:
    in_signal -- the signal to transform, [dX, signal_array], where dX
                 denotes the incoming sample interval, and we assume the
                 signal array is assumed symmetric (in the DFT sense)
                 about the origin

    Return the transformed signal in the form [dY, e^{-ii φ} FFT(signal)].
    """
    # NB: dX . ΔY = ΔX . dY = 1
    dX, signalX = in_signal
    Npts = len(signalX)
    dY = 1 / (Npts * dX)

    ii = 0 + 1j  # “double-struck” i as unit imaginary
    lct_coeff = np.exp(-ii * np.pi / 4) # DTA: KB Wolf says this requires care(!).

    # convert to frequency domain
    signalX = np.fft.ifftshift(signalX)
    signalY = dX * np.fft.fft(signalX)
    signalY = np.fft.fftshift(signalY)

    return [dY, lct_coeff * signalY]


def chirp_multiply(q, in_signal):
    """
    Transform the input signal by chirp multiplication with parameter q.

    This function implements the LCT version of chirp multiplication (CM)
    with parameter q acting on a given input signal:
        CM(q): LCT[1 0, q 1]{f}(x) <- e^{-ii π q x^2}f(x).
    The input data must have the form
        [dX, [f_0, ..., f_{N-1}]],
    where dX denotes the sample interval of the incoming signal.

    Arguments:
    q -- CM factor
    in_signal -- the signal to transform, [dX, signal_array],
                 where dX denotes the incoming sample interval

    Return the transformed signal.
    """
    # NB: dX . ΔY = ΔX . dY = 1
    dX, signalX = in_signal

    ii = 0 + 1j  # “double-struck” i as unit imaginary
    ptsX2 = lct_abscissae(len(signalX), dX) ** 2

    return [ dX, np.exp(-ii * np.pi * q * ptsX2) * signalX]


def lct_decomposition(M_lct):
    """
    Given an LCT matrix, M_lct, return a decomposition into simpler matrices.

    Any symplectic 2x2 ABCD matrix that defines a linear canonical transform
    (LCT) may be decomposed as a product of matrices that each correspond to
    simpler transforms for which fast [i.e., ~O(N log N)] algorithms exist.
    The transforms required here are scaling (SCL), chirp multiplication (CM),
    and the Fourier transform (FT). In addition, we must sometimes resample
    the data (SMPL) so as to maintain a time-bandwidh product sufficient to
    recover the original signal.

 ** DTA: Must we handle separately the case B = M_lct[1,2] = 0?
         What about the case |B| << 1?

    The decompositions used here comes from the work of Koç, et al.,
    in _IEEE Trans. Signal Proc._ 56(6):2383--2394, June 2008.

    Argument:
    M_lct -- symplectic 2x2 matrix that describes the desired LCT

    Return an array of size mx2, where m denotes the total number of
    operations in the decomposition. Each row has the form ['STR', p],
    where 'STR' specifies the operation, and p the parameter relevant
    for that operation.
    """
    alpha, beta, gamma = convert_params_4to3(M_lct)
    ag = abs(gamma)
    if ag <= 1:
        k = 1 + ag + abs(alpha) / beta ** 2 * (1 + ag) ** 2
        seq = [ [ 'SCL',   beta              ],
                ['RSMP',     2.              ],
                [  'CM', - gamma / beta ** 2 ],
                ['LCFT',     0               ],
                ['RSMP',    k/2              ],
                [  'CM', - alpha             ] ]
    else:
        k = 1 + 1 / ag + abs(alpha - beta ** 2 / gamma) / beta ** 2 * (1 + ag) ** 2
        seq = [ [ 'SCL', - gamma / beta              ],
                ['LCFT',     0                       ],
                ['RSMP',     2.                      ],
                [  'CM',   gamma / beta ** 2         ],
                ['LCFT',     0                       ],
                ['RSMP',    k/2                      ],
                [  'CM', - alpha + beta ** 2 / gamma ] ]

    return seq


def apply_lct(M_lct, in_signal):
    """
    Apply LCT[M_lct] to a given input signal, and return the result.

    Given a symplectic 2x2 ABCD matrix that defines an LCT, decompose
    the matrix into a sequence of simpler operations, so as to achieve
    an operation count of ~O(N log N).

    The algorithm implemented here is that given by Koç, et al.
    in IEEE Trans. Signal Proc. 56(6):2383--2394, June 2008.

 ** DTA: Consider implementing one or more of the other known
         fast LCT algorithms. Doing so can help with verifying
         correctness, as well as allow us to learn something
         about the relative performance of different algorithms.

    Arguments:
    M_lct -- symplectic 2x2 matrix that describes the desired LCT
    in_signal -- the signal to transform, [ dX, signal_array], where
                 dX denotes the sample interval of the given signal

    Return the transformed signal in the form [ dY, LCT[M_lct](signal)].
    """
    seq = lct_decomposition(M_lct)
    signal0 = in_signal
    for lct in seq:
        if   lct[0] == 'CM':
            signal1 = chirp_multiply(lct[1], signal0)
            signal0 = signal1
        elif lct[0] == 'LCFT':
            signal1 = lct_fourier(signal0)
            signal0 = signal1
        elif lct[0] == 'SCL':
            signal1 = scale_signal(lct[1], signal0)
            signal0 = signal1
        elif lct[0] == 'RSMP':
            signal1 = resample_signal(lct[1], signal0)
            signal0 = signal1
        else:
            assert False, 'LCT code ' + lct[0] + ' not recognized! Exiting now.'
            return -1

    return signal1


def apply_lct_2d_sep(MX_lct, MY_lct, in_signal):
    """
    Apply LCT[M_lct] to a given input signal, and return the result.

    Given a pair of symplectic 2x2 ABCD matrix that defines an uncoupled
    LCT in two dimensions, decompose
    the matrix into a sequence of simpler operations, so as to achieve
    an operation count of ~O(N log N).

    The algorithm implemented here is that given by Koç, et al.
    in IEEE Trans. Signal Proc. 56(6):2383--2394, June 2008.

    Arguments:
    M_lct -- symplectic 2x2 matrix that describes the desired LCT
    in_signal -- the signal to transform, [ dX, signal_array], where
                 dX denotes the sample interval of the given signal

    Return the transformed signal in the form [ dY, LCT[M_lct](signal)].
    """
    seq_x = lct_decomposition(MX_lct)
    seq_y = lct_decomposition(MY_lct)
    signal0 = in_signal
    for lct in seq:
        if   lct[0] == 'CM':
            signal1 = chirp_multiply(lct[1], signal0)
            signal0 = signal1
        elif lct[0] == 'LCFT':
            signal1 = lct_fourier(signal0)
            signal0 = signal1
        elif lct[0] == 'SCL':
            signal1 = scale_signal(lct[1], signal0)
            signal0 = signal1
        elif lct[0] == 'RSMP':
            signal1 = resample_signal(lct[1], signal0)
            signal0 = signal1
        else:
            assert False, 'LCT code ' + lct[0] + ' not recognized! Exiting now.'
            return -1

    return signal1


