import numpy as np
import peakutils as pu
from scipy.spatial import ConvexHull
from scipy.signal import hilbert
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def translate_ranges(x, search_ranges):
    assert len(x) == len(search_ranges)
    result = np.zeros(len(x))
    for idx in np.arange(len(x)):
        cr = search_ranges[idx]
        result[idx] = cr[0] + x[idx] * (cr[1] - cr[0])
    return result

def find_dft_peaks(signal_fft, signal_fft_frequencies, maxima_indexes_th=1e-1):
    dist = np.argwhere(signal_fft_frequencies>=2.0)[0] # 2 Hz distance
    indexes = pu.indexes(signal_fft, thres=0.1, min_dist=dist)
    indexesfiltered = [x for x in indexes if signal_fft[x]>maxima_indexes_th]
    estimated_A = np.abs(signal_fft[indexesfiltered])
    estimated_frequencies = signal_fft_frequencies[indexesfiltered]
    return [estimated_A, estimated_frequencies]

def smooth(x, window_len=11, window='hanning'):
    if window_len < 3:
        return x

    # s = np.r_[
    #     x[window_len - 1:0:-1],
    #     x,
    #     x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), x, mode='valid')
    return y


def smooth_fft(signal, Fs, wsize=5, method='hanning'):
    subN = len(signal)
    signal_fft = np.fft.fft(signal) / subN * 2
    signal_freqs = Fs / subN * np.arange(subN)
    # Apply convolution to smooth
    signal_fft = smooth(signal_fft,wsize,'hanning')
    signal_fft = signal_fft[:subN//2-wsize//2+1]
    signal_freqs = signal_freqs[wsize//2-1:subN//2]
    return [signal_freqs, np.abs(signal_fft)]


def extract_transients(Fs, t, signal, th1=1e-4, th2=0.05):
    """This function will extract the transients
    from the signal giving as result an array of
    transients based at time zero.
    Input parameter th is used for abs threshold"""
    transients = []
    hil = hilbert(signal)
    amplitude_envelope = np.abs(hil)
    inst_phase = np.unwrap(np.angle(hil))
    inst_frequency = (np.diff(inst_phase) / (2.0 * np.pi) / (t[1] - t[0]))
    idxs = np.arange(len(amplitude_envelope))
    search_offset = 0
    while True:
        try:
            idx_start = next(
                x for x in idxs[search_offset:]
                if amplitude_envelope[x] > th2)
            idx_start = idx_start + 1
            idx_end = next(
                x for x in idxs[idx_start:]
                if amplitude_envelope[x] < th1)
            search_offset = idx_end
            transient_signal = signal[idx_start:idx_end]
            transient_time = t[idx_start:idx_end]
            # plt.figure()
            # plt.plot(transient_time,transient_signal)
            # plt.show()
            # if len(transient_signal) < 20:
            #     break
            # [fft_freqs, fft] = smooth_fft(transient_signal, Fs)
            # chull = ConvexHull(np.array([[fft_freqs[k], fft[k]] for k in range(len(fft))]))
            # interp = interp1d(fft_freqs[chull.vertices], fft[chull.vertices], kind='cubic')
            # fft = interp(fft_freqs)
            frequency = np.median(inst_frequency[idx_start:idx_end])
            #idx_start += 1
            #idx_start = int(idx_start-np.ceil(Fs*(1/2)*1/frequency))
            # print("Start: {}, End: {}".format(idx_start, idx_end))
            # Use Convex hull and cubic interpolation to smooth out
            # It is useful for only one peak
            # chull = ConvexHull(np.array([[fft_freqs[k], fft[k]] for k in range(len(fft))]))
            # interp = interp1d(fft_freqs[chull.vertices], fft[chull.vertices], kind='cubic')
            # fft = interp(fft_freqs)
            #frequency = fft_freqs[np.argmax(fft)]
            # frequency = np.median(inst_frequency[idx_start:idx_end])
            transients.append({
                "c": np.max(amplitude_envelope[idx_start:idx_end]),
                "omega": frequency * 2.0 * np.pi,
                "phi": 0.0,
                "start": idx_start,
                "end": idx_end,
                "decay": 0.0,
                "time": transient_time,
                "signal": transient_signal
            })
        except StopIteration:
            break
    return transients


def create_transient(A, omega, phi, decay, alpha, t):
    return create_transients([A], [omega], [phi], [decay], [alpha], t)


def create_transients(A, omega, phi, decay, alpha, t):
    N = len(t)
    t_os = list(map(lambda x: np.argmax(t >= x), alpha))
    signal = np.zeros(N)
    S = len(A)
    for idx in range(S):
        subt = t[t_os[idx]:]
        signal[t_os[idx]:] += A[idx] * \
                              np.exp(-(subt - t[t_os[idx]]) / decay[idx]) * \
                              np.cos(omega[idx] * subt + phi[idx])
    return signal

