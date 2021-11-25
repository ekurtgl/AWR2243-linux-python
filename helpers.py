import numpy as np

def stft(data, window, nfft, shift):
    n = (len(data) - window - 1) // shift
    out1 = np.zeros((nfft, n), dtype=complex)

    for i in range(n):
        tmp1 = data[i * shift: i * shift + window].T
        tmp2 = np.hanning(window)
        tmp3 = tmp1 * tmp2
        tmp = np.fft.fft(tmp3, n=nfft)
        out1[:, i] = tmp
    return out1
