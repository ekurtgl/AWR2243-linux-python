import numpy as np
from helpers import stft

main_path = '/home/emre/Desktop/77ghz/open radar/temp/data/'
fname = main_path + 'mm_real_asl_Raw_0.bin'

f = open(fname)
data = np.fromfile(f, dtype=np.int16)
f.close()

# Parameters
save_spectrograms = True
SweepTime = 40e-3
NTS = 256
numADCSamples = NTS
numTX = 1
NoC = 255
NPpF = numTX * NoC
numRX = 4
numChirps = int(np.ceil(len(data) / NTS / numRX))
NoF = round(numChirps / NPpF)
dT = SweepTime / NPpF
prf = 1 / dT
isReal = 0
duration = np.ceil(numChirps*dT)

# zero pad
zerostopad = int(NTS*numChirps*numRX-len(data))
data = np.concatenate([data, np.zeros((zerostopad,))])

# Organize data per RX
data = data.reshape(numRX * NTS, numChirps, order='F')
data2 = np.zeros((NTS, numChirps, numRX))
for i in range(numRX):
    data2[:, :, i] = data[i::numRX]

# Range FFT
rp = np.fft.fft(data2)

# micro-Doppler Spectrogram
rBin = np.arange(15, 36)
nfft = 2**12
window = 256
noverlap = 200
shift = window - noverlap

y2 = np.sum(rp[rBin, :], 0)
sx = stft(y2[:, 0], window, nfft, shift)
sx2 = np.abs((np.fft.fftshift(sx, 0)))

# Plot
if save_spectrograms:
    from matplotlib import colors
    import matplotlib.pyplot as plt

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    savename = fname[:-4] + '_py.png'

    maxval = np.max(sx2)
    norm = colors.Normalize(vmin=-45, vmax=None, clip=True)

    # imwrite (no axes)
    # ax.imshow(20 * np.log10((abs(sx2) / maxval)), cmap='jet', norm=norm, aspect="auto",
    #           extent=[0, duration, -prf/2, prf/2])
    # ax.set_xlabel('Time (sec)')
    # ax.set_ylabel('Frequency (Hz)')
    # ax.set_title('Complex mmwave ASL python')
    # ax.set_ylim([-prf/6, prf/6])
    # # ax.set_axis_off()
    # fig.add_axes(ax)
    # fig.savefig(savename, dpi=200)

    # gcf (with axes)
    plt.imshow(20 * np.log10((abs(sx2) / maxval)), cmap='jet', norm=norm, aspect="auto",
              extent=[0, duration, -prf/2, prf/2])
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim([-prf/6, prf/6])
    plt.title('Complex mmwave ASL python')
    fig.savefig(savename, dpi=200)
