import numpy as np
from helpers import radar_sample_reader, stft
import matplotlib.pyplot as plt

save_spectrograms = True
numADCSamples = 91*2
numTxAntennas = 1
numRxAntennas = 4
numLoopsPerFrame = 1008
numRangeBins = int(numADCSamples/2)
sweeptime = 80e-3
dt = sweeptime / numLoopsPerFrame
prf = 1/dt
duration = 5

main = '/home/emre/Desktop/77ghz/open radar/temp/data/openradar/'
filename = main + "openradarParam_asl_real.open_radar"
reader = radar_sample_reader(filename)

count = 0
datax = []
while True:
    header, np_raw_frame = reader.getNextSample()
    if header == 0 and np_raw_frame == 0:
        break
    raw_frame = np.array(np_raw_frame, dtype=np.float32)

    data = raw_frame.reshape(numRxAntennas * numADCSamples, numLoopsPerFrame, order='F')
    data2 = np.zeros((numADCSamples, numLoopsPerFrame, numRxAntennas))
    for i in range(numRxAntennas):
        data2[:, :, i] = np.array(data[i::numRxAntennas])
    datax.append(data2)
    # ret = np.zeros(len(raw_frame), dtype=float)
    # quart_length = int(ret.shape[0] / 4)
    #
    # ret[0:quart_length] = raw_frame[0::4]
    # ret[quart_length:2 * quart_length] = raw_frame[1::4]
    # ret[2 * quart_length:3 * quart_length] = raw_frame[2::4]
    # ret[3 * quart_length:4 * quart_length] = raw_frame[3::4]
    #
    # ret = ret.reshape((numRxAntennas, numLoopsPerFrame, numADCSamples))
    #
    # range_processed = np.fft.fft(ret.transpose() , axis=0).transpose()
    # range_processed = range_processed[:, :, 0:int(range_processed.shape[2] / 2)]
    #
    # range_processed = np.fft.fftshift(np.fft.fft(range_processed[0, :, :], axis=0), axes=0)
    # doppler_view_np = (20 * np.log10(np.abs(range_processed)).transpose())
datax = np.array(datax)
datax = np.swapaxes(datax, 0, 1).reshape(numADCSamples, numLoopsPerFrame * datax.shape[0], numRxAntennas)
print('datax', datax.shape)

# Range FFT
rp = np.fft.fft(datax)

# micro-Doppler Spectrogram
rBin = np.arange(5, 16)
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
    savename = filename[:-12] + '_py_process_openradar.png'

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
    plt.title('Real openradar ASL python')
    fig.savefig(savename, dpi=200)


