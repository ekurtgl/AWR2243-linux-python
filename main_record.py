from helpers import my_UDP_Receiver, radar_sample_writer
import numpy as np
import threading
from multiprocessing import Pipe
import time
from datetime import datetime

numADCSamples = 256
numTxAntennas = 1
numRxAntennas = 4
numLoopsPerFrame = 255
numTX = 1
numRX = 4
NPpF = numTX * numLoopsPerFrame
SweepTime = 40e-3
sampleFreq = 6.25e6

isComplex = 1  # 1 for real, 2 for complex
plot_rangedoppler = 1
plot_microdoppler = 0

numChirpsPerFrame = numTxAntennas * numLoopsPerFrame

numRangeBins = numADCSamples / 2
numDopplerBins = numLoopsPerFrame

count = 0
if __name__ == '__main__':
    now = datetime.now()
    main_data = '/home/emre/Desktop/77ghz/open_radar/temp/data/my_receiver/'
    date_time = now.strftime(main_data+"%Y_%m_%d_%H_%M_%S")
    filename = str(date_time) + ".bin"

    logger = radar_sample_writer(filename)

    is_running_event = threading.Event()
    output_p, input_p = Pipe(False)
    sampling_thread = my_UDP_Receiver(is_running_event, input_p, n_chirps=numLoopsPerFrame,
                                      n_samples=numADCSamples, isComplex=isComplex)
    is_running_event.set()
    sampling_thread.start()

    last_timestamp = datetime.now()
    cnt = 0
    while True:
        np_raw_frame = output_p.recv()
        timestamp = datetime.now()
        # logging_header[-1] = time.mktime(timestamp.timetuple()) * 1e3 + timestamp.microsecond / 1e3
        # logger.writeNextSample(logging_header, np_raw_frame)
        logger.writeNextSample(np_raw_frame)
        print(timestamp - last_timestamp)
        last_timestamp = timestamp
        cnt += 1

        if plot_rangedoppler:
            from matplotlib import colors
            import matplotlib.pyplot as plt

            if cnt == 1:

                # params
                idletime = 100e-6
                adcStartTime = 5e-6
                rampEndTime = 50e-6
                c = 299792458
                slope = 80e12
                fstart = 77e9
                Bw = 4e9
                fstop = fstart + Bw
                fc = (fstart + fstop) / 2
                lamda = c / fc
                Rmax = sampleFreq * c / (2 * slope)
                Tc = idletime + adcStartTime + rampEndTime
                Tf = SweepTime
                velmax = lamda / (Tc * 4)
                DFmax = velmax / (c / fc / 2)
                rResol = c / (2 * Bw)
                vResol = lamda / (2 * Tf)
                RNGD2_GRID = np.linspace(0, Rmax, numADCSamples)
                DOPP_GRID = np.linspace(DFmax, -DFmax, numLoopsPerFrame)
                V_GRID = (c / fc / 2) * DOPP_GRID

            data = np.array(np_raw_frame, dtype=np.int16)
            numChirps = int(np.ceil(len(data) / 2 / numADCSamples / numRX))

            # zero pad
            zerostopad = int(numADCSamples*numChirps*numRX*2-len(data))
            data = np.concatenate([data, np.zeros((zerostopad,))])
            print('zeropad:', zerostopad)

            # Organize data per RX
            data = data.reshape(numRX * 2, -1, order='F')
            data = data[0:4, :] + data[4:8, :] * 1j
            data = data.T
            data = data.reshape(numADCSamples, numChirps, numRX, order='F')
            print('Data:', data.shape)

            # rd_frame = data[:, :, 0] - np.mean(data[:, :, 0], 1)
            rd_frame = data
            rd_frame = np.fft.fftshift(np.fft.fft2(rd_frame), 1)
            maxval = np.max(abs(rd_frame))
            norm = colors.Normalize(vmin=-75, vmax=None, clip=True)

            # plt.imshow(20 * np.log10((abs(rd_frame) / maxval)), cmap='jet', norm=norm, aspect="auto",
            #            extent=[-velmax, velmax, 0, Rmax])
            # # plt.show()
            # plt.draw()
            # plt.pause(1e-3)

            if cnt == 1:
                print('plot1')
                fig = plt.figure()
                im = plt.imshow((20 * np.log10((abs(rd_frame) / maxval))).astype(np.uint8), cmap='jet', norm=norm, aspect="auto",
                                extent=[-velmax, velmax, 0, Rmax])  # , animated=True)
                plt.xlabel('Velocity (m/s)')
                plt.ylabel('Range (m)')
                plt.title('Range-Doppler map')
                plt.grid('both')
                # fig.canvas.flush_events()
                plt.draw()
                # plt.show()
                plt.pause(1e-3)
                # plt.clf()
            else:
                # print('plot2')
                im.set_data((20 * np.log10((abs(rd_frame) / maxval))).astype(np.uint8))
                # im.set_array(20 * np.log10((abs(rd_frame) / maxval)))
                # fig.canvas.flush_events()
                plt.draw()
                # plt.show()
                plt.pause(1e-3)




