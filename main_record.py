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
isComplex = 1  # 1 for real, 2 for complex

numChirpsPerFrame = numTxAntennas * numLoopsPerFrame

numRangeBins = numADCSamples / 2
numDopplerBins = numLoopsPerFrame

count = 0
if __name__ == '__main__':
    now = datetime.now()
    main_data = '/home/emre/Desktop/77ghz/open_radar/temp/data/my_receiver/'
    date_time = now.strftime(main_data+"%Y_%m_%d_%H_%M_%S")
    # logging_header = np.array([1, numRxAntennas, numTxAntennas, numRangeBins, numLoopsPerFrame, 1], dtype=np.int64)
    filename = str(date_time) + ".bin"

    logger = radar_sample_writer(filename)

    is_running_event = threading.Event()
    output_p, input_p = Pipe(False)
    sampling_thread = my_UDP_Receiver(is_running_event, input_p, n_chirps=numLoopsPerFrame,
                                                n_samples=numADCSamples, isComplex=isComplex)
    is_running_event.set()
    sampling_thread.start()

    last_timestamp = datetime.now()
    while True:
        np_raw_frame = output_p.recv()
        timestamp = datetime.now()
        # logging_header[-1] = time.mktime(timestamp.timetuple()) * 1e3 + timestamp.microsecond / 1e3
        # logger.writeNextSample(logging_header, np_raw_frame)
        logger.writeNextSample(np_raw_frame)
        print(timestamp - last_timestamp)
        last_timestamp = timestamp
