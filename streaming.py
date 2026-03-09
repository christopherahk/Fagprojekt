import serial
import time
import numpy as np
import tqdm
import os

ser = serial.Serial("COM3", 115200)

folder = "data2/RAT3_PRICKING"

WINDOW_SIZE = 100
STREAM_RATE_HZ = 300

# hvis True: sender "chX,val1,val2,..."
# hvis False: sender kun "val1,val2,..."
INCLUDE_CHANNEL_PREFIX = False

for root, dirs, files in os.walk(folder):
    for file in tqdm.tqdm(sorted(files), desc="Files"):
        if not file.endswith(".csv"):
            continue

        path = os.path.join(root, file)

        data = np.loadtxt(path, delimiter=",")
        data = np.atleast_2d(data)

        # rows=tidsindeks, cols=kanaler -> transpose til [kanal, tid]
        channel_data = data.T  # shape: (n_channels, n_samples)
        n_channels, n_samples = channel_data.shape

        # vindue for vindue over tid
        for start_idx in range(0, n_samples, WINDOW_SIZE):
            payload_batch = []  # vigtig: ny liste pr. loop

            # første 100 fra kanal 1, første 100 fra kanal 2, osv.
            for ch_idx in range(n_channels):
                chunk = channel_data[ch_idx, start_idx:start_idx + WINDOW_SIZE]
                if chunk.size == 0:
                    continue

                values = ",".join(str(x) for x in chunk)
                if INCLUDE_CHANNEL_PREFIX:
                    line = f"ch{ch_idx + 1},{values}\n"
                else:
                    line = values + "\n"

                payload_batch.append(line)  # gem string i listen

            # tøm listen til USB/serial
            for line in payload_batch:
                ser.write(line.encode("utf-8"))

            # pacing pr. batch
            if STREAM_RATE_HZ > 0:
                sent_samples = min(WINDOW_SIZE, n_samples - start_idx)
                time.sleep(sent_samples / STREAM_RATE_HZ)

            # vigtig: tøm listen efter hvert loop
            payload_batch.clear()
