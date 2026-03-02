import numpy as np
import spikeinterface.extractors as se
import pandas as pd
recording = se.read_intan("data/RAT3_PRICKING/pricking_170227_133227.rhd", stream_id="0")

downsample = 100

data = recording.get_traces(start_frame=0, end_frame = 10000)

data = np.asarray(data)

df = pd.DataFrame(data)

print(data.shape)

if data.shape[0] < data.shape[1]:
    data = data.T

downsampled = data[::downsample, :]
print("downsample", downsampled.shape)
dfdowns = pd.DataFrame(downsampled)

fs = 30000

# print(f"Original sampling frequency: {fs} Hz")
new_fs = fs / downsample
# print(f"Downsampled sampling frequency: {new_fs} Hz")

# print("first row(first five values) of downsampled data: ", downsampled[0, :5])
# print("="*50)
# print("first row(first five values) of original data: ", df.iloc[0, :5].values)
# print("="*50)

# print(df.head())
# print(df.tail())
# print(df.shape)


# print("="*50)
# print("first row(first five values) of downs data: ", dfdowns.iloc[0, :5].values)
# print("="*50)

# print(dfdowns.head())
# print(dfdowns.tail())
# print(dfdowns.shape)

import time
i = 0
t_ms = 0
frame = downsampled[i, :]
print("frame lenght (channels)= ", len(frame))
print("first five values of frame: ", frame[:5])
line = str(t_ms) + "," + ",".join(f"{v:.1f}" for v in frame) + "\n"

print(line[:100])

fields = line.strip().split(",")
print("nr of fields: ", len(fields)) # should be 57 ( 1 timestamp n + 56 channels)

effective_fs = fs/ downsample
dt = 1 / effective_fs

print(f"Effective sampling frequency: {effective_fs:.2f} Hz, dt: {dt:.4f} seconds")

t0 = time.perf_counter()
next_time = t0
start_time = time.time()
print(f"Start time: {start_time}, effective sampling frequency: {effective_fs:.2f} Hz, dt: {dt:.4f} seconds")
for i in range(10):
    now = time.perf_counter()
    if now < next_time:
        time.sleep(next_time - now)

    frame = downsampled[i,:]
    t_ms = int((time.perf_counter()- t0) * 1000)
    line = str(t_ms) + "," + ",".join(f"{v:.1f}" for v in frame) + "\n"
    print(line[:100])
    next_time += dt
end_time = time.time()

print("done")
print(f"End time: {end_time}, total elapsed time: {end_time - start_time:.2f} seconds")
