import numpy as np
import spikeinterface.extractors as se

recording = se.read_intan("data/RAT3_PRICKING/pricking_170227_133227.rhd", stream_id="0")

downsample = 20

data = recording.get_traces(start_frame=0, end_frame = 1000)

data = np.asarray(data)

print(data.shape)

if data.shape[0] < data.shape[1]:
    data = data.T

downsampled = data[::downsample, :]
print("downsample", downsampled.shape)

fs = 30000

print(f"Original sampling frequency: {fs} Hz")
new_fs = fs / downsample
print(f"Downsampled sampling frequency: {new_fs} Hz")
