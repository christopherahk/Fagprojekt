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

print(f"Original sampling frequency: {fs} Hz")
new_fs = fs / downsample
print(f"Downsampled sampling frequency: {new_fs} Hz")

print("first row(first five values) of downsampled data: ", downsampled[0, :5])

print("="*50)
print("first row(first five values) of original data: ", df.iloc[0, :5].values)
print("="*50)

print(df.head())
print(df.tail())
print(df.shape)


print("="*50)
print("first row(first five values) of downs data: ", dfdowns.iloc[0, :5].values)
print("="*50)

print(dfdowns.head())
print(dfdowns.tail())
print(dfdowns.shape)
