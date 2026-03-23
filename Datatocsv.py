import os
import numpy as np
import tqdm
import spikeinterface.extractors as se
from scipy.signal import decimate

INPUT_FOLDER = "data"
OUTPUT_FOLDER = "data2"

DOWNSAMPLE = 100

check = False

for root, dirs, files in os.walk(INPUT_FOLDER):


    for file in tqdm.tqdm(files, desc="Files"):

        if not file.endswith(".rhd"):
            continue

        input_path = os.path.join(root, file)

        # lav output mappe
        relative = os.path.relpath(root, INPUT_FOLDER)
        output_dir = os.path.join(OUTPUT_FOLDER, relative)
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(
            output_dir,
            file.replace(".rhd", ".csv")
        )

        print("Processing:", input_path)

        # load recording
        recording = se.read_intan(input_path, stream_id="0")

        fs = recording.get_sampling_frequency()
        traces = recording.get_traces()
        print("Original shape:", traces.shape)

        if check == True:
            WINDOW = int(fs * 0.1)   # 100 ms

            features = []

            for start in range(0, traces.shape[0] - WINDOW, WINDOW):

                segment = traces[start:start+WINDOW]

                mean = np.mean(segment, axis=0)
                std = np.std(segment, axis=0)
                rms = np.sqrt(np.mean(segment**2, axis=0))
                mx = np.max(np.abs(segment), axis=0)

                feature_vector = np.concatenate([mean, std, rms, mx])

                features.append(feature_vector)

            features = np.array(features)

            np.savetxt(
            output_file,
            features,
            delimiter=",")
        else:
        # downsample
            downsampled = decimate(traces, DOWNSAMPLE, axis=0)
            downsampled = downsampled.astype(np.float32) # konverter til float32 for at spare plads, kan også bruge int16 hvis nødvendigt

            new_fs = fs / DOWNSAMPLE
            print("New sampling rate:", new_fs)

        # gem som CSV
            np.savetxt(
            output_file,
            downsampled,
            delimiter=","
            )
        print("Saved:", output_file)
