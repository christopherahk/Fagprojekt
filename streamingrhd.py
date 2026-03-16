import serial
import time
import numpy as np
import tqdm
import os
import spikeinterface.extractors as se
from scipy.signal import decimate

ser = serial.Serial("COM3", 115200)

folder = "data/RAT3_PRICKING"

# Sæt til en konkret .rhd-fil for at streame kun den ene fil.
# fx:
# RHD_FILE = "data/RAT10_DORSIFLEXION/dorsi_170605_122607.rhd"
RHD_FILE = None

WINDOW_SIZE = 100
STREAM_RATE_HZ = 300

# stop stream hvis payload pr. kanal bliver mindre end dette
MIN_PAYLOAD_SAMPLES = WINDOW_SIZE

# downsample fra 30 kHz -> 300 Hz (samme som CSV-pipeline)
DOWNSAMPLE = 100

# hvis True: sender "chX,val1,val2,..."
# hvis False: sender kun "val1,val2,..."
INCLUDE_CHANNEL_PREFIX = False

# hvilket Intan stream-id der skal bruges. vores amp er "0"
STREAM_ID = "0"


def stream_rhd_file(path: str) -> bool:
	recording = se.read_intan(path, stream_id=STREAM_ID)
	fs = recording.get_sampling_frequency()
	n_samples = recording.get_num_samples()
	n_channels = recording.get_num_channels()

	print(f"Streaming: {path}")
	print(f"Raw fs: {fs} Hz | channels: {n_channels} | samples: {n_samples}")

	raw_window = WINDOW_SIZE * DOWNSAMPLE

	for raw_start in range(0, n_samples, raw_window):
		raw_end = min(raw_start + raw_window, n_samples)
		raw_chunk = recording.get_traces(start_frame=raw_start, end_frame=raw_end)

		# For korte chunks kan ikke decimeres stabilt
		if raw_chunk.shape[0] < DOWNSAMPLE:
			continue

		downsampled = decimate(raw_chunk, DOWNSAMPLE, axis=0).astype(np.float32)
		channel_data = downsampled.T  # [kanal, tid]

		if downsampled.shape[0] < MIN_PAYLOAD_SAMPLES:
			print(
				f"[SKIP] Payload for lille: {downsampled.shape[0]} < {MIN_PAYLOAD_SAMPLES} samples i {path}."
			)
			return False

		payload_batch = []
		for ch_idx in range(channel_data.shape[0]):
			chunk = channel_data[ch_idx]
			if chunk.size == 0:
				continue

			if chunk.size < MIN_PAYLOAD_SAMPLES:
				print(
					f"[SKIP] Payload for lille på kanal {ch_idx + 1}: {chunk.size} < {MIN_PAYLOAD_SAMPLES} i {path}."
				)
				return False

			values = ",".join(str(x) for x in chunk)
			if INCLUDE_CHANNEL_PREFIX:
				line = f"ch{ch_idx + 1},{values}\n"
			else:
				line = values + "\n"

			payload_batch.append(line)

		for line in payload_batch:
			ser.write(line.encode("utf-8"))

		if STREAM_RATE_HZ > 0:
			sent_samples = downsampled.shape[0]
			time.sleep(sent_samples / STREAM_RATE_HZ)

	return True


try:
	if RHD_FILE is not None:
		success = stream_rhd_file(RHD_FILE)
		if not success:
			print("Filen blev skippet pga. payload-størrelse.")
	else:
		total_ok = 0
		total_skipped = 0
		for root, _, files in os.walk(folder):
			rhd_files = [f for f in sorted(files) if f.endswith(".rhd")]

			for file in tqdm.tqdm(rhd_files, desc="RHD files"):
				path = os.path.join(root, file)
				success = stream_rhd_file(path)
				if success:
					total_ok += 1
				else:
					total_skipped += 1

		print(f"Done. Streamed: {total_ok} | Skipped: {total_skipped}")
finally:
	ser.close()
