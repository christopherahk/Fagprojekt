import serial
import time
import numpy as np
import tqdm
import os
import spikeinterface.extractors as se
from scipy.signal import decimate

SERIAL_PORT = "COM3"
BAUD_RATE = 115200

folder = "data"

# Sæt til en .rhd-fil for at streame kun den ene fil.
# fx:
# RHD_FILE = "data/RAT10_DORSIFLEXION/dorsi_170605_122607.rhd"
RHD_FILE = None

WINDOW_SIZE = 100
STREAM_RATE_HZ = 300

# stop stream hvis payload pr. kanal bliver mindre end dete
MIN_PAYLOAD_SAMPLES = WINDOW_SIZE

# downsample fra 30 kHz -> 300 Hz (samme som CSV-pipeline)
DOWNSAMPLE = 100

# vælg resampling-metode: "decimate", "mean" eller "none"
METHOD = "decimate"

# hvis True: sender "chX,val1,val2,..."
# hvis False: sender kun "val1,val2,..."
INCLUDE_CHANNEL_PREFIX = False

# hvilket Intan stream-id der skal bruges. vores amp er "0"
STREAM_ID = "0"


def resample_chunk(raw_chunk: np.ndarray) -> np.ndarray:
	method = METHOD.lower()

	if method == "decimate":
		return decimate(raw_chunk, DOWNSAMPLE, axis=0).astype(np.float32)

	if method == "mean":
		n_blocks = raw_chunk.shape[0] // DOWNSAMPLE
		if n_blocks == 0:
			return np.empty((0, raw_chunk.shape[1]), dtype=np.float32)
		trimmed = raw_chunk[: n_blocks * DOWNSAMPLE]
		return trimmed.reshape(n_blocks, DOWNSAMPLE, raw_chunk.shape[1]).mean(axis=1).astype(np.float32)

	if method == "none":
		# Hurtig benchmark-metode uden anti-alias filtering.
		return raw_chunk[::DOWNSAMPLE].astype(np.float32)

	raise ValueError(f"Ukendt METHOD: {METHOD}. Brug 'decimate', 'mean' eller 'none'.")


def collect_rhd_files(root_folder: str) -> list[str]:
	paths: list[str] = []
	for root, _, files in os.walk(root_folder):
		rhd_files = [f for f in sorted(files) if f.endswith(".rhd")]
		for file in rhd_files:
			paths.append(os.path.join(root, file))
	return paths


def stream_rhd_file(path: str, ser: serial.Serial) -> bool:
	recording = se.read_intan(path, stream_id=STREAM_ID)
	fs = recording.get_sampling_frequency()
	n_samples = recording.get_num_samples()
	n_channels = recording.get_num_channels()

	print(f"Streaming: {path}")
	print(f"Raw fs: {fs} Hz | channels: {n_channels} | samples: {n_samples}")
	print(f"Method: {METHOD} | factor: {DOWNSAMPLE}")

	raw_window = WINDOW_SIZE * DOWNSAMPLE

	for raw_start in range(0, n_samples, raw_window):
		raw_end = min(raw_start + raw_window, n_samples)
		is_last_chunk = raw_end >= n_samples
		raw_chunk = recording.get_traces(start_frame=raw_start, end_frame=raw_end)

		# For korte chunks kan ikke decimeres stabilt
		if raw_chunk.shape[0] < DOWNSAMPLE:
			continue

		downsampled = resample_chunk(raw_chunk)
		channel_data = downsampled.T  # [kanal, tid]

		if downsampled.shape[0] < MIN_PAYLOAD_SAMPLES:
			if is_last_chunk:
				print(
					f"[END] Ignorerer sidste korte payload: {downsampled.shape[0]} < {MIN_PAYLOAD_SAMPLES} i {path}."
				)
				break
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
				if is_last_chunk:
					print(
						f"[END] Ignorerer sidste korte payload pa kanal {ch_idx + 1}: {chunk.size} < {MIN_PAYLOAD_SAMPLES} i {path}."
					)
					payload_batch = []
					break
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


ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
try:
	if RHD_FILE is not None:
		success = stream_rhd_file(RHD_FILE, ser=ser)
		if not success:
			print("Filen blev skippet pga. payload-størrelse.")
	else:
		total_ok = 0
		total_skipped = 0
		for path in tqdm.tqdm(collect_rhd_files(folder), desc="RHD files"):
			success = stream_rhd_file(path, ser=ser)
			if success:
				total_ok += 1
			else:
				total_skipped += 1

		print(f"Done. Streamed: {total_ok} | Skipped: {total_skipped}")
finally:
	ser.close()
