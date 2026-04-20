import serial
import time
import numpy as np
import tqdm
import os
import random
from collections import deque
import spikeinterface.extractors as se
from scipy.signal import decimate

SERIAL_PORT = "COM3"
BAUD_RATE = 115200

folder = "data"

# Kørselstilstand:
# - "supervised_warmup": tillader shuffle + flere epoker
# - "live_online": kausal live-stream (ingen shuffle som default)
MODE = "live_online"

# Antal epoker bruges kun i supervised_warmup.
EPOCHS = 3

# Hvis True: blander rækkefølgen af .rhd-filer inden streaming.
SHUFFLE_FILE_ORDER = False

# Sæt til et heltal for reproducerbar shuffle (fx 42), eller None for ny shuffle hver kørsel.
SHUFFLE_SEED = 1

# Serial metadata før hver payload. Slå fra hvis Arduino-parser kun forventer rene tal-linjer.
SEND_METADATA = False

# Replay-buffer kan give mere stabil online-læring uden global shuffle.
ENABLE_REPLAY_BUFFER = False
REPLAY_BUFFER_SIZE = 32
REPLAY_EVERY_N_CHUNKS = 8

# Sæt til en .rhd-fil for at streame kun den ene fil.
# fx:
# RHD_FILE = "data/RAT10_DORSIFLEXION/dorsi_170605_122607.rhd"
RHD_FILE = None

WINDOW_SIZE = 100
STREAM_RATE_HZ = 300

# Standardiser alle filer til 56 kanaler.
TARGET_CHANNELS = 56

# stop stream hvis payload pr. kanal bliver mindre end dete
MIN_PAYLOAD_SAMPLES = WINDOW_SIZE

# downsample fra 30 kHz -> 300 Hz (samme som CSV-pipeline)
DOWNSAMPLE = 100

# vælg resampling-metode: "decimate", "mean" eller "none"
METHOD = "decimate"
# "decimate" bruger anti-aliasing filer
# "mean" tager gennemsnit over blokke
# "none" tager hver N-te prøve uden filtrering


NORMALIZE = True


# hvis True: sender "chX,val1,val2,..."
# hvis False: sender kun "val1,val2,..."
INCLUDE_CHANNEL_PREFIX = False

# hvilket Intan stream-id der skal bruges. vores amp er "0"
STREAM_ID = "0"


def resolve_mode_settings() -> tuple[bool, int, bool]:
	if MODE not in {"supervised_warmup", "live_online"}:
		raise ValueError("MODE must be 'supervised_warmup' or 'live_online'.")

	is_warmup = MODE == "supervised_warmup"
	epoch_count = max(1, int(EPOCHS)) if is_warmup else 1
	shuffle_enabled = bool(SHUFFLE_FILE_ORDER) if is_warmup else False
	return is_warmup, epoch_count, shuffle_enabled


def make_metadata_line(
	*,
	mode: str,
	epoch_idx: int,
	source: str,
	chunk_idx: int,
	feature_type: str,
	is_replay: bool,
) -> str:
	flag = 1 if is_replay else 0
	return (
		f"#META,mode={mode},epoch={epoch_idx},source={source},chunk={chunk_idx},"
		f"feature={feature_type},replay={flag}\n"
	)


def write_payload(
	ser: serial.Serial,
	payload_batch: list[str],
	*,
	send_metadata: bool,
	mode: str,
	epoch_idx: int,
	source: str,
	chunk_idx: int,
	feature_type: str,
	is_replay: bool,
) -> None:
	if send_metadata:
		meta_line = make_metadata_line(
			mode=mode,
			epoch_idx=epoch_idx,
			source=source,
			chunk_idx=chunk_idx,
			feature_type=feature_type,
			is_replay=is_replay,
		)
		ser.write(meta_line.encode("utf-8"))

	for line in payload_batch:
		ser.write(line.encode("utf-8"))


def maybe_send_replay_payload(
	*,
	ser: serial.Serial,
	replay_buffer: deque[list[str]] | None,
	replay_rng: random.Random | None,
	chunk_idx: int,
	mode: str,
	epoch_idx: int,
	source: str,
	send_metadata: bool,
	feature_type: str,
	chunk_duration_s: float,
) -> None:
	if mode != "live_online" or replay_buffer is None or replay_rng is None:
		return
	if REPLAY_EVERY_N_CHUNKS <= 0 or len(replay_buffer) == 0:
		return
	if chunk_idx == 0 or (chunk_idx % REPLAY_EVERY_N_CHUNKS) != 0:
		return

	replay_payload = replay_rng.choice(list(replay_buffer))
	write_payload(
		ser,
		replay_payload,
		send_metadata=send_metadata,
		mode=mode,
		epoch_idx=epoch_idx,
		source=source,
		chunk_idx=chunk_idx,
		feature_type=feature_type,
		is_replay=True,
	)
	if STREAM_RATE_HZ > 0 and chunk_duration_s > 0:
		time.sleep(chunk_duration_s)



def resample_chunk(raw_chunk: np.ndarray) -> np.ndarray:
	method = METHOD.lower()

	def normalize_chunk(chunk: np.ndarray) -> np.ndarray:
		chunk_f32 = chunk.astype(np.float32)
		if NORMALIZE:
			# Normaliserer til [-1, 1] baseret på 16-bit int range
			return chunk_f32 / 32768.0
		return chunk_f32

	if method == "decimate":
		decichunk = decimate(raw_chunk, DOWNSAMPLE, axis=0).astype(np.float32)
		return normalize_chunk(decichunk)

	if method == "mean":
		n_blocks = raw_chunk.shape[0] // DOWNSAMPLE
		if n_blocks == 0:
			return np.empty((0, raw_chunk.shape[1]), dtype=np.float32)
		trimmed = raw_chunk[: n_blocks * DOWNSAMPLE]
		return normalize_chunk(trimmed.reshape(n_blocks, DOWNSAMPLE, raw_chunk.shape[1]).mean(axis=1).astype(np.float32))

	if method == "none":
		# Hurtig benchmark-metode uden anti-alias filtering.
		return normalize_chunk(raw_chunk[::DOWNSAMPLE])

	raise ValueError(f"unknown METHOD: {METHOD}. Use 'decimate', 'mean' or 'none'.")


def collect_rhd_files(root_folder: str, shuffle: bool = False, seed: int | None = None) -> list[str]:
	paths: list[str] = []
	for root, dirs, files in os.walk(root_folder):
		dirs.sort()
		rhd_files = [f for f in sorted(files) if f.endswith(".rhd")]
		for file in rhd_files:
			paths.append(os.path.join(root, file))

	if shuffle:
		rng = random.Random(seed)
		rng.shuffle(paths)

	return paths


def trim_channels(data: np.ndarray, target_channels: int = TARGET_CHANNELS) -> np.ndarray:
	if data.shape[1] < target_channels:
		raise ValueError(f"expected at least {target_channels} channels, got {data.shape[1]}")
	if data.shape[1] == target_channels:
		return data
	return data[:, :target_channels]


def stream_rhd_file(
	path: str,
	ser: serial.Serial,
	*,
	mode: str,
	epoch_idx: int,
	send_metadata: bool,
	replay_buffer: deque[list[str]] | None,
	replay_rng: random.Random | None,
) -> bool:
	recording = se.read_intan(path, stream_id=STREAM_ID)
	fs = recording.get_sampling_frequency()
	n_samples = recording.get_num_samples()
	n_channels = recording.get_num_channels()

	print(f"Streaming: {path}")
	print(f"Raw fs: {fs} Hz | channels: {n_channels} | samples: {n_samples}")
	print(f"Method: {METHOD} | factor: {DOWNSAMPLE}")

	raw_window = WINDOW_SIZE * DOWNSAMPLE
	chunk_idx = 0
	chunk_duration_s = WINDOW_SIZE / STREAM_RATE_HZ if STREAM_RATE_HZ > 0 else 0.0

	for raw_start in range(0, n_samples, raw_window):
		raw_end = min(raw_start + raw_window, n_samples)
		is_last_chunk = raw_end >= n_samples
		raw_chunk = recording.get_traces(start_frame=raw_start, end_frame=raw_end)

		# For korte chunks kan ikke decimeres stabilt
		if raw_chunk.shape[0] < DOWNSAMPLE:
			continue

		downsampled = resample_chunk(raw_chunk)
		downsampled = trim_channels(downsampled)
		channel_data = downsampled.T  # [kanal, tid]

		if downsampled.shape[0] < MIN_PAYLOAD_SAMPLES:
			if is_last_chunk:
				print(
					f"[END] Ignore last short payload: {downsampled.shape[0]} < {MIN_PAYLOAD_SAMPLES} in {path}."
				)
				break
			print(
				f"[SKIP] Payload too small: {downsampled.shape[0]} < {MIN_PAYLOAD_SAMPLES} samples in {path}."
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
						f"[END] Ignore last short payload on channel {ch_idx + 1}: {chunk.size} < {MIN_PAYLOAD_SAMPLES} in {path}."
					)
					payload_batch = []
					break
				print(
					f"[SKIP] Payload too small on channel {ch_idx + 1}: {chunk.size} < {MIN_PAYLOAD_SAMPLES} in {path}."
				)
				return False

			values = ",".join(str(x) for x in chunk)
			if INCLUDE_CHANNEL_PREFIX:
				line = f"ch{ch_idx + 1},{values}\n"
			else:
				line = values + "\n"

			payload_batch.append(line)

		maybe_send_replay_payload(
			ser=ser,
			replay_buffer=replay_buffer,
			replay_rng=replay_rng,
			chunk_idx=chunk_idx,
			mode=mode,
			epoch_idx=epoch_idx,
			source=os.path.basename(path),
			send_metadata=send_metadata,
			feature_type="time",
			chunk_duration_s=chunk_duration_s,
		)

		write_payload(
			ser,
			payload_batch,
			send_metadata=send_metadata,
			mode=mode,
			epoch_idx=epoch_idx,
			source=os.path.basename(path),
			chunk_idx=chunk_idx,
			feature_type="time",
			is_replay=False,
		)

		if replay_buffer is not None:
			replay_buffer.append(list(payload_batch))

		chunk_idx += 1

		if STREAM_RATE_HZ > 0:
			sent_samples = downsampled.shape[0]
			time.sleep(sent_samples / STREAM_RATE_HZ)

	return True


ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
try:
	is_warmup, epoch_count, shuffle_enabled = resolve_mode_settings()
	replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE) if ENABLE_REPLAY_BUFFER else None
	replay_rng = random.Random(SHUFFLE_SEED) if ENABLE_REPLAY_BUFFER else None

	print(
		f"Mode={MODE} | epochs={epoch_count} | shuffle={shuffle_enabled} | "
		f"replay={ENABLE_REPLAY_BUFFER}"
	)

	if RHD_FILE is not None:
		success = stream_rhd_file(
			RHD_FILE,
			ser=ser,
			mode=MODE,
			epoch_idx=0,
			send_metadata=SEND_METADATA,
			replay_buffer=replay_buffer,
			replay_rng=replay_rng,
		)
		if not success:
			print("File skippet due to payload-size.")
	else:
		total_ok = 0
		total_skipped = 0
		for epoch_idx in range(epoch_count):
			epoch_seed = None if SHUFFLE_SEED is None else SHUFFLE_SEED + epoch_idx
			file_paths = collect_rhd_files(folder, shuffle=shuffle_enabled, seed=epoch_seed)
			if shuffle_enabled:
				print(f"Epoch {epoch_idx + 1}/{epoch_count}: shuffled {len(file_paths)} files (seed={epoch_seed}).")
			else:
				print(f"Epoch {epoch_idx + 1}/{epoch_count}: streaming {len(file_paths)} files in deterministic order.")

			for path in tqdm.tqdm(file_paths, desc=f"RHD files | epoch {epoch_idx + 1}"):
				success = stream_rhd_file(
					path,
					ser=ser,
					mode=MODE,
					epoch_idx=epoch_idx,
					send_metadata=SEND_METADATA,
					replay_buffer=replay_buffer,
					replay_rng=replay_rng,
				)
				if success:
					total_ok += 1
				else:
					total_skipped += 1

		print(f"Done. Streamed: {total_ok} | Skipped: {total_skipped}")
finally:
	ser.close()
