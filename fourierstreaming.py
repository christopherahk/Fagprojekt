import serial
import time
import numpy as np
import tqdm
import os
import random
from collections import deque
import spikeinterface.extractors as se
from scipy.signal import decimate, stft
from scipy.fft import dct

SERIAL_PORT = "COM3"
BAUD_RATE = 115200

# Vælg transformation:
# 0 = FFT, 1 = DCT, 2 = STFT
FT = 0
FT_NAMES = {0: "FFT", 1: "DCT", 2: "STFT"}

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

# Stream-hastighed for FFT-vinduer (vinduer/sek). 0 = så hurtigt som muligt.
STREAM_RATE_HZ = 20

# FFT-konfiguration
FFT_SIZE = 1024
MIN_PAYLOAD_BINS = 6
MAX_FREQ_HZ = 300.0

# Præprocessering før FFT
NORMALIZE = True
REMOVE_DC = True
APPLY_HANN_WINDOW = True

# Valgfri downsample før FFT. Sæt til 1 for at beholde 30 kHz.
DOWNSAMPLE_BEFORE_FFT = 1
DOWNSAMPLE_METHOD = "decimate"  # "decimate", "mean", "none"

# Hvis True: sender "chX,val1,val2,..."
# Hvis False: sender kun "val1,val2,..."
INCLUDE_CHANNEL_PREFIX = False

# Hvilket Intan stream-id der skal bruges. Vores amp er "0"
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


def maybe_downsample(raw_chunk: np.ndarray) -> np.ndarray:
	factor = int(DOWNSAMPLE_BEFORE_FFT)
	if factor <= 1:
		return raw_chunk.astype(np.float32)

	method = DOWNSAMPLE_METHOD.lower()
	if method == "decimate":
		return decimate(raw_chunk, factor, axis=0).astype(np.float32)

	if method == "mean":
		n_blocks = raw_chunk.shape[0] // factor
		if n_blocks == 0:
			return np.empty((0, raw_chunk.shape[1]), dtype=np.float32)
		trimmed = raw_chunk[: n_blocks * factor]
		return trimmed.reshape(n_blocks, factor, raw_chunk.shape[1]).mean(axis=1).astype(np.float32)

	if method == "none":
		return raw_chunk[::factor].astype(np.float32)

	raise ValueError(
		f"unknown DOWNSAMPLE_METHOD: {DOWNSAMPLE_METHOD}. Use 'decimate', 'mean' or 'none'."
	)


def preprocess_for_fft(chunk: np.ndarray) -> np.ndarray:
	data = chunk.astype(np.float32)

	if NORMALIZE:
		# Normaliserer til ca. [-1, 1] fra signed int16 range.
		data = data / 32768.0

	if REMOVE_DC:
		# Fjern DC-offset pr. kanal for at undgå stor 0 Hz-peak.
		data = data - np.mean(data, axis=0, keepdims=True)

	if APPLY_HANN_WINDOW:
		window = np.hanning(data.shape[0]).astype(np.float32)
		data = data * window[:, None]

	return data


def transform_channel(chunk: np.ndarray, fs: float) -> np.ndarray:
	if FT == 0:
		freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0 / fs).astype(np.float32)
		mag = np.abs(np.fft.rfft(chunk, n=FFT_SIZE)).astype(np.float32)
		if MAX_FREQ_HZ is not None:
			mag = mag[freqs <= float(MAX_FREQ_HZ)]
		return mag

	if FT == 1:
		return np.abs(dct(chunk, type=2, norm="ortho")).astype(np.float32)

	if FT == 2:
		nperseg = min(256, chunk.size)
		if nperseg < 2:
			return np.empty((0,), dtype=np.float32)
		noverlap = nperseg // 2
		freqs, _, zxx = stft(chunk, fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None, padded=False)
		if zxx.size == 0:
			return np.empty((0,), dtype=np.float32)
		mag = np.mean(np.abs(zxx), axis=1).astype(np.float32)
		if MAX_FREQ_HZ is not None:
			mag = mag[freqs <= float(MAX_FREQ_HZ)]
		return mag

	raise ValueError("FT must be 0 (FFT), 1 (DCT) or 2 (STFT).")


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
	raw_fs = float(recording.get_sampling_frequency())
	n_samples = recording.get_num_samples()
	n_channels = recording.get_num_channels()

	if FT not in FT_NAMES:
		raise ValueError(f"Invalid FT={FT}. Use 0, 1 or 2.")

	effective_fs = raw_fs / max(1, int(DOWNSAMPLE_BEFORE_FFT))
	raw_window = FFT_SIZE * max(1, int(DOWNSAMPLE_BEFORE_FFT))
	chunk_idx = 0
	chunk_duration_s = 1.0 / STREAM_RATE_HZ if STREAM_RATE_HZ > 0 else 0.0

	print(f"Streaming {FT_NAMES[FT]}: {path}")
	print(f"Raw fs: {raw_fs} Hz | channels: {n_channels} | samples: {n_samples}")
	print(
		f"FT: {FT_NAMES[FT]} | FFT_SIZE: {FFT_SIZE} | eff fs: {effective_fs} Hz | downsample: {DOWNSAMPLE_BEFORE_FFT} ({DOWNSAMPLE_METHOD})"
	)

	for raw_start in range(0, n_samples, raw_window):
		raw_end = min(raw_start + raw_window, n_samples)
		is_last_chunk = raw_end >= n_samples
		raw_chunk = recording.get_traces(start_frame=raw_start, end_frame=raw_end)

		if raw_chunk.shape[0] < max(2, int(DOWNSAMPLE_BEFORE_FFT)):
			continue

		downsampled = maybe_downsample(raw_chunk)
		if downsampled.shape[0] < 2:
			continue

		if downsampled.shape[0] < FFT_SIZE:
			if is_last_chunk:
				# Tillad padding i sidste vindue, så vi ikke mister hale-data.
				pad = np.zeros((FFT_SIZE - downsampled.shape[0], n_channels), dtype=np.float32)
				downsampled = np.vstack([downsampled, pad])
			else:
				continue
		elif downsampled.shape[0] > FFT_SIZE:
			downsampled = downsampled[:FFT_SIZE]

		prepared = preprocess_for_fft(downsampled)

		if prepared.shape[0] < MIN_PAYLOAD_BINS:
			if is_last_chunk:
				print(
					f"[END] Ignore last short spectrum: {prepared.shape[0]} < {MIN_PAYLOAD_BINS} in {path}."
				)
				break
			print(
				f"[SKIP] Spectrum too small: {prepared.shape[0]} < {MIN_PAYLOAD_BINS} in {path}."
			)
			return False

		channel_data = prepared.T  # [kanal, tid]
		payload_batch = []

		for ch_idx in range(channel_data.shape[0]):
			bins = transform_channel(channel_data[ch_idx], fs=effective_fs)
			if bins.size < MIN_PAYLOAD_BINS:
				if is_last_chunk:
					payload_batch = []
					break
				print(
					f"[SKIP] Spectrum too small on channel {ch_idx + 1}: {bins.size} < {MIN_PAYLOAD_BINS} in {path}."
				)
				return False

			values = ",".join(str(x) for x in bins)
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
			feature_type=FT_NAMES[FT].lower(),
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
			feature_type=FT_NAMES[FT].lower(),
			is_replay=False,
		)

		if replay_buffer is not None:
			replay_buffer.append(list(payload_batch))

		chunk_idx += 1

		if STREAM_RATE_HZ > 0:
			time.sleep(1.0 / STREAM_RATE_HZ)

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
			print("File skippet pga. payload-size.")
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

			for path in tqdm.tqdm(file_paths, desc=f"RHD files (FFT) | epoch {epoch_idx + 1}"):
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

		print(f"Done {FT_NAMES[FT]} stream. Streamed: {total_ok} | Skipped: {total_skipped}")
finally:
	ser.close()
