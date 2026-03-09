import time

effective_fs = fs / DOWNSAMPLE          # ny sampling rate
dt = 1.0 / effective_fs                 # tid mellem samples

N = 100                                 # hvor mange linjer vi vil lave
N = min(N, downsampled.shape[0])        # sikkerhed hvis filen er kort

t0 = time.perf_counter()                # start-tid
next_time = t0                          # næste sendetid

with open("stream_preview.csv", "w") as f:           # lav en fil vi kan åbne bagefter
    f.write("t_ms," + ",".join([f"ch{i}" for i in range(downsampled.shape[1])]) + "\n")  # header-linje

    for i in range(N):                                # loop over N samples
        now = time.perf_counter()                     # nu-tid
        if now < next_time:
            time.sleep(next_time - now)               # vent så tempoet bliver realistisk

        frame = downsampled[i, :]                     # én sample = alle kanaler
        t_ms = int((time.perf_counter() - t0) * 1000) # ms siden start

        line = str(t_ms) + "," + ",".join(f"{v:.6f}" for v in frame) + "\n"  # CSV-linje med newline
        f.write(line)                                  # skriv linjen til fil

        next_time += dt                                # næste tidspunkt

print("Wrote stream_preview.csv with", N, "lines.")
