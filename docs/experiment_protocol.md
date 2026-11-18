# Experiment protocol and results log

Run the experiments in this order. Each one checks a single element of the chain against an
independent ground truth before the next one depends on it. Fill in the tables as you go:
together they make the "measured results" section of the README. Do not report a number that
is not in these tables.

Common settings: `config/indoor.yaml` (2.45 GHz, 4 MS/s, 1.05 s CPI), gains from the
level check (`docs/hardware_setup.md`, section 4), `--n-cpi 5`.

Process every recording with
`python scripts/process_capture.py captures/<file>.sigmf-meta` (report and figures in `results/captures/<file>/`).

---

## E0 - Cable loop-back (receiver, sync and comm only, no radar)

**Set-up.** TX1 -> 50 dB attenuation -> RX1. **Purpose:** synchronisation, decoding, system delay, phase stability.

```bash
python scripts/pluto_capture.py config/indoor.yaml --level-check --tx-atten 10 --rx-gain 20
python scripts/pluto_capture.py config/indoor.yaml --tx-atten 10 --rx-gain <g> --n-cpi 5 --label E0_loopback
```

| Check | Expected | Measured |
|---|---|---|
| mean TX power at 10 dB and at 0 dB attenuation (power meter or spectrum analyser on TX1, channel power over 3.14 MHz) | about -15 / -5 dBm (assumed in `config/*.yaml`; enter the measured values there) | |
| sync metric | > 0.95 | |
| BER (about 5 M bits per CPI) | 0 | |
| EVM | < -25 dB | |
| direct-path delay (cable + chain) | constant over the 5 CPIs, +- 1 ns | |
| detections | none (static channel) | |
| residual direct-path phase drift per CPI (`--two-device` prints it) | small, stable | |

## E1 - Static scene (clutter floor and dynamic range)

**Set-up.** Antennas in place, nobody moving, fan off.

| Check | Expected | Measured |
|---|---|---|
| detections | none, or only at multiples of 24.4 Hz (if so: clipping -> lower the gain) | |
| floor below the zero-Doppler peak, \|f_D\| > 5 Hz | > 70 dB | |
| RX mean / peak level | about -15 / < -3 dBFS | |

## E2 - Pendulum (known Doppler law)

**Set-up.** Metal plate or corner reflector on a string. Measure the length l, the pivot position and the release angle.
Predicted Doppler: `python scripts/predict.py pendulum --tx ... --rx ... --pivot ... --length l --theta0 a --direction ...`

| Check | Predicted | Measured |
|---|---|---|
| period from the micro-Doppler plot (`--md-window-ms 64`) | 2 pi sqrt(l / g) | |
| peak \|Doppler\| | from `predict.py` | |

## E3 - Walking person (main indoor result)

**Set-up.** Tape a straight line on the floor (e.g. 6 m) towards the baseline. Time each walk with a stopwatch, or better, film it.
Record walks *towards*, *away* and *across* (tangential: expect near-zero bistatic Doppler).

```bash
python scripts/pluto_capture.py config/indoor.yaml --rx-gain <g> --tx-atten <a> --n-cpi 1 --label E3_towards_01
python scripts/predict.py walk --tx ... --rx ... --a <start> --b <end> --time <t>
```

| Walk | Speed (tape/stopwatch) | Predicted Doppler at mid-CPI | Measured torso Doppler | Error |
|---|---|---|---|---|
| towards 01 | | | | |
| towards 02 | | | | |
| away 01 | | | | |
| across 01 | | about 0 (in the notch) | | |

Also note: the limb micro-Doppler lines (step rate), and whether the comm BER stays at 0 while a person is in the beam.

## E4 - Desk fan (rotor / UAV proxy)

**Set-up.** Fan about 2-3 m from the baseline, axis roughly parallel to the baseline. Measure its rpm
independently (a phone stroboscope or tachometer app, or the data sheet).
Expected: `python scripts/predict.py rotor --blades <n> --rpm <rpm>`

```bash
python scripts/process_capture.py captures/<E4 file>.sigmf-meta --md-range 0 --md-window-ms 5 --rotor-band 80 550
```

| Speed setting | rpm (reference) | Expected line spacing n*f_rot | Estimated | Real-blade flash rate visible in the 5 ms spectrogram? |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

Note: the point-scatterer model predicts lines at multiples of n*f_rot. Real blades also produce
specular flashes, at n*f_rot for even n and at 2*n*f_rot for odd n. Report what you observe; do not adjust it to fit.

## E5 - Two PlutoSDRs (optional, true bistatic)

**Set-up.** TX node `--tx-only` (or `grc/pluto_tx_only.py`); RX node `--rx-only`, `grc/pluto_rx_record.py` or SDR++
(`sdrpp/README.md`); `xo_correction` calibrated. Repeat E3 (two walks). Process with `--two-device`.

| Check | One Pluto (E3) | Two Plutos, no correction | Two Plutos, `--two-device` |
|---|---|---|---|
| estimated CFO | 0 | n/a | |
| BER | | | |
| torso Doppler error | | | |

## E6 - Outdoor UAV (optional, only with a pilot and permission)

`config/outdoor_uav.yaml` (30.72 MS/s, 136.5 ms CPI), 12 dBi antennas, 10 m baseline, drone hovering, then flying a straight leg.
Record a GPS log from the drone as ground truth.

| Check | Truth (GPS) | Measured |
|---|---|---|
| excess range-sum at mid-CPI | | |
| range-sum rate | | |
| rotor line spacing vs. motor rpm (from the drone's telemetry if available) | | |
