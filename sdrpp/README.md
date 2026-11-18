# Using SDR++ as the receiver

[SDR++](https://github.com/AlexandreRouma/SDRPlusPlus) is a receive-only SDR application. It cannot
play the cyclic OFDM transmit buffer. In this project it is used as the **RX node of a two-Pluto
set-up**: one PlutoSDR transmits (from this repository), a second PlutoSDR is opened in SDR++, and
SDR++ records a baseband WAV file. `scripts/process_capture.py` then synchronises, decodes and runs
the radar on that file.

> Status: the WAV import and the full processing chain are tested on simulated, SDR++-format
> recordings (`tests/test_core.py::test_sdrpp_wav_two_device`). The SDR++ GUI steps below were
> written for current SDR++ releases; menu names can differ slightly between versions and nightly builds.
> They have not yet been tried on hardware.

## Why two Plutos (and not SDR++ on the TX Pluto)

The AD9363 uses one sample clock for RX and TX. If SDR++ opened the same Pluto that is
transmitting, it would reconfigure the shared clock and LO settings under the running TX buffer.
Use a second Pluto for SDR++, or use the one-Pluto tools (`scripts/pluto_capture.py`,
`grc/pluto_isac_live.py`) instead.

## 1. Give the two Plutos different IP addresses

Both Plutos default to 192.168.2.1. On the RX Pluto, open the `PlutoSDR` USB drive, edit
`config.txt` (section `[NETWORK]`, e.g. `ipaddr = 192.168.3.1`, `ipaddr_host = 192.168.3.10`), eject
the drive and power-cycle the Pluto. Alternatively, address the Plutos by USB URI (`iio_info -s`
lists them).

## 2. Start the transmitter (Pluto A)

```bash
python3 scripts/generate_tx_buffer.py config/indoor.yaml
python3 scripts/pluto_capture.py config/indoor.yaml --tx-only --uri ip:192.168.2.1 --tx-atten 10
#   or, with GNU Radio:  cd grc && python3 pluto_tx_only.py
```

## 3. Configure SDR++ (Pluto B)

| SDR++ setting | Value | Why |
|---|---|---|
| Source | PlutoSDR, IP of Pluto B (e.g. `192.168.3.1`) | |
| Frequency | 2 450 000 000 Hz, no offset or frequency correction | same LO as the TX |
| Sample rate | 4 MHz (any rate the list offers is accepted if 4 MHz is missing, e.g. 5 MHz; it is resampled) | the waveform is defined at 4 MS/s |
| Bandwidth | about 4 MHz (occupied: 3.14 MHz) | |
| Gain mode | **Manual** (not slow/fast attack, not hybrid) | AGC steps inside a CPI look like targets |
| Gain | so that the waterfall does not clip; target about -15 dBFS mean | see `docs/hardware_setup.md`, section 4 |
| Decimation (source menu) | none | the recorder must see the full rate |
| IQ correction | on or off (either is fine) | |
| Recorder module | Mode **Baseband**, container **WAV**; sample type int16 or float32 if your version offers a choice | |

Record **at least 1.2 s** (the indoor CPI is 1.05 s). Longer is fine: the processing takes one
CPI (the configuration's 2^22 samples = 1.05 s after resampling) starting at `--start-s` (default 0 s);
choose another window with `--start-s` and `--cpi-s`.

## 4. Process

```bash
python3 scripts/process_capture.py ~/sdrpp/recordings/baseband_2450000000Hz_*.wav \
        --config config/indoor.yaml --two-device
```

* `--two-device` switches on the blind CFO estimate and the direct-path referencing: the two Plutos
  have independent TCXOs.
* The report lists sample drops. SDR++ streams over USB, so overflows are possible; the processing
  then keeps the longest gap-free segment and says so.
* If nothing is found and the spectrum looks mirrored, add `--swap-iq` (some tools store Q in the
  first WAV channel).

## 5. Frequency calibration (needed once per Pluto pair)

The CFO estimator covers +-7.8 kHz (half a subcarrier spacing). At 2.45 GHz, 25 ppm of crystal
error would already be about 61 kHz. If the report shows a CFO close to that limit, or the sync fails:

1. Measure the TX Pluto's carrier with a calibrated receiver or spectrum analyser, or use a GPS
   disciplined reference.
2. Correct each Pluto's reference with its `xo_correction` setting (the `xo_correction` attribute
   of the `ad9361-phy` device, which can also be set in the Pluto's `config.txt`). One ppm at the
   40 MHz reference is 40 Hz.
3. Repeat until the reported CFO is within a few hundred Hz.
