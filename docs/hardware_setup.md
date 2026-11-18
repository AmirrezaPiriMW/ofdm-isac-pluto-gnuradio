# Hardware setup

This page describes how to build the PlutoSDR set-up, bring it up, and set the gains
before any measurement. The measurements themselves are in
[`experiment_protocol.md`](experiment_protocol.md).

## 1. Bill of materials

| Item | Recommendation | Why |
|---|---|---|
| SDR | ADALM-PLUTO (a second one for the two-Pluto / SDR++ mode) | AD9363, full duplex TX1/RX1, 12-bit ADC and DAC |
| Antennas, indoor | 2 x 2.4 GHz patch / panel, 6-8 dBi | The directivity reduces the direct TX-to-RX coupling and keeps the RX out of compression |
| Antennas, outdoor | 2 x 2.4 GHz panel or Yagi, 10-14 dBi | Link budget for small UAVs (see `docs/theory.md`, section 7) |
| Cables | Low-loss SMA (e.g. LMR-240 for a few metres, LMR-400 for 10 m) | Thin RG-174 loses roughly 1 dB/m at 2.4 GHz; check your cable's data sheet |
| Loop-back test | 30 dB + 20 dB SMA attenuators, one short cable | Experiment E0 (never connect TX to RX without attenuation) |
| Calibration target | Metal plate or corner reflector (about 20-30 cm) on a string (pendulum) | Moving target with a known Doppler law |
| Host | Linux PC with USB 2.0+ | GNU Radio 3.10 + gr-iio, libiio, Python 3.10+ |

## 2. Software installation (Ubuntu 22.04/24.04)

```bash
make setup                     # = scripts/setup_ubuntu.sh: GNU Radio 3.10 (incl. gr-iio), libiio, Python deps, .venv
source .venv/bin/activate
iio_info -s                    # the Pluto must appear (usb: or ip:192.168.2.1)
make test                      # 28 tests, about 2-3 minutes
make tx                        # TX buffers for the GNU Radio flowgraphs
```

Use a recent Pluto firmware release. The AD9363's default tuning range (325 MHz to 3.8 GHz) covers 2.45 GHz,
so the AD9364 frequency extension is not needed.

## 3. Geometry

**One-Pluto mode (start here).** TX1 and RX1 of the same Pluto drive two antennas 3-10 m
apart over low-loss cables. The cables are the synchronisation link: TX and RX share the
40 MHz reference, so there is no carrier or sampling-frequency offset. This is the same role
the fibre-optic synchronisation plays in a distributed MIMO testbed.

```
     TX antenna  <--- cable L_tx ---  [ PlutoSDR ]  --- cable L_rx --->  RX antenna
         |                              TX1   RX1                             |
         +------------------- baseline L (3-10 m) ----------------------------+
                          target: bistatic range sum R_t + R_r
```

* Point both antennas into the scene. Turn them slightly away from each other, or put an
  RF absorber between them, so that the direct path does not dominate the 12-bit ADC.
* Write down the antenna positions (x, y, z) to about 5 cm. `process_capture.py`
  reports excess range-sum and range-sum rate, and converting these to target position or
  speed needs the geometry.
* Measure the cable lengths. The processing calibrates the absolute delay on the direct path,
  so cable delays cancel. Writing them down still helps when you debug.

**Two-Pluto mode (optional).** One Pluto runs `pluto_capture.py --tx-only`, the other
`--rx-only`. Before measuring, calibrate both Plutos' `xo_correction` attribute so that the
residual offset is below about 5 kHz at 2.45 GHz: the blind CFO estimator covers +-df/2 =
+-7.8 kHz. Then process with `--two-device`; see `docs/theory.md`, section 5.

## 4. Gain staging (do this every time the scene changes)

The receiver has to cope with the direct path, static clutter and weak moving echoes at the
same time. AGC is off because a gain step inside a CPI looks like a modulated target.

1. Start with `--tx-atten 20` and `--rx-gain 20`.
2. `python scripts/pluto_capture.py config/indoor.yaml --level-check --tx-atten 20 --rx-gain 20`
3. Adjust `--rx-gain` (and, if needed, `--tx-atten`) until the **mean** level is about
   **-15 dBFS** and the **peak** stays below **-3 dBFS**. The TX buffer's peak-to-average ratio is
   11.1 dB, so at a higher mean level the ADC clips.
4. The valid manual RX gain at 1.3-4 GHz is -3 to 71 dB (1 dB steps). The TX attenuation is 0 to 89.75 dB
   (0.25 dB steps); according to Analog Devices, the default 10 dB keeps the output stage linear.
5. Why this matters: clipping is deterministic for a cyclic TX buffer, so it shows up as false
   Doppler lines at multiples of the buffer repetition rate (24.4 Hz indoors, 52.1 Hz outdoors).
   The simulator reproduces this effect (regression test
   `test_gain_staging_no_frame_rate_spurs`).

## 5. Regulatory and safety notes

* 2.45 GHz is in the 2400-2483.5 MHz ISM band. In the EU, wideband transmitters there are
  limited to 100 mW (20 dBm) EIRP (ETSI EN 300 328). The indoor set-up runs at about -15 dBm
  conducted + 6 dBi, i.e. about -9 dBm EIRP; the outdoor one at about -5 dBm + 12 dBi = 7 dBm EIRP
  (both are assumptions until E0 measures the TX power). Keep transmissions short, and check the rules
  that apply at your institution before transmitting outside a shielded room. The anechoic
  chamber is the cleanest place for E0-E2.
* The outdoor configuration occupies 2438-2462 MHz (24 MHz around 2.45 GHz), which is inside the band.
* UAV flights fall under EU drone rules (open category or authorisation). Fly only with a
  qualified pilot and the site owner's permission.
* Never connect TX directly to RX: use at least 50 dB of attenuation in loop-back.
