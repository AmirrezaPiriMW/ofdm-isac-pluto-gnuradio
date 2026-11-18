# Theory and design decisions

This page derives every processing step and every parameter in the project. All numbers come
from `isac/config.py` (`python -m isac.config config/*.yaml`) and from `results/metrics.json`;
the unit tests in `tests/` check the important ones.

## 1. Signal model

The transmitter sends OFDM symbols $m = 0, 1, \dots$ with $N$ subcarriers, spacing
$\Delta f = f_s/N$, cyclic prefix $N_{cp}$ and symbol period $T = (N + N_{cp})/f_s$. Subcarrier
$k$ of symbol $m$ carries $X[k,m]$: a Zadoff-Chu preamble at the start of every frame and QPSK
data otherwise. A point scatterer $p$ at transmitter distance $R_{t,p}$ and receiver distance
$R_{r,p}$ delays the signal by $\tau_p = (R_{t,p} + R_{r,p})/c$. As long as the excess delay fits in
the cyclic prefix and the scene is frozen during one symbol (stop-and-hop), the received
subcarrier symbol is

$$Y[k,m] = X[k,m]\sum_p a_p\, e^{-j2\pi (f_c + k\Delta f)\,\tau_p(mT)} + W[k,m].$$

Both conditions are checked against the numerology (section 3). The excess range-sum that fits in
the CP is 4.2 km indoors and 1.17 km outdoors. Body Doppler stays below 1 % of $\Delta f$, so
Doppler-induced inter-carrier interference (ICI) is negligible for the bodies. Rotor-blade tips
reach about 4 % of $\Delta f$; their ICI, $(\pi\varepsilon)^2/3 \approx -23$ dB relative to the
(already weak) blade echo, is not modelled.

## 2. OFDM radar: the reciprocal filter

The receiver knows $X[k,m]$ after decoding, so it divides it out:

$$H[k,m] = \frac{Y[k,m]}{X[k,m]} = \sum_p a_p\, e^{-j2\pi k\Delta f\,\tau_p}\; e^{-j2\pi f_c \tau_p(mT)} + \frac{W[k,m]}{X[k,m]}.$$

The payload disappears. What remains is the channel, sampled on a grid of subcarriers (fast
frequency) times symbols (slow time):

* an IFFT across $k$ turns the linear phase $e^{-j2\pi k\Delta f\tau}$ into a peak at delay $\tau$ (range-sum);
* an FFT across $m$ turns the carrier phase progression $e^{-j2\pi f_c \dot\tau mT}$ into a peak at the Doppler
  $f_D = -\frac{1}{\lambda}\frac{d(R_t + R_r)}{dt}$.

QPSK has constant modulus, so $|X| = 1$ and the reciprocal filter is the matched filter: it does not
amplify the noise. With 16-QAM it would, by about $E[1/|X|^2] = 1.9$ (2.8 dB) for unit-power 16-QAM.
That is one reason the default is QPSK. Reference: Sturm and Wiesbeck, *Proc. IEEE* 99(7), 2011.

**Decision-directed reference.** A bistatic receiver does not know the payload. The comm receiver
finds the frame (preamble correlation), estimates the channel from the preamble (least squares),
equalises, tracks the common phase error and decides the QPSK symbols. The re-modulated decisions
are the radar reference. When the bit error rate is zero, the result is identical to using the true
symbols (test `test_decision_reference_equals_genie_when_error_free`), so communication and sensing
share one waveform and one receiver.

**DC subcarrier.** DC is not transmitted (it would sit on the receiver's LO leakage). Left empty, this
one-bin hole in the Hann-windowed spectrum puts a flat range pedestal at
$20\log_{10}(1/\sum w)$, which is -52 dB for 800 subcarriers. The range transform therefore
interpolates $H$ at DC from the two neighbouring subcarriers (regression test `test_no_dc_pedestal`).

## 3. Numerology and why it was chosen

| Quantity | Indoor | Outdoor UAV | Formula |
|---|---|---|---|
| Sample rate | 4 MS/s | 30.72 MS/s | |
| FFT / CP | 256 / 64 | 1024 / 128 | |
| Subcarrier spacing | 15.625 kHz | 30 kHz | $f_s/N$ |
| Occupied bandwidth $B$ | 3.14 MHz | 24.0 MHz | $(2N_{half}+1)\Delta f$ |
| Range-sum resolution | 95.5 m | 12.5 m | $c/B$ |
| Slow-time rate | 12.5 kHz | 26.7 kHz | $1/T$ |
| CPI | 1.05 s (2^22 samples) | 136.5 ms (2^22 samples) | |
| Doppler resolution | 0.95 Hz | 7.33 Hz | $1/T_{CPI}$ |
| Range-sum-rate resolution | 0.117 m/s | 0.90 m/s | $\lambda/T_{CPI}$ |
| Unambiguous Doppler | +-6.25 kHz | +-13.3 kHz | $\pm 1/(2T)$ |
| 2-D coherent processing gain | 64.2 dB | 64.6 dB | $10\log_{10}(N_{used} N_{sym})$ |
| Cyclic TX buffer | 41.0 ms, repeats at 24.4 Hz | 19.2 ms, repeats at 52.1 Hz | |

**PlutoSDR constraints that set these numbers.**

1. *USB 2.0.* Continuous complex int16 streaming at $f_s$ needs $4f_s$ bytes/s. At 4 MS/s that is 16 MB/s, which streams
   reliably. At 30.72 MS/s it is 123 MB/s, which USB 2.0 cannot carry. The outdoor mode therefore plays TX from the Pluto's
   on-board **cyclic buffer**, so no TX data crosses USB, and records RX as **one kernel buffer** of $2^{22}$ samples, which is
   contiguous by construction. Each CPI is one buffer; consecutive CPIs are not contiguous.
2. *Indoor range resolution is useless (95 m), and that is accepted.* Indoors every target is in range cell 0, so
   targets are separated by Doppler alone. The long CPI buys 0.95 Hz Doppler resolution, which resolves a person's
   torso, limbs and a fan's rotor lines. Outdoors, 24 MHz gives 12.5 m range-sum cells: enough to separate a UAV from
   ground clutter and from people.
3. *Cyclic buffer = periodic waveform.* Any deterministic distortion (DAC/PA non-linearity, ADC clipping) repeats with the buffer
   and appears as Doppler lines at multiples of the buffer rate. The buffer holds 8 frames with different payloads, which pushes
   those lines to 24.4 Hz spacing and lowers each one by $10\log_{10}8 = 9$ dB. Gain staging (section 8) keeps them below the floor.

## 4. Bistatic geometry

With transmitter at $\mathbf t$, receiver at $\mathbf r$ and baseline $L = |\mathbf t - \mathbf r|$:

* excess range-sum: $R_t + R_r - L \ge 0$ (triangle inequality). Detections at negative excess range are sidelobes and are rejected;
* range-sum rate: $\dot R_\Sigma = \mathbf v\cdot(\hat{\mathbf u}_t + \hat{\mathbf u}_r)$, where $\hat{\mathbf u}$ are the unit vectors from transmitter and receiver to the target;
* Doppler: $f_D = -\dot R_\Sigma/\lambda$. Monostatic limit ($\mathbf t = \mathbf r$): $2v_r/\lambda$ (test `test_bistatic_doppler_...`).

$|\hat{\mathbf u}_t + \hat{\mathbf u}_r| = 2\cos(\beta/2)$, where $\beta$ is the bistatic angle. A target moving along the
bisector of a wide bistatic angle produces little Doppler, and a target crossing the baseline produces none. This is
the geometric reason a *multi-static* network helps: a second receiver sees the same motion through a different
bisector.

The absolute delay of cables and RF chains is unknown. The receiver therefore measures it on the direct path, located by an
interpolated peak search, and reports every target relative to it, so cable lengths cancel. This assumes the direct path is
the strongest *static* component. If a static reflector is stronger (antennas turned away from each other and towards a
wall), every excess range is offset by that reflector's excess range. `process_capture.py` prints the delay so that it can
be checked against the cable lengths.

## 5. Synchronisation

**One Pluto (shared clock).** TX and RX use the same 40 MHz reference, so there is no carrier frequency offset (CFO) or
sampling frequency offset (SFO). The AD9363 still has separate TX and RX synthesizers, so a small relative phase noise
remains; the direct-path referencing below can remove its slow part.

**Two Plutos (independent TCXOs).** A frequency error $\varepsilon$ (ppm) of one clock relative to the other causes

* a CFO of $\varepsilon f_c$: 4.9 kHz at 2 ppm and 2.45 GHz. This destroys both the link (BER 0.5) and the Doppler axis;
* an SFO of $\varepsilon$: the direct path drifts by $\varepsilon f_s T_{CPI}$ = 8.4 samples per 1.05 s CPI;
* uncorrelated LO phase noise.

Correction chain (`ProcessingOptions(cfo_correction=True, dp_referencing=True)`):

1. **Blind CFO estimate from the cyclic prefix** (van de Beek et al., 1997), averaged over the whole CPI:
   $\hat f = \frac{f_s}{2\pi N}\arg\sum_n r^*[n]\,r[n+N]$ over the CP samples. Its range is $\pm\Delta f/2$ (+-7.8 kHz indoors).
   Calibrate each Pluto's `xo_correction` so that the residual offset stays inside this range. Measured error in simulation: < 0.01 Hz.
2. **Direct-path referencing.** The direct path has zero bistatic Doppler. Any change in its delay or phase is therefore a
   synchronisation error, and the receiver removes it from the whole channel. The delay is fitted with a straight line over the
   CPI, which removes the SFO. The phase is low-pass filtered over `dp_smooth_s` (default 0.1 s), which removes residual CFO and
   slow phase noise.
   *Why smooth?* Indoors, the moving targets sit in the same range cell as the direct path. To first order, an unsmoothed
   per-symbol phase reference replaces the target echo $s$ by $\mathrm{Re}\{s\}$: it halves its power and creates a mirror at
   $-f_D$. Smoothing passes target Doppler above about $1/T_{smooth}$. Outdoors the targets are in other range cells,
   so the reference could follow the phase symbol by symbol.

Sensitivity (figure `fig05_two_device_sync.png`). For a Wiener phase-noise model of full linewidth $\beta$, the
single-sideband phase noise far from the carrier is $\mathcal L(f) \approx \beta/(2\pi f^2)$. Detection of the walker
survives up to $\beta$ = 0.3 Hz with 100 ms smoothing and up to 3 Hz with 20 ms smoothing (table in `results/metrics.json`).
An LO with about -54 dBc/Hz at 10 Hz offset, an assumed order of magnitude for a TCXO multiplied to 2.45 GHz, corresponds
to $\beta \approx 2.5\times10^{-3}$ Hz: about two orders of magnitude inside the working region. Measure your Plutos (E0, E5).

## 6. Clutter suppression, CFAR, estimation

* **Zero-Doppler suppression** subtracts the slow-time mean per subcarrier. This removes the direct path and all static clutter. The
  notch is about one Doppler cell wide, so targets slower than one cell are attenuated and biased away from zero. In the outdoor
  simulation a person at 3.7 Hz (0.5 cell) is reported at 7.0 Hz. Longer CPIs or adaptive clutter cancellation (e.g. ECA, as in
  passive radar) would reduce this.
* **Windows.** Hann in both dimensions: -31 dB first sidelobe, about 1.8 dB loss per dimension. This is the main reason the
  measured CFAR SNR lies below the link-budget prediction (section 7).
* **CFAR.** Ordered-statistic CFAR (median of the training cells; threshold from
  $P_{fa} = \prod_{i=0}^{k-1}\frac{n-i}{n-i+\alpha}$, Rohling 1983). A walking person or a UAV is a Doppler-*extended* target:
  limb or rotor echoes fill the training cells and raise a cell-averaging estimate until the body line itself is masked. This
  happened in development, and OS-CFAR fixed it. The interpolated map has correlated bins, so the training footprint takes one
  bin per resolution cell. The measured $P_{fa}$ is within a factor of 3 of the design value for both OS and CA (test
  `test_cfar_false_alarm_rate`). With only a handful of range cells (indoors), the CFAR runs in one dimension, along Doppler.
* **Estimation.** Parabolic interpolation of the log-power peak in range and Doppler. Detections closer than one Hann main lobe
  (2 cells) are merged.

## 7. Link budget (bistatic radar equation)

$$P_r = \frac{P_t G_t G_r \lambda^2 \sigma}{(4\pi)^3 R_t^2 R_r^2}, \qquad \mathrm{SNR}_{out} = \frac{P_r}{kTBF}\,N_{used}N_{sym}.$$

<!-- LINKBUDGET:BEGIN (generated by scripts/make_results_table.py) -->
| Target (assumptions in `config/*.yaml`) | Echo power | SNR per sample | SNR after processing | Below direct path |
|---|---|---|---|---|
| Indoor walker (0.5 m^2 torso + limbs, about 7 m) | -89.6 dBm | 12.3 dB | 76.5 dB | 14.4 dB |
| Indoor fan (0.02 m^2, about 3 m) | -91.7 dBm | 10.3 dB | 74.5 dB | 16.4 dB |
| Indoor static clutter (sum) | -66.2 dBm | 35.7 dB | 99.9 dB | -9.1 dB |
| Outdoor UAV (0.028 m^2 incl. rotors, about 95 m) | -126.8 dBm | -33.7 dB | 30.9 dB | 60.6 dB |
| Outdoor person (about 50 m) | -102.7 dBm | -9.6 dB | 55.1 dB | 36.5 dB |

Assumed mean TX power (section 8): -15 dBm indoors, -5 dBm outdoors. Direct path: -75.3 dBm indoors, -66.2 dBm outdoors. Indoors the clutter is stronger than the direct path and the system is limited by dynamic range (phase noise, ADC quantisation, residual direct-path variation), not by thermal noise.

Monte Carlo (12 random tracks per distance, `fig07_monte_carlo_uav.png`), detection probability: 50 m: 1.00, 100 m: 1.00, 150 m: 1.00, 200 m: 1.00, 250 m: 0.25, 300 m: 0.08, 400 m: 0.00. Where at least half the tracks are detected, the measured CFAR SNR is 0.4 to 4.4 dB below the link-budget prediction (Hann windows, about 3.5 dB, plus straddle loss); where few tracks are detected, their SNR is biased upwards (selection bias).
<!-- LINKBUDGET:END -->

The Doppler RMSE is larger at 50 m than at 100-200 m because close, fast tracks accelerate along the line of sight: over
one 136.5 ms CPI their Doppler changes by a median of 1.9 Hz and by up to 6.3 Hz (0.86 cell) for the Monte Carlo track
distribution, which widens and skews the peak.

## 8. TX level and gain staging

**TX level.** The AD9363 DAC is 12-bit; the HDL takes 16-bit samples and drops the 4 LSBs, and Analog Devices recommends
running the DAC close to full scale and setting the power with the attenuator
([ADI: Pluto transmit](https://wiki.analog.com/university/tools/pluto/users/transmit)). Both host paths therefore write the
same int16 samples with a peak of $32768\cdot10^{-1/20}$ per rail: pyadi-iio passes them unchanged, and gr-iio's PlutoSDR
Sink multiplies its float input by 32768 (gr-iio 3.10.9.2, `fmcomms2_sink_impl.cc`). The OFDM RMS is then about 12 dB below
a full-scale complex tone. With the data-sheet maximum of 7.5 dBm (1 MHz tone at 2.4 GHz, AD9363 Rev. D), the mean OFDM
power is about -4.5 dBm at 0 dB attenuation (outdoor configuration: -5 dBm) and about -15 dBm at ADI's default 10 dB
attenuation (indoor configuration). These are assumptions until E0 measures them; 1 dB less power lowers every SNR by 1 dB
and scales detection ranges by $10^{-1/40}$.

**Gain staging.**
The ADC sees the sum of the direct path, the clutter and the targets. In the indoor scene the room clutter is 9 dB
*stronger* than the direct path, so a gain set on the direct path alone would clip. The simulator and the capture script
therefore set the gain on the **total** received power (-15 dBFS mean). With the peak-to-average ratio of the TX buffer
(11.1 dB indoor, 10.9 dB outdoor), that leaves about 4 dB of headroom. This was the cause of a real bug during development: the frame-rate spurs described
in section 3.

## 9. Micro-Doppler and rotor rate

A blade point at radius $r$ rotating at $f_{rot}$ has Doppler $f_D(t) \propto r\,2\pi f_{rot}\cos(2\pi f_{rot}t + \phi)$,
a sinusoid. $N_b$ identical blades make the echo periodic with period $1/(N_b f_{rot})$, so its long-CPI spectrum is a
comb of lines ("JEM lines") at $f_{body} + nN_bf_{rot}$. The project estimates $N_bf_{rot}$ from the line spacing with a
comb-minus-anti-comb score. The anti-comb samples between the teeth at fractions 1/4, 1/3, 1/2, 2/3 and 3/4: a plain
comb also scores high at multiples of the true spacing (it did in development, at 3x the true 60 Hz), and at
sub-multiples. A spectrogram/cadence-velocity-diagram approach was rejected: resolving the blade period in time needs
windows shorter than half the period (8 ms for a fan, 2.5 ms for a UAV), which ruins the Doppler resolution.

Limitation: real blades also produce specular *flashes* (at $N_bf_{rot}$ for even $N_b$ and $2N_bf_{rot}$ for odd), which
the point-scatterer model does not include. The hardware experiment E4 checks which one appears.

## 10. What the simulation does not model

Antenna patterns (isotropic gains are assumed), multipath from walls and the ground, body shadowing and aspect-dependent
RCS, blade flashes, Doppler ICI of blade tips, TX/PA non-linearity, IQ imbalance, LO leakage, the AD9363's AGC and filters,
and USB sample drops (the capture design avoids these). These are exactly the effects the hardware experiments measure.
