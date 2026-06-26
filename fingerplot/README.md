
  TERZINA FPA — Binary Parser & Finger Plot Analysis

This script parses the .bin files from the TERZINA FPA
readout (4 DAQ x 5 ASIC x 32 CH = 640 channels),builds per-channel
ADC histograms, and produces finger plots and per-channel
statistics through an interactive menu.


PIPELINE
--------------------------------------------------------------
parse .bin -> histogram (16384 bins, optional rebin) -> Gaussian
smooth -> find_peaks -> Gaussian fit each peak -> gain = mean of
peak spacings, pedestal = first peak, etc.


HOW TO USE
--------------------------------------------------------------
1. Set BIN_FILE at the top of the notebook.
2. (Optional) tune NBINS, REBIN_FACTOR, SMOOTH_SIGMA,
   PEAK_PROMINENCE, PEAK_DISTANCE, MAX_PACKETS 
   for better visualization and enhancement
3. Run the cell. After parsing, type commands at the prompt.
   All plots are saved as PNG.


COMMANDS & PLOTS
--------------------------------------------------------------
GAIN defaults to HG. Replace with LG to use the low-gain branch.

  D A CH [GAIN]   (e.g. "1 A 30")
      Finger plot for one channel: log-y ADC spectrum, smoothed
      curve, detected peaks marked and annotated with p.e.
      number; prints gain (ADC/p.e.). Optionally followed by an
      8x4 grid of all 32 channels of that ASIC.

  all [GAIN]
      Full FPA overview: 2D map of ADC bin (x) vs channel 0-639
      (y), log-color counts. DAQ/ASIC boundaries drawn; dead
      channels show as blank rows.

  gain [GAIN]
      Gain distribution: histogram with Gaussian fit + per-
      channel scatter coloured by DAQ. IQR outlier rejection.

  pedestal [GAIN]
      Pedestal map: pedestal position per channel + distribution
      with Gaussian fit.

  resolution [GAIN]
      Energy resolution: sigma/mu vs p.e. with Poisson 1/sqrt(N)
      reference + sigma^2_signal vs p.e. linear fit (slope -> ENF).

  pvr [GAIN]
      Peak-to-valley ratio: per-channel scatter + distribution,
      with PVR = 2 reference.

  ext [GAIN]
      Extreme channels: finger plots of the channels with the
      highest ADC reach and highest gain.

  hglg   or   hglg D A CH
      HG vs LG correlation: 2D density + linear fit. Per-DAQ 2x2
      grid, or single channel (raw + pedestal-subtracted).

  q
      Quit.


CAVEATS
--------------------------------------------------------------
- The first detected peak is assumed to be the pedestal in
  pedestal/PVR/resolution routines — if it's actually noise or
  DCR, those metrics will be biased. Cross-check against the
  finger plot.


Author: Muhammad Abdullahi muhammad.abdullahi@gssi.it
