# Research assessment and next experiments

## Three papers, three different purposes

1. **Cosar et al., _Thermal Camera Based Physiological Monitoring with an Assistive Robot_ (2018), `sc18embc.pdf`.** This is the source of the five-person dataset in this repository: 382 x 288 images, about 27 Hz, two minutes per person, with one minute of stillness followed by head movements. The authors locate face and nose, resample irregular timestamps, apply a Hamming window and FFT, and use manually annotated inhalation times as respiration reference. Their reported respiration RMSE rises from 3.81 breaths/min while still to 6.20 while moving. This motivates time-aware processing, a tight nose crop, and separate still/moving evaluation.
2. **Kwon et al., _Breathing-Associated Facial Region Segmentation for Thermal Camera-Based Indirect Breathing Monitoring_ (2023), included PDF.** It constructs a four-second pixel variance map, normalizes it, then segments breathing-associated regions with a two-class MRF. Its 15-person, 640 x 480, 25 Hz FLIR dataset and BIOPAC reference are **different data** from this repository. The useful idea is to retain a spatial breathing attention map around the nose/upper lip, especially for non-frontal views. Its reported cycle accuracy and correlation must not be imported as performance for these five people.
3. **Adhikary et al., _JoulesEye: Energy Expenditure Estimation and Respiration Sensing from Thermal Imagery While Exercising_ (2023), `3631422.pdf`.** It uses nose ROI tracking and thermal respiration sensing during motion. Its deep network predicts **energy expenditure from respiration and other signals**; the 5.8% MAPE is energy-expenditure error, not respiration-rate error. Its discussion of occlusion and tracker failure supports adding a missing/quality channel and recovery logic. RGB landmark initialization is useful if synchronized RGB is available.

## Corrected dataset audit

| Person | Thermal frames | Reference rows | Marker-1 events | Full-CSV reference rate | Frame/CSV row match |
|---|---:|---:|---:|---:|---|
| Joao | 3,036 | 3,036 | 37 | 18.22/min | yes |
| anestis | 2,863 | 3,526 | 44 | 21.72/min | **no** |
| Claudio | 2,903 | 2,903 | 38 | 18.62/min | yes |
| Manuel | 2,700 | 2,700 | 38 | 18.68/min | yes |
| Jaime | 2,652 | 2,652 | 43 | 21.24/min | yes |

Equal row counts strongly suggest frame-level CSV correspondence for four people, but should still be checked against original bag timestamps if those become available. Joao's marker `2` alternates with marker `1`; only `1` is used as an inhalation event. For `anestis`, the overall reference rate can be calculated, but window labels are withheld until synchronization is recovered.

The interpretation of marker `2` as another phase of Joao's breath is inferred from the alternating sequence, the paper's inhalation-annotation description, and the resulting event intervals. It should be checked against the dataset's original annotation instructions before final benchmarking.

## Neural network path

### Measured first pilot

The spectral MLP has been run on the four recordings with matching frame/CSV counts. Its leave-one-person-out macro MAE is **4.80 breaths/min**; a training-median predictor scores **2.50 breaths/min** on exactly the same windows. The MLP underperforms in every fold. These are 77 overlapping test windows but only four independent held-out people, with one session each. The temporal CNN and CNN + GRU architectures are implemented; their experiment results are recorded separately below. This result rules out presenting a network as an accuracy improvement on the current data.

1. **Pilot now:** Extract 3 x 3 nasal patch intensity traces from the original frames. Preserve CSV timestamps and represent frozen runs as missing. Start with a small temporal CNN + GRU on 20-second windows, retaining the CNN as an architectural comparison. Predict rate from inhalation-event intervals. Use leave-one-person-out folds and compare with the median rate of training people. This checks whether learning adds value beyond the 18-21/min prior; the sample is too small for a reliable accuracy claim.
2. **Improve spatial localization:** Collect annotated face and nose boxes across stillness, head movement, profile view, and occlusion. Train or adapt a thermal face detector/landmark model; track across frames and re-detect after loss. Compare manually verified boxes, detector boxes, and the legacy broad box. Record failed detections rather than silently falling back to chest pixels.
3. **Model breathing directly:** With more synchronized people and sessions, use a frame encoder plus temporal convolution or transformer. Train against per-frame inhale event heatmaps or a reference waveform, then compute rates from the output. A spatial attention/segmentation head can be initialized with carefully reviewed BAFR pseudo-labels, but variance alone is not ground truth. Report event F1 with a time tolerance, waveform correlation when a reference waveform exists, and rate MAE per subject and condition.
4. **Face recognition:** Treat identity as a separate objective. One recording per person cannot establish recognition across days, camera angles, temperatures, or sessions. Obtain consented repeated sessions per identity. Train a thermal face embedding with contrastive or metric learning, then evaluate verification/identification on **held-out sessions**, including unknown-person rejection. For an open-set system, never force an unseen face into one of these five names. Keep identity split and respiration split explicit.
5. **Acquisition upgrade:** If original ROS bags exist, preserve `mono16` frames and timestamps. The current 8-bit grayscale files lose thermal precision. Add synchronized RGB only if the intended deployment has it; JoulesEye's RGB-assisted ROI procedure is not directly reproducible from these thermal-only files.

## Evaluation gates

- First verify event marker semantics and exact frame/reference synchronization. No training on `anestis` windows until its mismatch is resolved.
- Report both full-recording and windowed rates. Compute duration from timestamps, never from the count of retained frames divided by a nominal FPS.
- Separate first (still) and second (moving) minute; compare the model against a time-aware FFT/periodogram and training-median baseline.
- Split by person for respiration generalization, and by recording session for identity. Keep every overlapping window from the same person/session in one split.
- Report MAE and the five or fewer independent test people, failure/abstention rate, and bootstrap uncertainty over people when the dataset is enlarged. Do not present window count as independent sample size.
- Add a prospective test set with different subjects, camera distances, lighting/ambient temperature, breathing styles, occlusions, and face angles before making any reliability claim.

## Current implementation: CNN + GRU pilot

The first recurrent model uses the existing nasal traces. This is a temporal
1D CNN over nine intensity traces plus a missing-data mask, not a spatial CNN
trained on full images. A GRU is the selected recurrent unit, keeping the
initial model small for the four usable recordings.

Architecture (`thermal_face/models.py`):

- Input: batch x 10 channels x 200 samples (20 seconds at 10 Hz).
- Conv1D 10 -> 24, kernel 9, padding 4; ReLU; average pooling by 2.
- Conv1D 24 -> 32, kernel 9, padding 4; ReLU; average pooling by 2.
- Transpose to batch x 50 time steps x 32 features.
- One unidirectional GRU, input size 32, hidden size 32.
- Final hidden state -> Linear 32 -> 16 -> ReLU -> Linear 16 -> 1 BPM.

The complete window is required before predicting; the GRU state resets for
independent windows. The missing channel is an input feature, not a recurrent
state mask. Existing window filtering rejects windows with over 20% missing
samples. AdamW, Smooth L1 loss, 40 epochs and seed 7 match the initial CNN
training recipe. There is no tuning against the held-out person's labels.
Reports include architecture, parameter count, seed, epochs, training subjects,
per-window predictions and references, and the training-median comparator.
The loader excludes stale `anestis.npz` files as well as the extractor's exclusion.

Run from the repository root using the project Python 3.12 environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m thermal_face.train_rate --architecture cnn-gru --epochs 40 --seed 7
.\.venv\Scripts\python.exe -m thermal_face.train_rate --architecture cnn --epochs 40 --seed 7
```

Outputs default to `artifacts/rate_cnn_gru_pilot.json` and
`artifacts/rate_cnn_pilot.json`. These are experiment reports; the trainer does
not yet save deployment checkpoints.

### First CNN + GRU comparison

Completed with 40 epochs and seed 7 on the same 77 windows from four held-out
people. Each fold resets its random seed before model initialization. Runtime:
Python 3.12.5, NumPy 2.5.3, PyTorch 2.14.0+cpu, CPU.

| Method | Parameters | Macro MAE (breaths/min) |
|---|---:|---:|
| Training-median baseline | 0 | 2.501 |
| CNN + GRU | 16,009 | 2.986 |
| CNN | 18,921 | 3.577 |

| Held-out person | CNN + GRU MAE | CNN MAE | Median baseline MAE |
|---|---:|---:|---:|
| Claudio | 2.565 | 4.276 | 1.956 |
| Jaime | 4.241 | 3.830 | 2.626 |
| Joao | 1.653 | 3.104 | 2.146 |
| Manuel | 3.484 | 3.097 | 3.277 |

The recurrent model improves aggregate error over this CNN run but does not
beat the median baseline. It beats the baseline only for Joao. These are single
seed, exploratory results, not evidence of reliable generalization. The four
people, not the 77 overlapping windows, are the independent evaluation units.
Results are saved in `artifacts/rate_cnn_gru_pilot.json` and
`artifacts/rate_cnn_pilot.json`. Four unittest cases passed, including backward
propagation through convolution and recurrence, actual train/test forward-call
subject separation, report serialization, stale-trace exclusion, and the
existing marker interpretation regression.

### Further action, in order

1. Validate marker semantics and exact frame synchronization from original
   annotation instructions or ROS timestamps. Four recordings still use assumed
   row correspondence; `anestis` remains excluded until alignment is established.
2. Windowed evaluation is implemented in `thermal_face.evaluate`: shared
   timestamps, rejection reasons, still/moving/transition labels, coverage and a
   timestamp-aware spectral comparator. Explicit full-recording estimates remain
   a separate follow-up; current reports cover 20-second windows only.
3. Annotate and verify nose boxes during motion, add tracking and recovery, and
   measure how localization changes respiration error and valid coverage.
4. Repeat predetermined seeds with the same subject splits. Additional people
   and sessions are needed before drawing generalization conclusions. If the
   recurrent model fails to beat the baselines, prioritize data and localization
   before increasing model size.
5. With more synchronized recordings, evaluate a spatial frame encoder followed
   by recurrence and inhale-event supervision. Use waveform correlation only if
   an actual reference waveform becomes available.
6. Pursue identity embeddings after collecting repeated sessions per identity;
   evaluate held-out sessions and unknown-person rejection separately.

## Unified windowed evaluation

Run the complete comparison with:

```powershell
.\.venv\Scripts\python.exe -m thermal_face.evaluate --epochs 40 --seed 7
```

The default output is `artifacts/evaluation.json`; use `--out` and `--traces` to
change paths. `--still-until-s` sets the assumed protocol boundary (default 60).
The script trains both neural architectures from scratch in each subject fold,
using the same training helper as `train_rate`. It compares CNN, CNN + GRU,
training-median and spectral estimates. All three neural trainers, including the
MLP, now obtain their windows through `thermal_face/windows.py` (the legacy
`train_rate.load_windows` import remains available).

The report contains per-window timestamps, subject, condition, reference rate,
missing fraction, separate signal/reference rejection reasons and predictions.
It includes per-subject and per-condition errors, macro MAE across people, pooled
window MAE, training subjects, trace SHA-256 hashes, runtime versions and training
settings. `common_comparison` compares every method on the intersection of
windows where all methods returned predictions. Per-method results and coverage
are also retained, so an abstention cannot silently improve the comparison.
Empty groups have null metrics, not zero error.

Coverage denominators are all complete 20-second candidate windows at a
five-second stride. Evaluation eligibility requires both valid signal support
and usable reference events. Predictions are evaluated only on eligible windows;
coverage therefore includes reference-label screening and is not a measure of
live deployment availability. Each exclusion retains both its signal and
reference reason if both fail. There are no signal-quality confidence claims.

Conditions are inferred from elapsed time: windows ending at or before 60 seconds
are still, windows starting at or after 60 seconds are moving, and those crossing
the boundary are transition. These labels need verification against the actual
recording protocol. Transition windows participate in overall evaluation and
have their own condition results.

The spectral baseline uses the timestamp-resampled traces, linear detrending,
a Hann taper, and average normalized power across nonconstant nasal channels
in the fixed 6-45 BPM range. The missing mask is not treated as a nasal signal.
Eightfold zero padding samples the spectrum more densely but does not improve
the approximately 3 BPM raw frequency resolution of a 20-second observation.
Flat signals cause an explicit abstention; there is no tuned confidence cutoff.

### Measured unified evaluation (40 epochs, seed 7)

| Method | Overall macro MAE | Still | Moving | Transition |
|---|---:|---:|---:|---:|
| Training-median | 2.501 | 1.944 | 2.711 | 3.527 |
| Spectral | 10.618 | 7.192 | 14.046 | 12.771 |
| CNN | 3.577 | 2.787 | 3.832 | 5.452 |
| CNN + GRU | 2.986 | 2.286 | 3.573 | 3.885 |

All values are breaths/minute. All four methods produced predictions on the
same 77 eligible windows from four held-out people. Candidate coverage is 77/84
(91.7%): still 36/36, moving 29/36 and transition 12/12. All seven exclusions are
reference-related: six implausible event intervals and one window with fewer
than four marker-1 events. Signal filtering rejected no candidate windows in
this run. Label screening especially affects the moving-period results and
requires review before treating these numbers as a final benchmark.

The CNN + GRU remains worse than the median baseline; the spectral method is
also poor on these provisional fixed crops. These results support inspecting
reference events and localization before further architecture expansion; they
do not establish which factor causes the errors.

Verification: all 11 unittest cases passed. Synthetic irregularly timestamped
signals at 12, 18, 27 and 36 BPM were recovered within 0.75 BPM. Tests cover frozen
gaps, exact window endpoints, still/moving boundaries, invalid timestamps,
macro-versus-pooled error, common-window coverage, abstention, JSON reporting and
subject separation. Refactoring preserved all 77 original model inputs and
labels exactly, and both neural models reproduced their earlier predictions
exactly in the recorded environment.

## GPU training enabled and verified

The project environment now uses PyTorch 2.14.0+cu130 with CUDA runtime 13.0.
Both CNN and CNN + GRU were trained on `cuda:0`, the NVIDIA GeForce RTX 4060
Laptop GPU (8 GB), using 40 epochs per fold, seed 7 and the same four-person
leave-one-person-out protocol. Results are saved separately in
`artifacts/evaluation_gpu.json`, preserving the previous CPU report.

```powershell
.\.venv\Scripts\python.exe -m thermal_face.evaluate --device cuda --epochs 40 --seed 7 --out artifacts/evaluation_gpu.json
```

Both `train_rate` and `evaluate` accept `--device auto|cpu|cuda`. Automatic
selection chooses CUDA when available; an explicit CUDA request fails if the
GPU cannot be used. The model, input tensors, targets and minibatch indices
are placed on the selected device. Reports include the requested and actual
device, GPU name, PyTorch version and CUDA runtime. Trace preprocessing and
the spectral baseline remain on CPU.

GPU macro MAE: median baseline 2.501 BPM, spectral 10.618 BPM, CNN 3.577 BPM,
CNN + GRU 2.986 BPM. Coverage remains 77/84 candidate windows. These match the
CPU results at three decimal places; changing the compute device has not
established an accuracy improvement. CPU and GPU floating-point results are
not required to be exactly identical.

All 14 tests passed, including actual training and inference on CUDA for both
architectures, explicit CPU selection and failure rather than fallback when
CUDA is requested but unavailable. Evaluation reports are persisted; trained
weights still are not saved as checkpoints.

### Next implementation steps

1. Build `thermal_face/diagnostics.py` to produce a local HTML report with
   sampled frames and nose-box overlays, traces, reference events, predictions,
   rejected windows and the largest errors. Review the seven reference-rejected
   windows and inspect whether fixed crops remain on the nose during movement.
2. Correct independently verified label/alignment issues and implement nose
   tracking with explicit tracking-loss flags. Compare fixed and tracked crops
   on the same subject splits, including coverage.
3. Add per-fold checkpoint saving with architecture, preprocessing settings,
   training subjects and seed, so models can be reloaded for inference without
   retraining. Keep held-out evaluation weights separate from any model fitted
   on all available training data.
4. Repeat predetermined seeds after the data/localization changes. Keep the
   median and spectral comparators and collect additional people and sessions
   before treating this pilot as a generalizable model.
