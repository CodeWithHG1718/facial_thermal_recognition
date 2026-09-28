# Thermal face respiration research

Goal: recognize a face and estimate that person's respiration rate from thermal video. This repository currently contains **five people, one recording each**. It is a research pilot; it does not yet contain a validated recognition system or a proven respiration model.

## What was wrong with the previous result

The saved legacy result reports 15.40 breaths/min MAE from peak counting. It is not a trustworthy model comparison:

- The original dataset is recorded at about **27 Hz** with per-frame CSV timestamps. Legacy code assumes 25 fps and renumbers frames after deleting frozen runs, compressing elapsed time.
- The legacy face box includes much of the chest and sometimes starts at the nose. A mean over this box is not a clean nasal temperature trace.
- Joao has marker `1` and marker `2` events. The old evaluator counts both as breaths, giving 36.4 breaths/min. Counting only marker `1` gives **18.22 breaths/min** for the full CSV.
- The old FFT uses a 256-sample Welch segment: its frequency bins are about 5.86 breaths/min apart at 25 Hz. All five reported FFT estimates round to 11.7 breaths/min.
- This dataset's respiration reference is **manually annotated inhalation times**, according to Cosar et al. It is not a BIOPAC chest belt recording. The BIOPAC setup belongs to the separate Kwon et al. paper.
- `anestis` has 2,863 thermal frames but 3,526 reference rows. Frame-to-reference alignment cannot be assumed for this recording.

Run `python -m thermal_face.data --out artifacts/data_audit.json` to regenerate the audit. The old 15.40 figure is kept in `legacy/gt_comparison_report.txt` for provenance and must not be compared to a new model as a corrected benchmark.

## Layout

- `Joao/`, `anestis/`, `Claudio/`, `Manuel/`, `Jaime/`: original frames, retained locally.
- `*_respiration.csv`, `*_heartbeat.csv`: original reference files.
- `thermal_face/`: current data audit, feature extraction, and temporal neural network pilot.
- `config/nose_boxes.json`: provisional nose crops inspected on each existing subject.
- `docs/research_strategy.md`: paper synthesis, evaluation protocol, and roadmap.
- `legacy/`: old scripts and reports, retained as a record of the prior pipeline.
- `artifacts/`: generated reports and compressed traces; ignored by Git.

The provisional boxes in `config/nose_boxes.json` are dataset-specific annotations. They must be replaced by a detector and tracker before testing new people or camera positions.

The measured spectral MLP pilot achieved **4.80 breaths/min macro MAE** across four held-out people, while a training-median predictor achieved **2.50**. See `artifacts/rate_mlp_pilot.json`. This does not establish a useful learned model; it identifies the data and localization work needed before expanding model size.

The initial CNN + GRU run (40 epochs, seed 7) achieved **2.986 breaths/min
macro MAE**, compared with **3.577** for the CNN and **2.501** for the median
baseline on the same four people and 77 windows. The recurrent model does not
yet beat the simple baseline. See `docs/research_strategy.md` for per-person
results, architecture details, limitations, and the next experiments.

## Run the pilot

Use Python 3.10-3.12 in a virtual environment. Install NumPy and Pillow for extraction and the spectral MLP. PyTorch is needed for the temporal CNN and CNN + GRU experiments. `requirements.txt` lists all three.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m thermal_face.data --out artifacts/data_audit.json
python -m thermal_face.features --out artifacts/traces
python -m thermal_face.train_mlp --traces artifacts/traces --out artifacts/rate_mlp_pilot.json
python -m thermal_face.train_rate --architecture cnn-gru --epochs 40 --seed 7
python -m thermal_face.train_rate --architecture cnn --epochs 40 --seed 7
python -m thermal_face.evaluate --epochs 40 --seed 7
python -m unittest discover -s tests -v
```

The feature extractor reads the original frame folders and retains CSV time. It flags long identical-frame runs as missing, resamples the remaining samples, and records a missing-data channel. It refuses to use `anestis` until its reference alignment is established. The neural pilots use nine nasal patch traces in 20-second windows and report **leave-one-person-out** MAE alongside a training-median baseline. `train_mlp` is a small spectral MLP; `train_rate` defaults to a temporal CNN + GRU; `--architecture cnn` selects the original CNN comparison. Each architecture writes a separate report with its seed, epochs, parameter count, and per-window predictions. The recurrent model uses two convolution/pooling blocks, a 32-unit GRU, and a small rate-regression head. Overlapping windows are correlated; the four held-out people, not the number of windows, determine how much evidence the result provides. Training may underperform the baseline and should be reported as such.

## Evaluate all methods

`python -m thermal_face.evaluate` writes `artifacts/evaluation.json`, comparing
CNN + GRU, CNN, a timestamp-aware spectral estimate and the training-median
baseline. It reports still/moving/transition errors, per-subject metrics,
per-window predictions, coverage and exclusion reasons. The 60-second movement
boundary is assumed from the protocol and configurable with `--still-until-s`.

At 40 epochs and seed 7, all methods cover 77/84 candidate windows (91.7%).
Macro MAE is 2.501 BPM for the median baseline, 2.986 for CNN + GRU, 3.577 for
CNN and 10.618 for the spectral estimate. Seven windows fail reference-event
checks; coverage includes label screening and does not describe live availability.
See `docs/research_strategy.md` for the condition breakdown and remaining gates.

## Train on the NVIDIA GPU

The trainer and evaluator accept `--device auto|cpu|cuda`. The default `auto`
uses CUDA when available. Explicit `cuda` fails if CUDA is unavailable rather
than switching to CPU. Reports record the actual device, GPU name, PyTorch
version and CUDA runtime.

For this machine's RTX 4060, install the CUDA build in the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade "torch==2.14.0+cu130" --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m thermal_face.evaluate --device cuda --epochs 40 --seed 7 --out artifacts/evaluation_gpu.json
```

To run only CNN + GRU:

```powershell
.\.venv\Scripts\python.exe -m thermal_face.train_rate --architecture cnn-gru --device cuda --epochs 40 --seed 7 --out artifacts/rate_cnn_gru_gpu.json
```

The CUDA package comes from the [official PyTorch wheel index](https://download.pytorch.org/whl/cu130/torch/).
CPU remains selectable with `--device cpu`. The models and all neural training
and inference tensors use the selected device; trace loading and the spectral
baseline use NumPy on CPU. CPU and CUDA numerical results need not be identical.
Training currently saves evaluation reports, not model checkpoints.
Both commands print flushed startup, device, fold and epoch progress (first, every
10th, and final epoch), with training loss and elapsed time. Ctrl+C stops the run
with a concise interruption message; rerunning starts training again.

## Data handling

Original frames and CSVs are local data and excluded by `.gitignore`. Do not publish face recordings, identity labels, or trained recognition embeddings without confirming participant consent and dataset terms. The image files are 8-bit RGB containers with identical grayscale channels; they are not recoverable 16-bit temperature measurements. Do not convert these intensities into degrees Celsius.
