# 📊 Dataset & Methodology Analysis Report

**Project:** Breathing-Associated Facial Region Segmentation for Non-Contact Thermal Breathing Monitoring  
**Paper:** *Breathing-Associated Facial Region Segmentation for Thermal Camera-Based Indirect Breathing Monitoring*  
**Journal:** IEEE Journal of Translational Engineering in Health and Medicine, Vol. 11, 2023  
**Authors:** Junhwan Kwon, Oyun Kwon, Kyeong Taek Oh, Jeongmin Kim, Sun K. Yoo (Yonsei University)

---

## 1. 📖 Paper Overview & Objective

### What Problem Does the Paper Solve?
Traditional thermal-camera breathing monitors locate and track the **structural opening of nostrils** (physical holes). This approach fails in three real-world scenarios:

1. **Non-frontal camera angles** (45° or 90° profile) — nostril holes are not visible.
2. **Frontal view with obscured nostrils** — due to nose geometry or camera height, holes may not be detectable.
3. **Inter-subject anatomical variation** — nostril shapes vary widely between people.

### Proposed Solution
The paper proposes a **physiological feature-based unsupervised segmentation** approach that detects the **Breathing-Associated Facial Region (BAFR)** — the broader perinasal area (nostril rim, upper lip, cheeks) where respiration airflow causes periodic temperature fluctuations — using a **Markov Random Field (MRF)** segmentation framework.

> [!IMPORTANT]
> This approach does **NOT** require nostril detection, manual annotation, or deep learning training. It is fully unsupervised and view-independent.

---

## 2. 🗂️ Your Dataset — Detailed Analysis

### 2.1 Dataset Overview

| Subject | Bag File | Total Frames | Duration | Avg Frame Size |
|---------|----------|-------------|----------|----------------|
| **Joao** | `2016-11-22-14-09-37-Joao_onlythermal.bag` | 3,036 | ~121 s | ~155 KB |
| **anestis** | `2016-11-22-14-13-46-anestis_onlythermal.bag` | 2,863 | ~115 s | ~159 KB |
| **Claudio** | `2016-11-22-14-23-08-Claudio_onlythermal.bag` | 2,903 | ~116 s | ~155 KB |
| **Manuel** | `2016-11-22-14-29-29-Manuel_onlythermal.bag` | 2,700 | ~108 s | ~156 KB |
| **Jaime** | `2016-11-22-14-39-46-Jaime_onlythermal.bag` | 2,652 | ~106 s | ~158 KB |
| **TOTAL** | 5 subjects | **15,154 frames** | ~566 s total | — |

### 2.2 Image Technical Specifications

| Property | Value |
|----------|-------|
| **Resolution** | 382 × 288 pixels |
| **Original Encoding** | `mono16` (16-bit grayscale thermal) |
| **Saved Format** | 8-bit grayscale PNG (normalized from 16-bit) |
| **Camera Topic** | `thermal_image` (ROS sensor_msgs/Image) |
| **Estimated Frame Rate** | ~25 fps (consistent with paper's FLIR camera) |
| **Pixel Mean Intensity** | 63–73 (low, dark thermal background) |
| **Pixel Std Dev** | 54–60 (high dynamic range — good thermal contrast) |

> [!NOTE]
> The paper's camera is a FLIR A655sc at 640×480 resolution @ 25 Hz. Your dataset uses a different/similar FLIR camera at 382×288 pixels. The lower resolution is expected for older FLIR models (e.g., FLIR Lepton or A300 series). The methodology is fully applicable to this resolution.

### 2.3 ⚠️ Critical Data Quality Issue: Frozen Frames

**A major artifact was detected in every subject's data: frozen frame runs (sequences of frames with byte-for-byte identical pixel content).**

#### Frozen Frame Detection Results:

| Subject | # Frozen Runs | Total Frozen Frames | % of Total |
|---------|--------------|---------------------|------------|
| Joao | 7 | ~196 frames | 6.5% |
| anestis | 8 | ~216 frames | 7.5% |
| Claudio | 8 | ~213 frames | 7.3% |
| Manuel | 8 | ~197 frames | 7.3% |
| Jaime | 8 | ~208 frames | 7.8% |
| **Total** | **39 runs** | **~1,030 frames** | **~6.8%** |

#### Frozen Frame Locations (Joao subject as example):

| Run | Start Frame | End Frame | # Frozen | Notes |
|-----|------------|----------|----------|-------|
| 1 | 345 | 375 | 31 | ~1.2s frozen |
| 2 | 784 | 814 | 31 | ~1.2s frozen |
| 3 | 1190 | 1219 | 30 | ~1.2s frozen |
| 4 | 1581 | 1608 | 28 | ~1.1s frozen |
| 5 | 1947 | 1973 | 27 | ~1.1s frozen |
| 6 | 2339 | 2362 | 24 | ~1.0s frozen |
| 7 | 2686 | 2710 | 25 | ~1.0s frozen |

> [!WARNING]
> These frozen runs likely correspond to **camera buffer overflows or USB data transfer drops** during the bag recording. The frames are not corrupted — they simply repeat the last valid frame. If not removed, they will **corrupt the BAF variance computation** by artificially lowering variance to zero in those windows.

### 2.4 What Data is Useful

| Data | Status | Reason |
|------|--------|--------|
| ✅ Normal thermal frames (~14,124 frames) | **USE** | Valid thermal data with breathing-correlated pixel variation |
| ✅ Frames showing face/nose region | **USE** | Primary signal source for BAFR detection |
| ❌ Frozen frames (~1,030 total) | **REMOVE** | Zero temporal variance destroys BAF computation |
| ❌ Transition frames (first 1-2 frames after frozen run) | **REMOVE** | May contain transition artifacts |
| ⚠️ Frames at sequence start/end | **VALIDATE** | Some camera warm-up drift may exist |

**All 5 subjects are useful.** The dataset represents good cross-subject variability for a pilot study. For the paper's pipeline (unsupervised per-frame), no label splitting is needed.

---

## 3. 🧹 Data Cleaning Techniques & Justifications

### Step 1: Frozen Frame Detection & Removal

**Technique:** Detect consecutive frames with identical byte content (same file size AND identical pixel hash).  
**Why:** Frozen frames have zero temporal variance (σ = 0 for all pixels) in any temporal window that overlaps them. Since the entire BAF (Breathing-Associated Feature) computation depends on temporal variance, frozen frames will force BAF → 0, creating **false negatives** (no breathing detected) and corrupting the BAP probability map.  
**Implementation:** Flag frame index ranges where file size is identical for > 5 consecutive frames, or use numpy array diff equality check on loaded pixel arrays.

```python
# Pseudocode
if np.array_equal(frame[t], frame[t-1]):
    frozen_count += 1
else:
    if frozen_count > 5:
        mark frames [t-frozen_count, t] as FROZEN
    frozen_count = 0
```

---

### Step 2: 16-bit to Float Thermal Normalization

**Technique:** Convert extracted 8-bit PNG (which was normalized from mono16 during bag extraction) back to meaningful thermal range, OR re-extract at 16-bit for maximum precision.  
**Why:** The paper uses temperature values (Kelvin) in the BAF variance calculation. Our 8-bit conversion loses precision in the thermal quantization. For serious implementation, re-extract at 16-bit (`uint16`) from the bag files directly.  
**Implementation:** Load PNG as 16-bit or modify extraction to save as 16-bit TIFF.

---

### Step 3: Temporal Smoothing with Moving Window (BAF Computation)

**Technique:** Apply a **rolling standard deviation** over a 4-second window (= 100 frames at 25 fps) at each pixel position.  
**Why:** Human breathing cycles last approximately 3–5 seconds (12–20 breaths/min). A 4-second window captures exactly 1 full breath cycle's worth of signal. This window size maximally amplifies the breathing-related temperature oscillation while averaging out non-periodic (cardiac, motion) noise. Using a shorter window would miss slow breaths; longer would average out fast ones.  
**Formula:**
$$BAF(x, y, t) = \sqrt{\frac{1}{N_t - 1} \sum_{k=0}^{N_t - 1} (T_{x,y}[k] - \mu_{T_{x,y}})^2}$$
where $N_t = 100$ frames.

---

### Step 4: Spatial ROI Cropping (Face/Nose Region)

**Technique:** Detect and crop the face region using **Viola-Jones face detector** (Haar cascade) or **MediaPipe FaceMesh** to localize the nose/perinasal area as the primary ROI before processing.  
**Why:** Processing the full 382×288 frame would include background walls, clothing, and non-facial regions which add noise to the BAP computation and slow processing. The paper uses a manually-defined initial ROI tracked by KLT. For automation, a face detector initializes the ROI.  
**Why this data specifically:** At 382×288 resolution, the face occupies a smaller fraction of the frame — tight ROI extraction is even more important to preserve spatial resolution for fine perinasal region analysis.

---

### Step 5: Motion Compensation (KLT Optical Flow Tracking)

**Technique:** Apply **Kanade-Lucas-Tomasi (KLT) sparse optical flow** feature tracking to detect and compensate for frame-to-frame head displacement.  
**Why:** Even small head movements (nods, lateral drift) cause pixels on the nostril area to shift. Without tracking, the temporal variance at a fixed pixel coordinate would capture *motion variance* rather than *breathing thermal variance*, creating false high-BAF regions on moving edges. The KLT tracker propagates the ROI bounding box to follow the face, ensuring the same anatomical region is analyzed across frames.  
**Why this data specifically:** Since the recording setup is not a clinical bed-mounted system, subjects naturally make small postural adjustments over the ~2-minute sequences.

---

### Step 6: Temporal Gap Handling at Frozen Frame Boundaries

**Technique:** When a frozen run (e.g., 31 consecutive frames) is removed, interpolate the missing timestamp gap **in the pixel domain** using linear interpolation between the last valid frame before and first valid frame after the frozen run.  
**Why:** The BAF computation uses a continuous rolling window. A hard cut (removing 31 frames) creates a temporal discontinuity that would produce a spike in the BAF map at the seam, creating a false BAFR region. Interpolating ~31 frames of smooth transition prevents this artifact.  
**Alternative:** Simply **trim sequences** to avoid frozen-frame windows — exclude any 100-frame BAF computation window that overlaps with a frozen segment.

---

### Step 7: Pixel Intensity Range Normalization

**Technique:** Per-frame normalization of pixel values to a consistent range (e.g., z-score normalization or min-max to [0,1]).  
**Why:** The thermal scene temperature drifts slowly over time as the camera sensor itself warms up (thermal drift) and as the subject's skin temperature equilibrates with the environment. This slow global drift shifts all pixel values upward/downward without carrying breathing information. Subtracting a running mean or z-scoring within each window removes this DC drift component.  
**Why this data specifically:** The pixel mean varies between subjects (63–73), indicating different ambient temperatures or camera gain settings across recording sessions. Normalization ensures consistent BAF behavior across all 5 subjects.

---

## 4. 🔬 Full Technical Pipeline (Paper Replicated)

```
Raw .bag Files
      │
      ▼
[Step 1] Extract PNG frames (done ✓)
      │
      ▼
[Step 2] Reload as 16-bit / Normalize to float32 thermal range
      │
      ▼
[Step 3] Detect & Remove Frozen Frame Runs (~6.8% of data)
      │
      ▼
[Step 4] Face Detection → Initialize Perinasal ROI
      │
      ▼
[Step 5] KLT Optical Flow → Track ROI across frames
      │
      ▼
[Step 6] BAF Computation (rolling 100-frame std dev per pixel)
      │
      ▼
[Step 7] BAP Map = Normalize BAF by max(BAF) per frame → [0,1]
      │
      ▼
[Step 8] MRF Segmentation (2-class, 30 ICM iterations)
              → Binary BAFR Mask
      │
      ▼
[Step 9] Breathing Waveform = Spatial mean of differential
          thermal values within BAFR mask
      │
      ▼
[Step 10] Peak Detection → Breathing Rate Estimation
      │
      ▼
[Step 11] Evaluate vs ground truth (if BIOPAC belt available)
          Metrics: Cycle Detection Accuracy, Pearson r
```

---

## 5. 🧠 Model & Algorithm Details

### 5.1 Markov Random Field (MRF) Segmentation

**What it is:** A probabilistic graphical model that classifies each pixel based on:
1. Its own BAP value (data term / likelihood)
2. The labels of its neighbors (smoothness / spatial coherence term)

**Why MRF (not deep learning):**
- **No training data needed** — our dataset has no pixel-level breathing mask labels
- **Physiologically motivated** — the segmentation is driven by actual measured thermal variance
- **View-independent** — no assumption about face structure or nostril location
- **Per-sequence adaptive** — each subject's BAFR is computed from their own data

**MRF Energy Function:**
$$E(\mathbf{l}) = \sum_{i} E_{data}(l_i) + \lambda \sum_{i,j \in \mathcal{N}} E_{smooth}(l_i, l_j)$$

- $E_{data}(l_i) = -\log P(BAP_i | l_i)$ — how well pixel BAP matches class $l_i$
- $E_{smooth}$ — Potts model penalty for neighboring pixels with different labels
- $\mathcal{N}$ — 4-connected or 8-connected neighborhood

**ICM Optimization:** Iteratively updates each pixel's label to minimize local energy, conditioned on current neighbor labels. Converges in ~30 iterations.

### 5.2 KLT Feature Tracking

**What it is:** Sparse optical flow that tracks a set of good feature points (corners) from frame to frame using Lucas-Kanade differential estimation.

**Why for this data:** It is computationally lightweight (runs at >25 fps on CPU), doesn't require a GPU, and handles the small-amplitude head movements typical in seated recording scenarios.

### 5.3 BAF Temporal Variance

**Why 4-second / 100-frame window:**
- Normal resting breathing rate = 12–20 breaths/min → period = 3–5 seconds
- A 4-second window captures exactly 1 full breathing oscillation
- Variance over exactly 1 cycle maximally amplifies the periodic signal
- Shorter windows → miss slow breaths; Longer → attenuate fast breaths

---

## 6. 📉 Evaluation Metrics We Will Use

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **Breathing Cycle Detection Accuracy** | `(detected_cycles / reference_cycles) × 100%` | How many breaths are correctly identified |
| **Pearson Correlation Coefficient (r)** | $r = \frac{Cov(x,y)}{\sigma_x \sigma_y}$ | Waveform shape similarity with ground truth |
| **Mean Absolute Error (MAE)** in bpm | `|estimated_rate - reference_rate|` | Breathing rate estimation error |
| **BAFR Dice Score** | `2|A∩B| / (|A|+|B|)` | Segmentation mask overlap (if manual labels available) |

> [!NOTE]
> The paper uses a BIOPAC MP150 chest belt as ground truth at 1000 Hz. If your bag files only have thermal data (no BIOPAC), evaluation will be limited to visual waveform inspection and self-consistency metrics.

---

## 7. 📋 Summary: Why These Techniques on This Data

| Technique | Applied To | Justification |
|-----------|-----------|---------------|
| **Frozen frame removal** | All 5 subjects | ~1,030 frames with zero variance would corrupt BAF → must remove |
| **16-bit thermal preservation** | All frames | 8-bit loses quantization precision critical for 30mK sensitivity |
| **4-second rolling window BAF** | Temporal axis | Matches human breathing cycle duration; maximizes SNR |
| **KLT tracking** | All sequences | Subjects make small postural movements in seated 2-min recordings |
| **ROI-first processing** | 382×288 frames | Low resolution means face occupies small spatial area; ROI focus critical |
| **MRF+ICM segmentation** | Per-frame BAP maps | No labeled data available; unsupervised, view-adaptive approach |
| **Per-subject normalization** | Across 5 subjects | Different ambient conditions between subjects (pixel mean 63–73) |
| **Interpolation at frozen seams** | 39 frozen run boundaries | Prevents spike artifacts in rolling BAF computation |

---

## 8. 🚀 Recommended Implementation Order

- `[x]` **Phase 0:** Extract images from bag files ✅ (complete)
- `[ ]` **Phase 1:** Detect & index all frozen frame runs per subject
- `[ ]` **Phase 2:** Re-extract 16-bit thermal frames (or load PNGs + remap)
- `[ ]` **Phase 3:** Implement face detector + perinasal ROI cropper
- `[ ]` **Phase 4:** Implement KLT tracker for ROI propagation
- `[ ]` **Phase 5:** Implement BAF computation (rolling window std dev per pixel)
- `[ ]` **Phase 6:** Generate BAP probability maps
- `[ ]` **Phase 7:** Implement MRF + ICM segmentation (2-class, 30 iterations)
- `[ ]` **Phase 8:** Extract breathing waveform from BAFR mask
- `[ ]` **Phase 9:** Detect breathing peaks, estimate rate
- `[ ]` **Phase 10:** Evaluate and visualize results per subject

---

*Report generated: September 19, 2026*  
*Data location: `c:\Users\arpit\OneDrive\Desktop\dataset\`*  
*Paper: DOI 10.1109/JTEHM.2023.3295775*
