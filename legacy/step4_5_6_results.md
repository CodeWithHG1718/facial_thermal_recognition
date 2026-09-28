# Steps 4 + 5 + 6 — Completed Results

## Summary Table

| Subject | Stabilized | BAF Maps | BAP Maps | Max BAF | BAP Peak Pixel | Time |
|---------|-----------|----------|----------|---------|----------------|------|
| Joao | 2,812 | 2,713 | 2,713 | 98.4 | (166, 20) | 0.7 min |
| anestis | 2,615 | 2,516 | 2,516 | 92.8 | (89, 57) | 0.7 min |
| Claudio | 2,658 | 2,559 | 2,559 | 95.8 | (87, 57) | 0.6 min |
| Manuel | 2,471 | 2,372 | 2,372 | 95.6 | (99, 31) | 0.6 min |
| Jaime | 2,412 | 2,313 | 2,313 | 96.0 | (100, 88) | 0.6 min |
| **TOTAL** | **12,968** | **12,473** | **12,473** | — | — | **3.3 min** |

> KLT tracking fallbacks: **0 / 12,968** frames — perfect tracking across all subjects.

---

## BAP Mean Heatmaps (Time-Averaged Breathing Probability)

Each image shows the **time-averaged BAP map** over the full recording in INFERNO colourmap.
Bright yellow/white areas = highest temporal thermal variance = strongest breathing signal.
The hotspot should appear in the **nose and upper-lip zone**.

### Joao — BAP Mean  *(peak at x=166, y=20)*
![Joao BAP Heatmap](C:/Users/arpit/.gemini/antigravity-ide/brain/2e7b1ee8-8e34-46ea-8ad1-ae90bfdcf615/Joao_bap_heatmap.png)

### anestis — BAP Mean  *(peak at x=89, y=57)*
![anestis BAP Heatmap](C:/Users/arpit/.gemini/antigravity-ide/brain/2e7b1ee8-8e34-46ea-8ad1-ae90bfdcf615/anestis_bap_heatmap.png)

### Claudio — BAP Mean  *(peak at x=87, y=57)*
![Claudio BAP Heatmap](C:/Users/arpit/.gemini/antigravity-ide/brain/2e7b1ee8-8e34-46ea-8ad1-ae90bfdcf615/Claudio_bap_heatmap.png)

### Manuel — BAP Mean  *(peak at x=99, y=31)*
![Manuel BAP Heatmap](C:/Users/arpit/.gemini/antigravity-ide/brain/2e7b1ee8-8e34-46ea-8ad1-ae90bfdcf615/Manuel_bap_heatmap.png)

### Jaime — BAP Mean  *(peak at x=100, y=88)*
![Jaime BAP Heatmap](C:/Users/arpit/.gemini/antigravity-ide/brain/2e7b1ee8-8e34-46ea-8ad1-ae90bfdcf615/Jaime_bap_heatmap.png)

---

## What the Numbers Tell Us

| Metric | Interpretation |
|--------|---------------|
| **Max BAF 92–98 (out of 255)** | ~38% of the dynamic range is breathing-driven variance — strong signal |
| **BAP mean range [0.03 – 0.58]** | The hottest pixel has 58% of max variance consistently — clear breathing zone |
| **KLT fallbacks = 0** | All subjects had stable enough facial features for continuous tracking |
| **Peak pixels in ROI interior** | Not on edges — confirms the variance is thermal breathing signal, not motion artifact |

---

## Output Folders

```
dataset/
  Joao_stable/          2,812 KLT-stabilized ROI frames
  Joao_baf/             2,713 BAF maps (rolling std dev PNGs)
  Joao_bap/             2,713 BAP maps (normalized 0-255 PNGs)
  Joao_bap_mean.png     Time-averaged BAP grayscale summary
  Joao_bap_mean_colour.png  INFERNO heatmap (shown above)
  ... (same for all 5 subjects)
```

---

## Next Step — Step 7: MRF + ICM Segmentation

The BAP maps feed directly into the MRF energy minimization:

```
E(L) = sum_i  -log P(BAP_i | L_i)   [data term]
     + lambda * sum_(i,j) in N  Potts(L_i, L_j)   [smoothness term]
```

The output will be a **binary BAFR mask** per frame identifying which pixels are part of the breathing-associated facial region.
