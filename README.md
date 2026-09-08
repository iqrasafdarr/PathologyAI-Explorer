<div align="center">

# 🔬 PathologyAI-Explorer

**Preliminary Exploration of Histopathology Patch Classification and Attention-Based Multiple Instance Learning**

[![Status](https://img.shields.io/badge/status-Phase%201%20%7C%20Data%20Pipeline%20Complete-yellow)](#-project-progress)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset: PCam](https://img.shields.io/badge/dataset-PatchCamelyon-lightgrey)](https://github.com/basveeling/pcam)

*A scoped, honest research exploration into computational pathology — built to learn, not to overclaim.*

</div>

---

## 📌 Overview

**PathologyAI-Explorer** documents an undergraduate-level exploration of **computational pathology**, focused on:

- **Patch-level histopathology classification** using the [PatchCamelyon (PCam)](https://github.com/basveeling/pcam) dataset (metastatic breast cancer detection in lymph node sections)
- The conceptual bridge between patch-level predictions and **whole-slide image (WSI)** analysis via **Multiple Instance Learning (MIL)**
- A transparent record of what has actually been built vs. what is planned — no fabricated results, no premature claims

> ⚠️ This is **not** a production or clinical-grade system, and it does not (yet) implement full WSI classification. It is a scoped research exercise demonstrating the ability to learn and reason rigorously about a new medical-imaging subdomain.

---

## 🧭 Where This Fits: Patch Classification vs. Whole-Slide Diagnosis

Whole-slide images are gigapixel-scale — far too large to feed directly into a CNN. PCam solves this by pre-extracting fixed-size patches; this project consumes exactly that. The diagram below shows the full conceptual path from raw slide to patch, and where this repo's implemented pipeline (green) currently ends.

```mermaid
%%{init: {'theme':'dark', 'themeVariables': {'background':'#1a1a1a'}}}%%
flowchart LR
    A["🔬 Whole-Slide Image<br/>~100k × 100k px"]:::implemented --> B["🧫 Tissue Detection<br/>Filter background"]:::implemented
    B --> C["✂️ Extract Patches<br/>96×96 crops"]:::implemented
    C --> D["🟥 Patch 1<br/>Tumor"]:::patch
    C --> E["🟩 Patch 2<br/>Normal"]:::patch
    C --> F["🟦 Patch N<br/>Mixed"]:::patch

    classDef implemented fill:#0e7c86,stroke:#0a5a61,color:#ffffff,font-weight:bold;
    classDef patch fill:#8b3a3a,stroke:#5e2727,color:#ffffff,font-weight:bold;
```
<sub><i>PCam dataset: ~327k patches, 2 classes (tumor / normal), sourced from ~400 slides · patch-level labels available; slide-level labels are withheld by design → the exact motivation for MIL (see below).</i></sub>

---

### 🎯 Planned Next Step: Attention-Based Slide Aggregation (Phase 3–4, not yet implemented)

Once patch-level predictions exist, the standard CLAM-style approach learns an **attention weight (α)** per patch so the most diagnostically relevant patches drive the final slide-level call — no patch-level ground truth required during training. The diagram below illustrates the *mechanism*, not actual output from this repo.

```mermaid
%%{init: {'theme':'dark', 'themeVariables': {'background':'#1a1a1a'}}}%%
flowchart TD
    P1["P(tumor) = 0.92"]:::pred
    P2["P(tumor) = 0.18"]:::pred
    P3["P(tumor) = 0.76"]:::pred
    P1 -->|"α = 0.40"| AGG["⚖️ Weighted Attention<br/>Aggregation"]:::agg
    P2 -->|"α = 0.12"| AGG
    P3 -->|"α = 0.28"| AGG
    AGG --> OUT["✅ Slide Diagnosis"]:::out

    classDef pred fill:#5c4a13,stroke:#3d3009,color:#ffffff,font-weight:bold;
    classDef agg fill:#7a651f,stroke:#544512,color:#ffffff,font-weight:bold;
    classDef out fill:#1e5233,stroke:#123222,color:#ffffff,font-weight:bold;
```
<sub><i>Illustrative example with placeholder numbers — for exposition only. This repo has not trained an attention/MIL model; see <a href="#-project-progress">Project Progress</a>.</i></sub>

**Legend:** 🟢 Teal/Green = implemented in this repo · 🟡 Gold/dashed = conceptual, planned for a later phase (not yet built)

---

## ⚙️ Data Pipeline Architecture (Implemented — Phase 1)

```mermaid
%%{init: {'theme':'dark', 'themeVariables': {'background':'#1a1a1a', 'primaryColor':'#0e7c86'}}}%%
flowchart TD
    A[("PCam HDF5 files<br/>(Zenodo mirror)")]:::done --> B["download_data.py<br/>download + checksum verify"]:::done
    B --> C["decompress → data/"]:::done
    C --> D["PyTorch Dataset<br/>src/data/dataset.py"]:::done
    D --> E["Transform pipelines<br/>src/data/transforms.py<br/>(ImageNet norm + mild augmentation)"]:::done
    E --> F["DataLoader<br/>(multi-worker safe:<br/>Linux / Windows / Colab)"]:::done
    F --> G["Sanity Check<br/>--sanity-check"]:::done
    G --> H["Dataset sizes,<br/>class balance,<br/>tensor shapes"]:::done
    G --> I["results/figures/<br/>sample_patches.png"]:::done

    classDef done fill:#0e5c47,stroke:#0a3f30,color:#ffffff,font-weight:bold;
```

Every box above corresponds to code that runs today — no placeholders, no stubs.

---

## 🧩 Key Concepts

| Concept | Description |
|---|---|
| **Histopathology** | Microscopic tissue examination for disease diagnosis — here, metastatic breast cancer in lymph node sections. |
| **Whole-Slide Imaging (WSI)** | Gigapixel-scale digitized tissue slides, too large for direct CNN input. |
| **Patch Classification** | Small, fixed-size crops classified individually — exactly what PCam provides pre-extracted. |
| **Multiple Instance Learning (MIL)** | Standard method for aggregating many patch-level predictions into one slide-level label without per-patch ground truth. |
| **Attention-Based Aggregation** | Learned weighting (as in CLAM) so the most diagnostically relevant patches drive the slide-level decision. |

---

## 📊 Project Progress

**Current status: Data pipeline implemented — model training pending.**

```mermaid
flowchart LR
    P1["Phase 1: Data Pipeline<br/>✅ Complete"]:::complete --> P2["Phase 2: Training Baseline<br/>ResNet50 + Metrics"]:::next
    P2 --> P3["Phase 3: Documentation<br/>WSI / MIL Concepts"]:::later
    P3 --> P4["Phase 4: CLAM Exploration<br/>If time permits"]:::later

    classDef complete fill:#2f7d4f,stroke:#1e5233,color:#ffffff,font-weight:bold;
    classDef next fill:#0e7c86,stroke:#0a5a61,color:#ffffff,font-weight:bold;
    classDef later fill:#4a4fa0,stroke:#33366e,color:#ffffff,font-weight:bold;
```

<details>
<summary><b>📅 Detailed timeline (click to expand)</b></summary>

```mermaid
gantt
    dateFormat  X
    axisFormat %s
    section Phase 1: Data Pipeline
    Download + checksum verification      :done, p1a, 0, 1
    PyTorch Dataset/DataLoader             :done, p1b, 0, 1
    Transform pipelines                    :done, p1c, 0, 1
    Sanity check + sample-patch grid       :done, p1d, 0, 1
    section Phase 2: Training & Evaluation
    ResNet50 transfer learning             :active, p2a, 1, 2
    Evaluation metrics (Acc/P/R/F1/CM)     :p2b, 1, 2
    Training curves + failure analysis     :p2c, 1, 2
    section Phase 3: WSI / MIL
    WSI-MIL conceptual docs + diagram      :p3a, 2, 3
    Scoped CLAM exploration                :p3b, 2, 3
```

</details>

| Component | Status |
|---|:---:|
| PCam download, checksum, decompression | ✅ Done |
| Dataset / DataLoader pipeline | ✅ Done |
| Train/eval transforms | ✅ Done |
| Data-pipeline sanity check + visualization | ✅ Done |
| ResNet50 transfer-learning training | ⏳ Planned |
| Evaluation metrics & confusion matrix | ⏳ Planned |
| Training/validation curves | ⏳ Planned |
| WSI/MIL conceptual documentation | ⏳ Planned |
| Scoped CLAM exploration | ⏳ Planned |

---

## 📈 Results

No experiments have been run yet. **No accuracy, F1, or other performance numbers exist in this repository**, and none will be added until they come from an actual executed training run.

---

## 🔍 CLAM Exploration

Not started. Will be added as a clearly separate, explicitly scoped exploration once the core PCam classification pipeline (Phase 1) is complete — see the approved project scope in `docs/` (to be added).

---

## ⚠️ Limitations

- PCam provides pre-extracted 96×96 patches, not full WSIs — this project does not perform WSI-scale processing.
- Patch-level classification does not by itself constitute slide-level diagnosis; the relationship between the two is a documentation exercise (Phase 3), not an additional experiment.
- No stain normalization is applied in Phase 1 beyond the augmentation pipeline's mild color jitter; PCam's own patch-selection process already filters for tissue content (see `data/README.md`).

Full limitations will be documented in `docs/limitations.md` once experiments exist.

---

## 🚀 Setup

```bash
pip install -r requirements.txt
```

### Download the dataset

```bash
python scripts/download_data.py
```

Downloads PCam's HDF5 files from the official Zenodo mirror (~8 GB total), verifies checksums, and decompresses them into `data/`. See `data/README.md` for source details, manual fallback links, and split/leakage documentation.

### Run the data pipeline sanity check

```bash
python -m src.data.dataset --sanity-check
```

This does **not** train anything. It loads all three splits, reports dataset sizes and class balance, checks tensor shapes end-to-end through a real `DataLoader`, and saves a sample-patch grid to `results/figures/sample_patches.png`.

---

## 📁 Repository Structure

```
PathologyAI-Explorer/
├── configs/            # Experiment / pipeline configuration files
├── data/                # Downloaded + decompressed PCam data (gitignored)
├── notebooks/          # Exploratory notebooks
├── results/
│   └── figures/         # Saved sanity-check visualizations
├── scripts/
│   └── download_data.py # PCam download + checksum + decompression
├── src/
│   └── data/
│       ├── dataset.py    # PyTorch Dataset/DataLoader
│       └── transforms.py # Train/eval transform pipelines
├── requirements.txt
└── LICENSE
```

---

## 📚 Citation

If referencing the dataset used here:

> B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling. "Rotation Equivariant CNNs for Digital Pathology." arXiv:1806.03962 (2018).

PCam is derived from the CAMELYON16 challenge dataset.

---

## 📄 License

Code in this repository is MIT licensed (see `LICENSE`). The PCam dataset itself is separately CC0-licensed by its original authors — see `data/README.md`.
