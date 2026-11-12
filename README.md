# 🐟 Aquaculture Fish Health Vision Monitor: Two-Stage Cascaded Edge AI Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime-green.svg)](https://onnxruntime.ai/)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-red.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

## 📌 Problem Statement & Motivation
Smallholder aquaculture farms in West Africa frequently suffer catastrophic stock losses due to delayed detection of dermal infections (ulcerative erythema, fungal patches) and acute environmental stress. Existing computer vision solutions typically rely on cloud-hosted, GPU-heavy deep learning models that are financially and infrastructurally unviable in rural pond environments with intermittent connectivity.

This project implements a **resource-conscious, Two-Stage Cascaded Edge AI Pipeline** capable of running real-time visual diagnostics locally on low-spec CPU hardware (Intel Core i5, 8GB RAM, no dedicated GPU).

---
### Demo
[Watch Video on Google Drive](https://shorturl.at/CLTr1)

___

### Result for catfish with not-noticeable lesion
<img width="1024" height="891" alt="e49d4a20-5703-4ddc-9b8e-23d66d316da6" src="https://github.com/user-attachments/assets/f819e677-9faa-4639-a93f-62b680fe096f" />

### Result for healthy catfish
<img width="1024" height="1017" alt="dc422522-761b-48f7-bda7-de9efa6ec686" src="https://github.com/user-attachments/assets/6ac6e96e-233c-4db4-a9a7-c1203516feae" />


## 🛠️ System Architecture & Engineering Strategy

To minimize false positives on noisy field inputs while maintaining sub-300ms CPU inference latency, the pipeline decouples domain validation from lesion localization:

1. **Stage 1 — Out-of-Distribution (OOD) ONNX Gatekeeper (`YOLOv8n.onnx`):**
   * Executes a lightweight, FP16-quantized YOLOv8-Nano model via `ONNX Runtime`.
   * Screens incoming frames to filter out non-aquatic/out-of-distribution scenes (e.g., human portraits, vehicles, indoor backgrounds) before diagnostic processing.
2. **Stage 2 — Localized Dermal Lesion Delineation (`OpenCV`):**
   * Applies HSV color-space transformations and morphological segmentation to isolate hyperemic/ulcerative lesions and fungal anomalies against the dark dorsal pigmentation of African catfish (*Clarias gariepinus*).
   * Outputs localized bounding boxes, peak confidence scores, and actionable biosecurity alerts.

---

## 📊 Performance Benchmarks (Local CPU Execution)

Evaluated locally on an **Intel Core i5 (10th Gen) CPU @ 1.60GHz, 8GB RAM (No Discrete GPU)**:

| Test Scenario | Input Resolution | Stage 1 + Stage 2 Latency | Diagnostic Output |
| :--- | :--- | :--- | :--- |
| **Diseased Specimen (Ex-situ)** | `225 × 225 px` | **179 ms** (~5.6 FPS) | 2 Dermal Lesions Localized (Peak Conf: 69.6%) |
| **Healthy Specimens (In-situ Pond)** | `1250 × 833 px` | **288 ms** (~3.5 FPS) | Healthy Specimen Verified (0 False Positives) |

---

## 🔬 Current Limitations & Future MSc Research Direction
While heuristic HSV color-space segmentation provides an effective, zero-shot baseline for ex-situ inspection, **color thresholding degrades under high water turbidity, fluctuating natural illumination, and algal blooms**. 

**Proposed Graduate Research Extension:**
1. Curating a domain-specific, annotated dataset of *Clarias gariepinus* under controlled turbidity gradients.
2. Replacing Stage 2's heuristic segmentation with a fine-tuned, lightweight neural architecture (**YOLOv8-Nano / MobileNetV3-SSD**) trained with synthetic turbidity augmentations.
3. Multimodal fusion combining visual lesion scores with time-series water-quality telemetry (dissolved oxygen, pH, unionized ammonia).

---

## 📂 Repository Structure
```text
catfish-lesion-detector-edge/
├── notebooks/
│   └── Catfish_Lesion_Detection_YOLOv8.ipynb  # Colab T4 GPU export & quantization pipeline
├── models/
│   └── yolov8n.onnx                           # Quantized ONNX model for local CPU inference
├── app/
│   └── app.py                                 # Streamlit diagnostic dashboard & cascaded pipeline
└── README.md                                  # Architecture, benchmarks & research roadmap
