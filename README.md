# 🐟 [Ifeka OD Fish Health Vision Monitor]
## Aquaculture Fish Health Vision Monitor: Two-Stage Cascaded Edge AI Pipeline

## Overview
A lightweight, edge-optimized computer vision pipeline designed for early skin lesion and fungal patch detection in aquaculture (specifically targeting African catfish). Built with a resource-conscious architecture, the system balances low-spec local hardware constraints with advanced Out-of-Distribution (OOD) filtering and local CPU inference.

### Result for Healthy Catfish
<img width="1567" height="1557" alt="image" src="https://github.com/user-attachments/assets/9ae524af-0ab8-4b89-880d-1b66a428ed89" />


### Result for Catfish with Lesion
<img width="1615" height="1546" alt="image" src="https://github.com/user-attachments/assets/7d6d92ab-179e-4ad2-931a-7af6194570f7" />


---

## 🛠️ Architecture & Strategy
The system utilizes a **Two-Stage Cascaded Pipeline** to eliminate false positives and ensure real-time local performance on restricted hardware:
1. **Stage 1 (ONNX Gatekeeper):** Uses a COCO-pretrained YOLOv8n model via ONNX Runtime to inspect pixel distributions and screen out-of-distribution (OOD) inputs (such as portraits, vehicles, or indoor scenes) before running diagnostics.
2. **Stage 2 (Localized Delineation):** Employs OpenCV HSV color-space and morphological segmentation to isolate ulcerative erythema (reddish open sores) and fungal patches on dark catfish skin, drawing bounding boxes with confidence scores.

---

## 💻 Hardware & Deployment Strategy
* **Cloud Training Tier (Google Colab T4 GPU):** Heavy model training, fine-tuning, and data augmentation occur on free-tier cloud accelerators.
* **Edge Inference (Local Intel i5, 8GB RAM Laptop):** Model execution runs locally via ONNX Runtime (`yolov8n.onnx`) and OpenCV, completely eliminating the need for heavy local GPUs or costly cloud web hosting.

---

## 📂 Repository Structure
```text
catfish-lesion-detector-edge/
├── notebooks/
│   └── Catfish_Lesion_Detection_YOLOv8.ipynb # Trained on Colab T4 GPU
├── models/
│   └── yolov8n.onnx                          # Optimized FP16 ONNX model for edge CPU
├── app/
│   └── app.py                                # Lightweight Streamlit Web Dashboard (Two-Stage Pipeline)
└── README.md                                 # Project documentation & benchmarks
