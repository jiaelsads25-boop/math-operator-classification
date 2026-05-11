# math-operator-classification
Handwritten mathematical operator symbol classification using HOG + SVM/KNN/Decision Tree
# 🔢 Mathematical Operator Symbol Classification from Handwritten Images

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-green) ![Accuracy](https://img.shields.io/badge/SVM%20Accuracy-99.29%25-brightgreen) ![License](https://img.shields.io/badge/License-MIT-yellow)

> **Course:** Predictive Analytics | **Topic:** 34 | **Team:** Adithyan | Jia | Theertha

---

## 📌 Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Setup & Run Locally](#setup--run-locally)
- [Team Contributions](#team-contributions)

---

## 🎯 Problem Statement

Handwritten mathematical notation is widely used in education, research, and engineering. Automatically recognising handwritten mathematical operator symbols is a challenging computer vision problem due to:

- **Visual similarity** between certain symbols (e.g., `×` vs `√`, `−` vs `÷`)
- **Natural variation** in individual handwriting styles
- **Stroke ambiguity** after preprocessing (skeletonization)

This project builds a **multi-class classification system** that identifies **10 handwritten mathematical operator symbols** from images.

---

## 📦 Dataset

| Property | Value |
|---|---|
| Source | Synthetically generated (handwritten-style) |
| Total images (raw) | 500 (50 per class) |
| Total images (after augmentation) | 3,500 |
| Image size | 100×100 px (raw) → 32×32 px (processed) |
| Classes | 10 |
| Format | PNG, black on white background |

### 10 Target Symbols

| Folder | Symbol | Name |
|---|---|---|
| `plus` | + | Addition |
| `minus` | − | Subtraction |
| `multiply` | × | Multiplication |
| `divide` | ÷ | Division |
| `equal` | = | Equals |
| `not_equal` | ≠ | Not Equal |
| `less_than` | < | Less Than |
| `greater_than` | > | Greater Than |
| `plus_minus` | ± | Plus Minus |
| `sqrt` | √ | Square Root |

**Class distribution:** Perfectly balanced — 50 images per class (before augmentation), 350 per class (after augmentation).

---

## 🔬 Methodology

### Data Science Life Cycle

#### Stage 1 — Problem Definition & Literature Review
- Defined the scope: 10 operator symbols, multi-class classification
- Reviewed relevant work: CROHME dataset, HOG+SVM for symbol recognition, HASYv2 dataset

#### Stage 2 — Data Collection & Understanding
- Generated 500 synthetic handwritten-style images using Python (PIL)
- Each image features natural stroke variation: wobble, pressure, rotation, scale
- Verified class balance with distribution chart

#### Stage 3 — Preprocessing & Cleaning
- **Grayscale conversion** — standardise input
- **Otsu binarization** — adaptive thresholding for clean black/white separation
- **Skeletonization** — reduce strokes to single-pixel width using `skimage.morphology.skeletonize`
- **Resize to 32×32** — fixed-size normalisation
- **Data augmentation** — rotation (±10°), scaling (0.9–1.1×) → 3,500 total samples

#### Stage 4 — Exploratory Data Analysis
- Sample grid (5 per class) — verified visual quality
- Mean image per class — identified stroke patterns and confusable pairs
- Pixel intensity histograms — confirmed binary distribution
- Confusable pairs identified: `mul/sqrt`, `minus/div`, `equal/not_equal`

#### Stage 5 — Feature Engineering & Selection
- **HOG features** (1764 dims) — captures stroke orientation using `orientations=9, pixels_per_cell=(4,4)`
- **Projection profiles** (64 dims) — horizontal + vertical pixel sums as shape descriptors
- **Run-length features** (32 dims) — encodes stroke structure per row
- **Total feature vector: 1,860 dimensions**
- Normalised with `StandardScaler`

#### Stage 6 — Model Building & Training
- **SVM** — RBF kernel, C=10, gamma='scale', One-vs-Rest
- **KNN** — k selected by 5-fold CV (best k=5)
- **Decision Tree** — max_depth selected by CV (best=None)
- 80/20 stratified train-test split

#### Stage 7 — Model Evaluation & Comparison

| Model | Accuracy | Macro F1 | Precision | Recall |
|---|---|---|---|---|
| **SVM** | **99.29%** | **0.9929** | **0.9933** | **0.9929** |
| KNN | 98.86% | 0.9886 | 0.9888 | 0.9886 |
| Decision Tree | 92.71% | 0.9274 | 0.9286 | 0.9271 |

#### Stage 8 — Model Interpretation & Explainability
- Decision Tree feature importances — top features are early HOG indices (stroke orientation)
- Misclassified samples analysis — all 5 SVM errors involved `sqrt` as predicted class
- Confusable pairs: `mul→sqrt` (3), `!= →sqrt` (1), `>→sqrt` (1)
- Per-class precision-recall chart — `sqrt` has lowest precision (0.94)

#### Stage 9 — Deployment
- Streamlit web application with drawing canvas and image upload
- Model selector (SVM/KNN/Decision Tree) in sidebar
- Confidence scores and class probability bar chart
- Deployed on Streamlit Community Cloud

#### Stage 10 — Documentation
- This README, PPT presentation, and GitHub repository

---

## 📊 Results

### Best Model: SVM (RBF Kernel)
- **Test Accuracy: 99.29%**
- **Macro F1: 0.9929**
- Only 5 misclassifications out of 700 test samples

### Key Findings
1. SVM outperforms KNN and Decision Tree significantly
2. Decision Tree overfits (100% train, 92.71% test) — classic depth=None problem
3. Most confusable pair: `multiply (×)` vs `sqrt (√)` — both have diagonal strokes
4. HOG features are most discriminative — top importance indices are early HOG bins

---

## 🚀 Deployment

### Live Application
🔗 https://math-operator-classification-py2ylywnbp9i5rt44nkr4g.streamlit.app/

### App Features
- ✏️ **Draw mode** — freehand canvas drawing
- 📁 **Upload mode** — upload any PNG/JPG
- 🤖 **Model selector** — SVM / KNN / Decision Tree
- 📊 **Probability chart** — all 10 class probabilities
- 🔍 **Preprocessing view** — see the 32×32 skeleton

### Screenshot
> *(Add screenshot of deployed app here)*

---

## 📁 Project Structure

```
math-operator-classification/
│
├── math_operator_classification.ipynb   ← Main notebook (all 10 stages)
├── app.py                               ← Streamlit deployment app
├── requirements.txt                     ← Python dependencies
├── README.md                            ← This file
├── Math_Operator_Classification.pptx   ← Presentation slides
│
├── models/
│   ├── svm_model.pkl                   ← Trained SVM model
│   ├── knn_model.pkl                   ← Trained KNN model
│   ├── dt_model.pkl                    ← Trained Decision Tree
│   └── scaler.pkl                      ← StandardScaler
│
└── individual_profiles/
    ├── adithyan_github_profile.png     ← GitHub contribution graph
    ├── jia_github_profile.png
    └── theertha_github_profile.png
```

---

## ⚙️ Setup & Run Locally

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/math-operator-classification.git
cd math-operator-classification

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Run the Notebook
1. Open `math_operator_classification.ipynb` in Google Colab or Jupyter
2. Run the **Dataset Generator cell first** (generates 500 images)
3. Run all subsequent stages in order
4. Do not restart runtime between stages

---

## 👥 Team Contributions

| Member | Stages | Branches |
|---|---|---|
| **Adithyan** | Stage 1, 2, 4, 8 | feature/stage1, feature/stage2, feature/stage4, feature/stage8 |
| **Jia** | Stage 3, 6, 9 | feature/stage3, feature/stage6, feature/stage9 |
| **Theertha** | Stage 5, 7, 10 | feature/stage5, feature/stage7, feature/stage10 |

All members contributed to every stage of the data science life cycle. Individual GitHub contribution graphs are in `/individual_profiles/`.

---

## 📚 References

1. Mouchere et al. (2014) — CROHME: Competition on Recognition of Online Handwritten Mathematical Expressions
2. Dalal & Triggs (2005) — Histograms of Oriented Gradients for Human Detection
3. Cortes & Vapnik (1995) — Support-Vector Networks
4. Thoma (2017) — HASYv2 — A Large-Scale Document Dataset
5. Zhang & Suen (1984) — A fast parallel algorithm for thinning digital patterns

---

*Mathematical Operator Symbol Classification | Predictive Analytics Course | Topic 34*

