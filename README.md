---
title: DCG Slag Viscosity Controller
emoji: 🏭
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: true
---

# 🏭 Real-Time Slag Viscosity Prediction for Dry Centrifugal Granulation (DCG)

**Mineral & Metallurgical Engineering · IIT (ISM) Dhanbad**

> A machine learning system that predicts blast furnace slag viscosity in real time
> and recommends disc RPM adjustments for Dry Centrifugal Granulation heat recovery.
> Built with 4 ML models, Bayesian hyperparameter tuning, SHAP explainability,
> PCA anomaly detection, and a Qwen2.5-7B LLM expert report panel.

---

## Why This Problem Matters

In **Dry Centrifugal Granulation (DCG)**, molten blast furnace slag at 1450–1550 °C
is poured onto a spinning disc (1000–3000 RPM). The disc breaks the slag into fine
droplets for waste-heat recovery — but only if the slag viscosity is in a narrow window:

| Viscosity | What Happens | RPM Action |
|-----------|-------------|------------|
| < 0.055 Pa·s | Slag too fluid → large irregular blobs | 🔵 Reduce RPM |
| 0.055 – 0.080 Pa·s | ✅ Fine spherical granules — optimal heat transfer | 🟢 Maintain RPM |
| > 0.080 Pa·s | Slag too viscous → fibres form, system clogs | 🔴 Increase RPM |

Viscosity changes every tap depending on temperature and chemistry.
**No commercial real-time control system exists globally.**
This project solves that with a deployed ML demo.

---

## How It Works

### Input Features

| Feature | Range | Physical Meaning |
|---------|-------|-----------------|
| Temperature | 1400 – 1600 °C | Tap temperature |
| Basicity (CaO/SiO₂) | 0.8 – 1.4 | Higher = less viscous (CaO breaks Si–O network) |
| Al₂O₃ | 8 – 18 wt% | Network former — raises viscosity |
| MgO | 4 – 12 wt% | Network modifier — slightly reduces viscosity |
| Coke Rate | 450 – 550 kg/t | Proxy for furnace heat input |
| Tap Time | 0 – 90 min | Elapsed time since tap — slag cools over time |
| Optical Basicity Λ | 0.55 – 0.75 | Auto-computed from mole fractions (Duffy & Ingram 1976) |

Optical Basicity is derived from the slag composition:

```
Λ = X_CaO × 1.00 + X_SiO₂ × 0.48 + X_Al₂O₃ × 0.60 + X_MgO × 0.78
```

where X values are mole fractions. Higher Λ → more basic → lower viscosity.

### Viscosity Model (Data Generation)

Synthetic training data (5,000 points) is generated using the **Urbain model**:

```
η = A · exp(B / T_K)
```

where A and B are empirical functions of the slag oxide composition, with ±5% Gaussian noise
to simulate real plant measurement variation.

---

## ML Pipeline

### Four Models Trained and Compared

| Model | Architecture | Expected CV R² |
|-------|-------------|----------------|
| Random Forest | 300 trees, max_depth=15 | ~0.97 |
| XGBoost | 300 estimators, lr=0.05, depth=6 | ~0.97–0.98 |
| CatBoost | 500 iterations, lr=0.05, depth=6 | ~0.97–0.98 |
| Neural Network | 64→32→16, BatchNorm + Dropout | ~0.95–0.96 |

> *Note: Exact numbers depend on the random seed and training run.
> Run the notebook to see your actual results.*

### Bayesian Hyperparameter Tuning (Optuna)

The best model by test R² is automatically selected and tuned with **50 Optuna trials**
(TPE sampler, minimising 5-fold CV RMSE). The tuned model is what drives all predictions
in the Gradio demo.

### SHAP Explainability

**SHAP (SHapley Additive exPlanations)** values are computed on the tuned model:
- Beeswarm plot — how each feature affects every prediction
- Bar chart — global average feature importance (embedded in the demo)
- Dependence plots — how the top 2 features relate to viscosity individually

The SHAP bar chart is saved as `plot_shap_bar.png` and displayed live in the app.

### PCA Anomaly Detector

A 5-component PCA is fitted on the training data. For each new input, the Mean
Squared Reconstruction Error (MSRE) is computed. If MSRE > training mean + 3σ,
the input is flagged as **out-of-distribution** and a warning is shown — the
prediction is still made but marked as an extrapolation.

---

## Gradio Demo — How to Use

1. **Adjust the sliders** on the left panel to your current slag conditions.
   Optical Basicity Λ is auto-computed from your inputs — the slider is display only.
2. **Click "Predict"** to get viscosity predictions from all four models instantly.
   The Tuned Best Model drives the RPM recommendation.
3. **Check the Data Quality panel** — the PCA anomaly detector confirms whether
   your inputs are within the model's training range.
4. **Click "Expert LLM Report"** to get a Qwen2.5-7B metallurgist explanation.
   The LLM prompt includes your exact SHAP attribution values for this prediction,
   not just generic feature importance. Requires `HF_TOKEN` secret to be set.
   Falls back to a rule-based explanation if the API is unavailable.

---

## Repository Structure

```
├── app.py                        ← Gradio demo (loads saved models, runs UI)
├── requirements.txt              ← Python dependencies for the HF Space
├── README.md                     ← This file
├── dcg_slag_viscosity_ml_final.py ← Training script (run on Colab)
│
└── [Generated after running Colab — uploaded to HF Space separately]
    ├── model_rf_reg.joblib
    ├── model_xgb_reg.joblib
    ├── model_cat_reg.joblib
    ├── model_nn_reg.keras
    ├── model_tuned_best.joblib   ← or .keras if NN wins
    ├── scaler.joblib
    ├── label_encoder.joblib
    ├── model_metadata.json
    ├── pca_detector.joblib
    ├── anomaly_config.json
    └── plot_shap_bar.png
```

---

## Running the Training Notebook

1. Open `dcg_slag_viscosity_ml_final.py` in Google Colab as a notebook.
2. Change `SAVE_DIR = "./"` to `SAVE_DIR = "/content/"` (line ~85).
3. Set Runtime → GPU (optional, speeds up Neural Network training).
4. Run All — takes approximately 20–30 minutes.
5. Download all `.joblib`, `.keras`, `.json` files and `plot_shap_bar.png`
   from the `/content/` directory.

---

## References

1. **Xin et al. (2025)** — Bayesian-Optimized CatBoost + SHAP for BF slag viscosity.
   Achieved R²=0.9897, RMSE=1.0619, hit ratio 95.1%.
   *Journal of Non-Crystalline Solids.*
   https://doi.org/10.1007/s42243-025-01608-z

2. **Zhang et al. (2025)** — RF, GBRT, and ANN for BF slag performance prediction.
   R² consistently above 0.97 on the CaO–SiO₂–Al₂O₃–MgO system.
   *Ironmaking & Steelmaking.*
   https://doi.org/10.1177/03019233251353314

3. **Chen et al. (2026)** — Optical basicity as domain-knowledge feature for ML
   viscosity prediction. Validation error reduced to 8–15%.
   *International Journal of Minerals, Metallurgy and Materials.*
   https://doi.org/10.1007/s12613-025-3189-4

4. **Shankar et al. (2020)** — PCA-KNN model for BF slag viscosity, 99% accuracy.
   *JOM, 72, 3687–3696.*
   https://doi.org/10.1007/s11837-020-04360-9

---

*Mineral & Metallurgical Engineering · IIT (ISM) Dhanbad*      
*Presented as Innovation 1 in a proposed DCG heat-recovery control system*