# ============================================================
#  🏭 DCG Slag Viscosity ML System —
#  IIT (ISM) Dhanbad | Mineral & Metallurgical Engineering
#  Student: | Innovation 1 — Real-time DCG Control
# ============================================================

print("=== STEP 0: Installing Dependencies ===")
# Uncomment the line below when running in Colab:
# !pip install -q gradio scikit-learn tensorflow matplotlib seaborn pandas numpy xgboost catboost optuna shap huggingface_hub

import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

required = {
    "xgboost": "xgboost",
    "catboost": "catboost",
    "optuna": "optuna",
    "shap": "shap",
    "gradio": "gradio",
    "huggingface_hub": "huggingface_hub",
}

for mod, pkg in required.items():
    try:
        __import__(mod)
        print(f"  ✅ {pkg} already installed")
    except ImportError:
        print(f"  📦 Installing {pkg}...")
        install(pkg)
        print(f"  ✅ {pkg} installed")

print("✅ All dependencies ready.\n")

print("=== STEP 1: Imports ===")
import os, random, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (r2_score, mean_squared_error,
                             accuracy_score, confusion_matrix,
                             classification_report)
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import shap

warnings.filterwarnings("ignore")

# ── Fixed random seeds ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ── Plot theme constants ──────────────────────────────────────────────────────
BG_COLOR     = "#1a1a2e"
ACCENT       = "#f0a500"
TEXT_COLOR   = "#ffffff"
GREEN_COLOR  = "#00c896"
RED_COLOR    = "#e05c5c"
BLUE_COLOR   = "#4a9eff"

SAVE_DIR = "./"   # Change to /content/ on Colab

def styled_fig(w=10, h=6):
    """Return a dark-styled figure and axes."""
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(ACCENT)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    return fig, ax

print("✅ All libraries imported.\n")

print("=== STEP 2: Data Generation (Urbain-style model + Optical Basicity) ===")
N = 5000

# ── 6 base features ──────────────────────────────────────────────────────────
T_C       = np.random.uniform(1400, 1600, N)
basicity  = np.random.uniform(0.8,  1.4,  N)
Al2O3     = np.random.uniform(8,    18,   N)
MgO       = np.random.uniform(4,    12,   N)
coke_rate = np.random.uniform(450,  550,  N)
tap_time  = np.random.uniform(0,    90,   N)

T_K = T_C + 273.15

# ── Urbain-style viscosity coefficients ──────────────────────────────────────
A_coeff = np.exp(
    -17.51
    + 0.9  * basicity
    - 0.04 * Al2O3
    + 0.02 * MgO
    - 0.001 * coke_rate
)
B_coeff = (
    28000
    - 4000 * basicity
    + 500  * (Al2O3 / 10)
    - 200  * (MgO / 8)
    + 20   * tap_time
)
viscosity_true = A_coeff * np.exp(B_coeff / T_K)
viscosity_true = np.clip(viscosity_true, 0.05, 8.0)

# ±5 % Gaussian noise
noise     = np.random.normal(1.0, 0.05, N)
viscosity = np.clip(viscosity_true * noise, 0.05, 8.0)

# ── 7th feature: Optical Basicity (Λ) ─────────────────────────────────────────
# Assume SiO2_wt = 35 ± small variation (±2 wt%)
SiO2_wt = 35.0 + np.random.uniform(-2, 2, N)
CaO_wt  = basicity * SiO2_wt   # B = CaO/SiO2

# Molar masses (g/mol)
M_CaO   = 56.08
M_SiO2  = 60.09
M_Al2O3 = 101.96
M_MgO   = 40.30

# Moles (from wt% → proportional to wt/M)
n_CaO   = CaO_wt  / M_CaO
n_SiO2  = SiO2_wt / M_SiO2
n_Al2O3 = Al2O3   / M_Al2O3
n_MgO   = MgO     / M_MgO

n_total = n_CaO + n_SiO2 + n_Al2O3 + n_MgO

X_CaO   = n_CaO   / n_total
X_SiO2  = n_SiO2  / n_total
X_Al2O3 = n_Al2O3 / n_total
X_MgO   = n_MgO   / n_total

# Optical basicity formula
Lambda = (X_CaO*1.00 + X_SiO2*0.48 + X_Al2O3*0.60 + X_MgO*0.78)

print(f"  Optical Basicity range: {Lambda.min():.4f} – {Lambda.max():.4f}")

# ── Class labels with FIXED thresholds matching actual data range ─────────────
# Viscosity range after clipping is ~0.05–0.21 Pa·s for the generated data
# (the Urbain model with given parameters stays below 1 Pa·s for 1400-1600°C)
# Print actual range first
v_min, v_max = viscosity.min(), viscosity.max()
print(f"  Viscosity range: {v_min:.4f} – {v_max:.4f} Pa·s")

LOW_THRESH  = 0.055   # below → "Reduce RPM"   (too fluid)
HIGH_THRESH = 0.080   # above → "Increase RPM"  (too viscous)

def label_rpm(v):
    if v < LOW_THRESH:
        return "Reduce RPM"
    elif v <= HIGH_THRESH:
        return "Maintain RPM"
    else:
        return "Increase RPM"

disc_rec = np.array([label_rpm(v) for v in viscosity])

# Build DataFrame
FEATURES = ["Temperature_C", "Basicity", "Al2O3_wt", "MgO_wt",
            "Coke_Rate", "Tap_Time_min", "Optical_Basicity"]

df = pd.DataFrame({
    "Temperature_C":      T_C,
    "Basicity":           basicity,
    "Al2O3_wt":           Al2O3,
    "MgO_wt":             MgO,
    "Coke_Rate":          coke_rate,
    "Tap_Time_min":       tap_time,
    "Optical_Basicity":   Lambda,
    "Viscosity_Pas":      viscosity,
    "Disc_Recommendation": disc_rec,
})

print(f"\n✅ Dataset generated: {df.shape}")
print(f"\nViscosity statistics:\n{df['Viscosity_Pas'].describe().round(5)}")
print(f"\n📊 Class distribution (MUST have all 3 classes):")
print(df["Disc_Recommendation"].value_counts())
print(f"\nClass fractions:")
print(df["Disc_Recommendation"].value_counts(normalize=True).round(3))

assert len(df["Disc_Recommendation"].unique()) == 3, (
    "ERROR: Not all 3 classes present. Adjust thresholds!"
)
print("\n✅ All three RPM classes confirmed.\n")

print("=== STEP 3: Preprocessing — Scaling & Splitting ===")

TARGET_REG = "Viscosity_Pas"
TARGET_CLS = "Disc_Recommendation"

X     = df[FEATURES].values
y_reg = df[TARGET_REG].values
y_cls = df[TARGET_CLS].values

le          = LabelEncoder()
y_cls_enc   = le.fit_transform(y_cls)
class_names = le.classes_
print(f"  Class encoding: {dict(zip(range(len(class_names)), class_names))}")

X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
    X, y_reg, y_cls_enc,
    test_size=0.2, random_state=SEED, stratify=y_cls_enc
)

scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)

print(f"  Train: {X_train_sc.shape}  |  Test: {X_test_sc.shape}")
print(f"  Classes: {list(class_names)}")
print("✅ Preprocessing complete.\n")

print("=== STEP 4: Training All Four Models ===")

cv5 = KFold(n_splits=5, shuffle=True, random_state=SEED)

results = {}   # will hold per-model metrics

# ─────────────────────────────────────────────────────────────────────────────
# MODEL A — Random Forest
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [Model A] Random Forest (n_estimators=300, max_depth=15)...")

rf_reg = RandomForestRegressor(n_estimators=300, max_depth=15,
                                n_jobs=-1, random_state=SEED)
rf_reg.fit(X_train_sc, yr_train)
yr_pred_rf = rf_reg.predict(X_test_sc)

rf_cls = RandomForestClassifier(n_estimators=300, max_depth=15,
                                 n_jobs=-1, random_state=SEED)
rf_cls.fit(X_train_sc, yc_train)
yc_pred_rf = rf_cls.predict(X_test_sc)

cv_rf  = cross_val_score(rf_reg, X_train_sc, yr_train, cv=cv5, scoring="r2")
r2_rf  = r2_score(yr_test, yr_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(yr_test, yr_pred_rf))
acc_rf  = accuracy_score(yc_test, yc_pred_rf)

results["Random Forest"] = {
    "reg": rf_reg, "cls": rf_cls,
    "cv_r2": cv_rf, "test_r2": r2_rf, "rmse": rmse_rf, "acc": acc_rf,
    "yr_pred": yr_pred_rf, "yc_pred": yc_pred_rf,
}
print(f"    CV R² = {cv_rf.mean():.4f} ± {cv_rf.std():.4f} | "
      f"Test R² = {r2_rf:.4f} | RMSE = {rmse_rf:.5f} | Acc = {acc_rf*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL B — XGBoost
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [Model B] XGBoost (n_estimators=300, lr=0.05, max_depth=6)...")

xgb_reg = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                             subsample=0.8, random_state=SEED,
                             eval_metric="rmse", verbosity=0)
xgb_reg.fit(X_train_sc, yr_train)
yr_pred_xgb = xgb_reg.predict(X_test_sc)

xgb_cls = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                              subsample=0.8, random_state=SEED,
                              eval_metric="mlogloss", verbosity=0,
                              num_class=len(class_names), use_label_encoder=False)
xgb_cls.fit(X_train_sc, yc_train)
yc_pred_xgb = xgb_cls.predict(X_test_sc)

cv_xgb  = cross_val_score(xgb_reg, X_train_sc, yr_train, cv=cv5, scoring="r2")
r2_xgb  = r2_score(yr_test, yr_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(yr_test, yr_pred_xgb))
acc_xgb  = accuracy_score(yc_test, yc_pred_xgb)

results["XGBoost"] = {
    "reg": xgb_reg, "cls": xgb_cls,
    "cv_r2": cv_xgb, "test_r2": r2_xgb, "rmse": rmse_xgb, "acc": acc_xgb,
    "yr_pred": yr_pred_xgb, "yc_pred": yc_pred_xgb,
}
print(f"    CV R² = {cv_xgb.mean():.4f} ± {cv_xgb.std():.4f} | "
      f"Test R² = {r2_xgb:.4f} | RMSE = {rmse_xgb:.5f} | Acc = {acc_xgb*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL C — CatBoost
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [Model C] CatBoost (iterations=500, lr=0.05, depth=6)...")

cat_reg = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                             random_seed=SEED, verbose=0)
cat_reg.fit(X_train_sc, yr_train)
yr_pred_cat = cat_reg.predict(X_test_sc)

cat_cls = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                              random_seed=SEED, verbose=0)
cat_cls.fit(X_train_sc, yc_train)
yc_pred_cat = cat_cls.predict(X_test_sc).flatten().astype(int)

cv_cat  = cross_val_score(cat_reg, X_train_sc, yr_train, cv=cv5, scoring="r2")
r2_cat  = r2_score(yr_test, yr_pred_cat)
rmse_cat = np.sqrt(mean_squared_error(yr_test, yr_pred_cat))
acc_cat  = accuracy_score(yc_test, yc_pred_cat)

results["CatBoost"] = {
    "reg": cat_reg, "cls": cat_cls,
    "cv_r2": cv_cat, "test_r2": r2_cat, "rmse": rmse_cat, "acc": acc_cat,
    "yr_pred": yr_pred_cat, "yc_pred": yc_pred_cat,
}
print(f"    CV R² = {cv_cat.mean():.4f} ± {cv_cat.std():.4f} | "
      f"Test R² = {r2_cat:.4f} | RMSE = {rmse_cat:.5f} | Acc = {acc_cat*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL D — Neural Network (Keras)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [Model D] Neural Network (64→32→16, BatchNorm + Dropout, 80 epochs)...")

def build_nn_reg(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ], name="NN_Regressor")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def build_nn_cls(input_dim, n_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(16, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ], name="NN_Classifier")
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

nn_reg = build_nn_reg(X_train_sc.shape[1])
history_reg = nn_reg.fit(
    X_train_sc, yr_train,
    validation_split=0.1,
    epochs=80, batch_size=64, verbose=0,
    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
)

nn_cls = build_nn_cls(X_train_sc.shape[1], len(class_names))
history_cls = nn_cls.fit(
    X_train_sc, yc_train,
    validation_split=0.1,
    epochs=80, batch_size=64, verbose=0,
    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
)

yr_pred_nn = nn_reg.predict(X_test_sc, verbose=0).flatten()
yc_pred_nn = np.argmax(nn_cls.predict(X_test_sc, verbose=0), axis=1)

# NN cross-val: wrap in sklearn-compatible wrapper via manual loop
def nn_cv_r2(X_data, y_data, cv):
    scores = []
    for tr_idx, val_idx in cv.split(X_data):
        m = build_nn_reg(X_data.shape[1])
        m.fit(X_data[tr_idx], y_data[tr_idx],
              epochs=50, batch_size=64, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
              validation_split=0.1)
        pred = m.predict(X_data[val_idx], verbose=0).flatten()
        scores.append(r2_score(y_data[val_idx], pred))
    return np.array(scores)

print("    Running 5-fold CV for NN (this may take ~60s)...")
cv_nn   = nn_cv_r2(X_train_sc, yr_train, cv5)
r2_nn   = r2_score(yr_test, yr_pred_nn)
rmse_nn = np.sqrt(mean_squared_error(yr_test, yr_pred_nn))
acc_nn  = accuracy_score(yc_test, yc_pred_nn)

results["Neural Network"] = {
    "reg": nn_reg, "cls": nn_cls,
    "cv_r2": cv_nn, "test_r2": r2_nn, "rmse": rmse_nn, "acc": acc_nn,
    "yr_pred": yr_pred_nn, "yc_pred": yc_pred_nn,
    "history_reg": history_reg, "history_cls": history_cls,
}
print(f"    CV R² = {cv_nn.mean():.4f} ± {cv_nn.std():.4f} | "
      f"Test R² = {r2_nn:.4f} | RMSE = {rmse_nn:.5f} | Acc = {acc_nn*100:.2f}%")

print("\n✅ All four models trained.\n")

print("=== STEP 5: Performance Summary Table ===\n")

rows = []
best_name, best_r2 = "", -np.inf
for name, m in results.items():
    rows.append({
        "Model":            name,
        "CV R² mean":       round(m["cv_r2"].mean(), 4),
        "CV R² std":        round(m["cv_r2"].std(),  4),
        "Test R²":          round(m["test_r2"], 4),
        "Test RMSE (Pa·s)": round(m["rmse"],    5),
        "Cls Accuracy":     f"{m['acc']*100:.2f}%",
    })
    if m["test_r2"] > best_r2:
        best_r2   = m["test_r2"]
        best_name = name

summary_df = pd.DataFrame(rows)

print("=" * 75)
print("          DCG SLAG VISCOSITY PREDICTION — MODEL PERFORMANCE SUMMARY")
print("=" * 75)
print(summary_df.to_string(index=False))
print("=" * 75)
print(f"\n🏆 BEST MODEL by Test R²: {best_name}  (R² = {best_r2:.4f})")
print()
for name, m in results.items():
    print(f"\n  ── {name} Classification Report ──")
    print(classification_report(yc_test, m["yc_pred"], target_names=class_names))

print("✅ Summary complete.\n")

print(f"=== STEP 6: Optuna Tuning for Best Model → {best_name} (50 trials) ===\n")

best_reg_model = results[best_name]["reg"]

def make_objective(model_name):
    def objective(trial):
        if model_name == "Random Forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth":    trial.suggest_int("max_depth", 5, 25),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 5),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
            m = RandomForestRegressor(**params, n_jobs=-1, random_state=SEED)
        elif model_name == "XGBoost":
            params = {
                "n_estimators":  trial.suggest_int("n_estimators", 100, 600),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth":     trial.suggest_int("max_depth", 3, 10),
                "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha":     trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda":    trial.suggest_float("reg_lambda", 0.5, 5.0),
            }
            m = xgb.XGBRegressor(**params, random_state=SEED, verbosity=0, eval_metric="rmse")
        elif model_name == "CatBoost":
            params = {
                "iterations":    trial.suggest_int("iterations", 200, 800),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "depth":         trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            }
            m = CatBoostRegressor(**params, random_seed=SEED, verbose=0)
        else:   # Neural Network — tune architecture
            params = {
                "lr":      trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "dropout": trial.suggest_float("dropout", 0.0, 0.3),
                "units1":  trial.suggest_int("units1", 32, 128),
                "units2":  trial.suggest_int("units2", 16, 64),
            }
            m_nn = keras.Sequential([
                layers.Input(shape=(X_train_sc.shape[1],)),
                layers.Dense(params["units1"], activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(params["dropout"]),
                layers.Dense(params["units2"], activation="relu"),
                layers.Dense(1),
            ])
            m_nn.compile(optimizer=keras.optimizers.Adam(params["lr"]), loss="mse")
            rmse_scores = []
            for tr_idx, val_idx in cv5.split(X_train_sc):
                m_nn_i = keras.models.clone_model(m_nn)
                m_nn_i.compile(optimizer=keras.optimizers.Adam(params["lr"]), loss="mse")
                m_nn_i.fit(X_train_sc[tr_idx], yr_train[tr_idx],
                           epochs=30, batch_size=64, verbose=0,
                           validation_split=0.1)
                pred = m_nn_i.predict(X_train_sc[val_idx], verbose=0).flatten()
                rmse_scores.append(np.sqrt(mean_squared_error(yr_train[val_idx], pred)))
            return np.mean(rmse_scores)

        scores = cross_val_score(m, X_train_sc, yr_train, cv=cv5,
                                 scoring="neg_root_mean_squared_error")
        return -scores.mean()

    return objective

study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(make_objective(best_name), n_trials=50, show_progress_bar=False)

print(f"  Best Optuna RMSE: {study.best_value:.6f} Pa·s")
print(f"  Best parameters: {study.best_params}\n")

# ── Retrain Tuned Best Model ──────────────────────────────────────────────────
print(f"  Retraining Tuned Best Model ({best_name}) with optimal parameters...")
bp = study.best_params

if best_name == "Random Forest":
    tuned_reg = RandomForestRegressor(**bp, n_jobs=-1, random_state=SEED)
    tuned_cls = RandomForestClassifier(
        n_estimators=bp.get("n_estimators", 300),
        max_depth=bp.get("max_depth", 15),
        n_jobs=-1, random_state=SEED
    )
elif best_name == "XGBoost":
    tuned_reg = xgb.XGBRegressor(**bp, random_state=SEED, verbosity=0, eval_metric="rmse")
    tuned_cls = xgb.XGBClassifier(
        n_estimators=bp.get("n_estimators", 300),
        learning_rate=bp.get("learning_rate", 0.05),
        max_depth=bp.get("max_depth", 6),
        random_state=SEED, verbosity=0, eval_metric="mlogloss",
        num_class=len(class_names)
    )
elif best_name == "CatBoost":
    tuned_reg = CatBoostRegressor(**bp, random_seed=SEED, verbose=0)
    tuned_cls = CatBoostClassifier(
        iterations=bp.get("iterations", 500),
        learning_rate=bp.get("learning_rate", 0.05),
        depth=bp.get("depth", 6),
        random_seed=SEED, verbose=0
    )
else:
    tuned_reg = build_nn_reg(X_train_sc.shape[1])
    tuned_cls = build_nn_cls(X_train_sc.shape[1], len(class_names))

tuned_reg.fit(X_train_sc, yr_train)
tuned_cls.fit(X_train_sc, yc_train)

yr_pred_tuned = tuned_reg.predict(X_test_sc)
if best_name == "Neural Network":
    yr_pred_tuned = yr_pred_tuned.flatten()
    yc_pred_tuned = np.argmax(tuned_cls.predict(X_test_sc, verbose=0), axis=1)
elif best_name == "CatBoost":
    yc_pred_tuned = tuned_cls.predict(X_test_sc).flatten().astype(int)
else:
    yc_pred_tuned = tuned_cls.predict(X_test_sc)

r2_tuned   = r2_score(yr_test, yr_pred_tuned)
rmse_tuned = np.sqrt(mean_squared_error(yr_test, yr_pred_tuned))
acc_tuned  = accuracy_score(yc_test, yc_pred_tuned)

print(f"\n  ✅ Tuned Best Model ({best_name}):")
print(f"     Test R²   = {r2_tuned:.4f}")
print(f"     Test RMSE = {rmse_tuned:.5f} Pa·s")
print(f"     Cls Acc   = {acc_tuned*100:.2f}%\n")

print("=== STEP 7: SHAP Explainability for Tuned Best Model ===\n")

# Use a subsample for speed (500 points from test set)
N_SHAP = min(500, len(X_test_sc))
X_shap = X_test_sc[:N_SHAP]

if best_name in ("Random Forest", "XGBoost", "CatBoost"):
    explainer   = shap.TreeExplainer(tuned_reg)
    shap_values = explainer.shap_values(X_shap)
else:
    background  = shap.kmeans(X_train_sc, 50)
    explainer   = shap.KernelExplainer(tuned_reg.predict, background)
    shap_values = explainer.shap_values(X_shap, nsamples=100)

# Global feature importance from SHAP
mean_shap_abs = np.abs(shap_values).mean(axis=0)
shap_importance = pd.Series(mean_shap_abs, index=FEATURES).sort_values(ascending=False)
top_features    = shap_importance.index.tolist()

print(f"  SHAP Feature Importance ranking:")
for i, (feat, val) in enumerate(shap_importance.items()):
    print(f"    {i+1}. {feat:25s} → {val:.5f}")

feat1, feat2 = top_features[0], top_features[1]
print(f"\n  📌 The most important feature for viscosity prediction is "
      f"[{feat1}], followed by [{feat2}].")

# ── Plot 8: SHAP Beeswarm ─────────────────────────────────────────────────────
print("\n  Generating SHAP plots...")
import matplotlib
matplotlib.rcParams.update({
    "figure.facecolor": BG_COLOR, "axes.facecolor": BG_COLOR,
    "text.color": TEXT_COLOR, "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR, "ytick.color": TEXT_COLOR,
})

shap_exp = shap.Explanation(
    values=shap_values,
    base_values=np.array([explainer.expected_value] * N_SHAP)
               if hasattr(explainer, "expected_value") else
               np.zeros(N_SHAP),
    data=X_shap,
    feature_names=FEATURES,
)

fig_shap_bee, ax_bee = plt.subplots(figsize=(11, 7), facecolor=BG_COLOR)
shap.plots.beeswarm(shap_exp, show=False, plot_size=None)
ax_bee = plt.gca()
ax_bee.set_facecolor(BG_COLOR)
fig_shap_bee = plt.gcf()
fig_shap_bee.set_facecolor(BG_COLOR)
plt.title(f"SHAP Beeswarm — Tuned {best_name}", color=TEXT_COLOR,
          fontsize=13, fontweight="bold", pad=10)
plt.tight_layout()
p8_path = os.path.join(SAVE_DIR, "plot8_shap_beeswarm.png")
plt.savefig(p8_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print(f"  ✅ Plot 8 saved: SHAP Beeswarm")

# ── Plot: SHAP Bar (mean |SHAP|) ──────────────────────────────────────────────
fig_bar, ax_bar = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
shap_importance_sorted = shap_importance.sort_values()
colors_bar = [ACCENT if v > shap_importance_sorted.median() else "#8a7fad"
              for v in shap_importance_sorted]
ax_bar.barh(shap_importance_sorted.index, shap_importance_sorted.values, color=colors_bar)
ax_bar.set_xlabel("Mean |SHAP value| (impact on viscosity)", color=TEXT_COLOR)
ax_bar.set_title(f"Global Feature Importance — SHAP (Tuned {best_name})",
                 color=TEXT_COLOR, fontsize=13, fontweight="bold")
ax_bar.set_facecolor(BG_COLOR)
ax_bar.tick_params(colors=TEXT_COLOR)
for spine in ax_bar.spines.values():
    spine.set_edgecolor(ACCENT)
for i, v in enumerate(shap_importance_sorted.values):
    ax_bar.text(v + 0.00001, i, f"{v:.5f}", va="center", color=TEXT_COLOR, fontsize=9)
plt.tight_layout()
p_shap_bar_path = os.path.join(SAVE_DIR, "plot_shap_bar.png")
plt.savefig(p_shap_bar_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
print(f"  ✅ SHAP Bar plot saved")

# ── SHAP Dependence plots for top 2 features ──────────────────────────────────
for feat in [feat1, feat2]:
    feat_idx = FEATURES.index(feat)
    fig_dep, ax_dep = plt.subplots(figsize=(9, 5), facecolor=BG_COLOR)
    ax_dep.set_facecolor(BG_COLOR)
    sc = ax_dep.scatter(
        X_shap[:, feat_idx],
        shap_values[:, feat_idx],
        c=X_shap[:, feat_idx],
        cmap="plasma", alpha=0.6, s=15
    )
    cbar = plt.colorbar(sc, ax=ax_dep)
    cbar.ax.tick_params(colors=TEXT_COLOR)
    cbar.set_label(feat, color=TEXT_COLOR)
    ax_dep.axhline(0, color="white", lw=0.8, ls="--")
    ax_dep.set_xlabel(feat, color=TEXT_COLOR)
    ax_dep.set_ylabel("SHAP value (impact on viscosity)", color=TEXT_COLOR)
    ax_dep.set_title(f"SHAP Dependence — {feat}  [Tuned {best_name}]",
                     color=TEXT_COLOR, fontsize=12, fontweight="bold")
    ax_dep.tick_params(colors=TEXT_COLOR)
    for spine in ax_dep.spines.values():
        spine.set_edgecolor(ACCENT)
    plt.tight_layout()
    safe_feat = feat.replace("/", "_").replace(" ", "_")
    dep_path = os.path.join(SAVE_DIR, f"plot_shap_dep_{safe_feat}.png")
    plt.savefig(dep_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  ✅ SHAP Dependence plot saved for {feat} → {dep_path}")

print("\n✅ SHAP explainability complete.\n")

print("=== STEP 8: Full Visualization Suite (Plots 1–9) ===")

matplotlib.rcParams.update({
    "figure.facecolor": BG_COLOR, "axes.facecolor": BG_COLOR,
    "text.color": TEXT_COLOR, "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR, "ytick.color": TEXT_COLOR,
    "axes.edgecolor": ACCENT, "grid.color": "#333355",
})

# ── Plot 1: RF Feature Importance (updated, 7 features) ───────────────────────
fig, ax = styled_fig(9, 6)
feat_imp_rf = pd.Series(rf_reg.feature_importances_, index=FEATURES).sort_values()
bar_colors  = [ACCENT if v >= feat_imp_rf.median() else "#5a6478" for v in feat_imp_rf]
feat_imp_rf.plot(kind="barh", ax=ax, color=bar_colors)
ax.set_title("Feature Importance — Random Forest Regressor (7 Features)",
             fontsize=13, fontweight="bold", color=TEXT_COLOR)
ax.set_xlabel("Importance Score", color=TEXT_COLOR)
for i, v in enumerate(feat_imp_rf):
    ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=9, color=TEXT_COLOR)
plt.tight_layout()
p1_path = os.path.join(SAVE_DIR, "plot1_rf_importance.png")
plt.savefig(p1_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print(f"  ✅ Plot 1 saved: RF Feature Importance → {p1_path}")

# ── Plot 2: Predicted vs Actual — all 4 models + Tuned ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG_COLOR)
model_colors = {"Random Forest": "#4a9eff", "XGBoost": ACCENT,
                "CatBoost": GREEN_COLOR, "Neural Network": "#c77dff"}
mn, mx = yr_test.min(), yr_test.max()

ax_l, ax_r = axes
for ax in [ax_l, ax_r]:
    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(ACCENT)
    ax.tick_params(colors=TEXT_COLOR)

for name, col in model_colors.items():
    ax_l.scatter(yr_test, results[name]["yr_pred"],
                 alpha=0.3, s=10, color=col, label=f"{name} (R²={results[name]['test_r2']:.3f})")
ax_l.plot([mn, mx], [mn, mx], "w--", lw=1.5, label="Perfect")
ax_l.set_xlabel("Actual Viscosity (Pa·s)"); ax_l.set_ylabel("Predicted (Pa·s)")
ax_l.set_title("All 4 Models — Predicted vs Actual", color=TEXT_COLOR, fontweight="bold")
ax_l.legend(fontsize=8, facecolor="#22223a", labelcolor=TEXT_COLOR)

ax_r.scatter(yr_test, yr_pred_tuned, alpha=0.5, s=12,
             color=ACCENT, label=f"Tuned {best_name} (R²={r2_tuned:.3f})")
ax_r.plot([mn, mx], [mn, mx], "w--", lw=1.5, label="Perfect")
ax_r.set_xlabel("Actual Viscosity (Pa·s)"); ax_r.set_ylabel("Predicted (Pa·s)")
ax_r.set_title(f"Tuned {best_name} — Best Model", color=TEXT_COLOR, fontweight="bold")
ax_r.legend(fontsize=9, facecolor="#22223a", labelcolor=TEXT_COLOR)

plt.suptitle("Predicted vs Actual Viscosity Comparison", color=TEXT_COLOR,
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
p2_path = os.path.join(SAVE_DIR, "plot2_pred_vs_actual.png")
plt.savefig(p2_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print(f"  ✅ Plot 2 saved: Predicted vs Actual → {p2_path}")

# ── Plot 3: Viscosity vs Temperature, coloured by class ───────────────────────
color_map_cls = {"Increase RPM": RED_COLOR, "Maintain RPM": GREEN_COLOR, "Reduce RPM": BLUE_COLOR}
fig, ax = styled_fig(10, 6)
for cls_label, grp in df.groupby("Disc_Recommendation"):
    ax.scatter(grp["Temperature_C"], grp["Viscosity_Pas"],
               alpha=0.35, s=10, color=color_map_cls[cls_label], label=cls_label)
ax.axhline(LOW_THRESH,  color=ACCENT, lw=1.8, ls="--",
           label=f"Lower threshold ({LOW_THRESH} Pa·s)")
ax.axhline(HIGH_THRESH, color="tomato", lw=1.8, ls="--",
           label=f"Upper threshold ({HIGH_THRESH} Pa·s)")
ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Viscosity (Pa·s)")
ax.set_title("Viscosity vs Temperature — Coloured by Disc Recommendation",
             fontsize=13, fontweight="bold", color=TEXT_COLOR)
ax.legend(markerscale=2, fontsize=9, facecolor="#22223a", labelcolor=TEXT_COLOR)
plt.tight_layout()
p3_path = os.path.join(SAVE_DIR, "plot3_viscosity_vs_temp_classes.png")
plt.savefig(p3_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print(f"  ✅ Plot 3 saved: Viscosity vs Temp (Classes) → {p3_path}")

# ── Plot 4: NN Learning Curves ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG_COLOR)
nn_data = results["Neural Network"]
for ax, hist, title in zip(
    axes,
    [nn_data["history_reg"], nn_data["history_cls"]],
    ["Regression — MSE Loss", "Classification — Cross-Entropy Loss"]
):
    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values(): spine.set_edgecolor(ACCENT)
    ax.tick_params(colors=TEXT_COLOR)
    ax.plot(hist.history["loss"],     label="Train loss", color=BLUE_COLOR, lw=2)
    ax.plot(hist.history["val_loss"], label="Val loss",   color=ACCENT,    lw=2, ls="--")
    ax.set_title(f"NN: {title}", color=TEXT_COLOR, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(facecolor="#22223a", labelcolor=TEXT_COLOR)
plt.suptitle("Neural Network Learning Curves", color=TEXT_COLOR,
             fontsize=14, fontweight="bold")
plt.tight_layout()
p4_path = os.path.join(SAVE_DIR, "plot4_nn_learning_curves.png")
plt.savefig(p4_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print(f"  ✅ Plot 4 saved: NN Learning Curves → {p4_path}")

# ── Plot 5: Confusion Matrices for all 4 models ───────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor=BG_COLOR)
for ax, (name, m) in zip(axes, results.items()):
    ax.set_facecolor(BG_COLOR)
    cm = confusion_matrix(yc_test, m["yc_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                linewidths=0.5, linecolor=BG_COLOR)
    ax.set_title(f"{name}\n(Acc {m['acc']*100:.1f}%)",
                 color=TEXT_COLOR, fontweight="bold", fontsize=10)
    ax.set_xlabel("Predicted", color=TEXT_COLOR)
    ax.set_ylabel("Actual", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
plt.suptitle("Confusion Matrices — All Four Models", color=TEXT_COLOR,
             fontsize=14, fontweight="bold")
plt.tight_layout()
p5_path = os.path.join(SAVE_DIR, "plot5_confusion_matrices.png")
plt.savefig(p5_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print(f"  ✅ Plot 5 saved: Confusion Matrices → {p5_path}")

# ── Plot 6: Viscosity Distribution Histogram with 3 zones ────────────────────
fig, ax = styled_fig(10, 6)
ax.hist(df["Viscosity_Pas"], bins=80, color="#7b61ff", edgecolor="none", alpha=0.85)

ax.axvspan(df["Viscosity_Pas"].min(), LOW_THRESH,
           alpha=0.25, color=BLUE_COLOR, label=f"Reduce RPM (< {LOW_THRESH})")
ax.axvspan(LOW_THRESH, HIGH_THRESH,
           alpha=0.25, color=GREEN_COLOR, label=f"Maintain RPM ({LOW_THRESH}–{HIGH_THRESH})")
ax.axvspan(HIGH_THRESH, df["Viscosity_Pas"].max(),
           alpha=0.25, color=RED_COLOR, label=f"Increase RPM (> {HIGH_THRESH})")

ax.axvline(LOW_THRESH,  color=BLUE_COLOR,  lw=2, ls="--")
ax.axvline(HIGH_THRESH, color=RED_COLOR,   lw=2, ls="--")

ax.set_xlabel("Viscosity (Pa·s)"); ax.set_ylabel("Count")
ax.set_title("Viscosity Distribution with RPM Control Zones",
             fontsize=13, fontweight="bold", color=TEXT_COLOR)
ax.legend(facecolor="#22223a", labelcolor=TEXT_COLOR)
plt.tight_layout()
p6_path = os.path.join(SAVE_DIR, "plot6_viscosity_histogram.png")
plt.savefig(p6_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print(f"  ✅ Plot 6 saved: Viscosity Histogram → {p6_path}")

# ── Plot 7: Correlation Heatmap (7 features + viscosity) ──────────────────────
fig, ax = styled_fig(11, 9)
corr_cols = FEATURES + ["Viscosity_Pas"]
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=ax, linewidths=0.5, linecolor="#0d0d1a",
            annot_kws={"size": 9, "color": TEXT_COLOR},
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap (7 Features + Viscosity)",
             color=TEXT_COLOR, fontsize=13, fontweight="bold")
ax.tick_params(colors=TEXT_COLOR, labelsize=9)
plt.tight_layout()
p7_path = os.path.join(SAVE_DIR, "plot7_correlation_heatmap.png")
plt.savefig(p7_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print(f"  ✅ Plot 7 saved: Correlation Heatmap → {p7_path}")

# (Plot 8 was saved during SHAP step above)

# ── Plot 9: CV R² Comparison Bar Chart with Error Bars ───────────────────────
fig, ax = styled_fig(10, 6)
model_names = list(results.keys())
cv_means    = [results[n]["cv_r2"].mean() for n in model_names]
cv_stds     = [results[n]["cv_r2"].std()  for n in model_names]
bar_cols    = [BLUE_COLOR, ACCENT, GREEN_COLOR, "#c77dff"]
bars = ax.bar(model_names, cv_means, color=bar_cols,
              yerr=cv_stds, capsize=6,
              error_kw={"ecolor": TEXT_COLOR, "lw": 2, "capthick": 2},
              edgecolor=TEXT_COLOR, linewidth=1.2, width=0.55)
ax.set_ylim(max(0, min(cv_means) - 0.05), min(1.0, max(cv_means) + 0.08))
ax.set_ylabel("5-Fold CV R²")
ax.set_title("Cross-Validation R² Comparison — All Four Models (mean ± std)",
             fontsize=12, fontweight="bold", color=TEXT_COLOR)
for bar, mean, std in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.003,
            f"{mean:.4f}\n±{std:.4f}", ha="center", va="bottom",
            fontsize=9, color=TEXT_COLOR, fontweight="bold")
ax.set_xticklabels(model_names, rotation=10, ha="right")
ax.yaxis.grid(True, color="#333355", lw=0.8)
plt.tight_layout()
p9_path = os.path.join(SAVE_DIR, "plot9_cv_r2_comparison.png")
plt.savefig(p9_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print(f"  ✅ Plot 9 saved: CV R2 Comparison → {p9_path}")

print("\n✅ All 9 plots saved to disk.\n")

print("=== STEP 9: Saving Models to Disk ===")

joblib.dump(rf_reg,    os.path.join(SAVE_DIR, "model_rf_reg.joblib"))
joblib.dump(rf_cls,    os.path.join(SAVE_DIR, "model_rf_cls.joblib"))
joblib.dump(xgb_reg,   os.path.join(SAVE_DIR, "model_xgb_reg.joblib"))
joblib.dump(xgb_cls,   os.path.join(SAVE_DIR, "model_xgb_cls.joblib"))
joblib.dump(cat_reg,   os.path.join(SAVE_DIR, "model_cat_reg.joblib"))
joblib.dump(cat_cls,   os.path.join(SAVE_DIR, "model_cat_cls.joblib"))
joblib.dump(scaler,    os.path.join(SAVE_DIR, "scaler.joblib"))
joblib.dump(le,        os.path.join(SAVE_DIR, "label_encoder.joblib"))

nn_reg.save(os.path.join(SAVE_DIR, "model_nn_reg.keras"))
nn_cls.save(os.path.join(SAVE_DIR, "model_nn_cls.keras"))
tuned_reg.save(os.path.join(SAVE_DIR, "model_tuned_best.keras")) \
    if best_name == "Neural Network" else \
    joblib.dump(tuned_reg, os.path.join(SAVE_DIR, "model_tuned_best.joblib"))
tuned_cls.save(os.path.join(SAVE_DIR, "model_tuned_cls.keras")) \
    if best_name == "Neural Network" else \
    joblib.dump(tuned_cls, os.path.join(SAVE_DIR, "model_tuned_cls.joblib"))

# Save metadata
import json
metadata = {
    "best_model_name":  best_name,
    "features":         FEATURES,
    "class_names":      list(class_names),
    "low_thresh":       LOW_THRESH,
    "high_thresh":      HIGH_THRESH,
    "feat1":            feat1,
    "feat2":            feat2,
    "shap_importance":  {k: float(v) for k, v in shap_importance.items()},
    "tuned_params":     study.best_params,
    "r2_tuned":         float(r2_tuned),
    "rmse_tuned":       float(rmse_tuned),
}
with open(os.path.join(SAVE_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("  ✅ All models saved.")
print("  ✅ Metadata saved to model_metadata.json\n")

print("=== STEP 9B: Training PCA Anomaly Detector ===")
from sklearn.decomposition import PCA

# Use n_components=5 (drop last 2 PCs which carry noise, not signal).
# Reconstruction error on those dropped PCs = distance from training manifold.
pca_detector = PCA(n_components=5, random_state=SEED)
pca_detector.fit(X_train_sc)

var_explained = pca_detector.explained_variance_ratio_.sum()
print(f"  Explained variance (5 PCs): {var_explained*100:.2f}%")

# Compute per-sample Mean Squared Reconstruction Error on training set
X_recon_train = pca_detector.inverse_transform(pca_detector.transform(X_train_sc))
recon_errors_train = np.mean((X_train_sc - X_recon_train) ** 2, axis=1)

anomaly_mean = float(recon_errors_train.mean())
anomaly_std  = float(recon_errors_train.std())
anomaly_threshold_3sigma = anomaly_mean + 3.0 * anomaly_std  # 3-sigma rule: flags ~0.3% of valid data

print(f"  Training recon error  — mean: {anomaly_mean:.6f}  std: {anomaly_std:.6f}")
print(f"  Anomaly threshold (mean + 3σ): {anomaly_threshold_3sigma:.6f}")

# Quick sanity check on test set
X_recon_test = pca_detector.inverse_transform(pca_detector.transform(X_test_sc))
recon_errors_test = np.mean((X_test_sc - X_recon_test) ** 2, axis=1)
test_anomaly_rate = float((recon_errors_test > anomaly_threshold_3sigma).mean())
print(f"  Test-set false-positive rate: {test_anomaly_rate*100:.2f}% (expected ≤1%)")

joblib.dump(pca_detector, os.path.join(SAVE_DIR, "pca_detector.joblib"))

anomaly_cfg = {
    "n_components": 5,
    "explained_variance": round(var_explained, 6),
    "threshold": anomaly_threshold_3sigma,
    "mean_error": anomaly_mean,
    "std_error":  anomaly_std,
    "explained_variance_ratio": pca_detector.explained_variance_ratio_.tolist(),
}
with open(os.path.join(SAVE_DIR, "anomaly_config.json"), "w") as f:
    json.dump(anomaly_cfg, f, indent=2)

print("  ✅ pca_detector.joblib + anomaly_config.json saved.")
print("\n  ℹ️  How it works in production (app.py):")
print("     → Each operator input is projected onto 5 PCs then reconstructed.")
print("     → If MSE > threshold, input is outside the training manifold.")
print("     → A warning is shown in the UI before the prediction is displayed.\n")

print("=== STEP 10: Launching Gradio Demo ===")

import gradio as gr
import os

HF_TOKEN = os.environ.get("HF_TOKEN", "hf_YOUR_TOKEN_HERE")

# ── Helper functions ──────────────────────────────────────────────────────────
def compute_optical_basicity(basicity_val, al2o3, mgo,
                              sio2_base=35.0):
    """Compute Λ from composition."""
    SiO2_w = sio2_base
    CaO_w  = basicity_val * SiO2_w
    M_CaO_m=56.08; M_SiO2_m=60.09; M_Al2O3_m=101.96; M_MgO_m=40.30
    n_CaO   = CaO_w  / M_CaO_m
    n_SiO2  = SiO2_w / M_SiO2_m
    n_Al2O3 = al2o3  / M_Al2O3_m
    n_MgO   = mgo    / M_MgO_m
    n_tot   = n_CaO + n_SiO2 + n_Al2O3 + n_MgO
    X_CaO_   = n_CaO / n_tot
    X_SiO2_  = n_SiO2 / n_tot
    X_Al2O3_ = n_Al2O3 / n_tot
    X_MgO_   = n_MgO / n_tot
    return X_CaO_*1.00 + X_SiO2_*0.48 + X_Al2O3_*0.60 + X_MgO_*0.78

def get_recommendation(visc):
    if visc < LOW_THRESH:
        return "Reduce RPM"
    elif visc <= HIGH_THRESH:
        return "Maintain RPM"
    else:
        return "Increase RPM"

def confidence_bar(visc):
    if LOW_THRESH <= visc <= HIGH_THRESH:
        mid  = (LOW_THRESH + HIGH_THRESH) / 2
        dist = abs(visc - mid) / (mid - LOW_THRESH)
        pct  = int((1 - dist) * 100)
        bar  = "🟩" * (pct // 10) + "⬜" * (10 - pct // 10)
        return f"{bar}  {pct}% — OPTIMAL WINDOW ✅"
    elif visc < LOW_THRESH:
        pct = int((visc / LOW_THRESH) * 100)
        bar = "🟦" * (pct // 10) + "⬜" * (10 - pct // 10)
        return f"{bar}  Slag too fluid ({pct}% toward optimal)"
    else:
        excess = min(visc - HIGH_THRESH, 0.5)
        pct    = max(0, 100 - int((excess / 0.5) * 100))
        bar    = "🟥" * (pct // 10) + "⬜" * (10 - pct // 10)
        return f"{bar}  Slag too viscous ({pct}% toward optimal)"

def rule_based_explanation(temp, basicity_val, al2o3, mgo, coke, tap, ob, visc, rec):
    """Fallback explanation if LLM is unavailable."""
    msgs = []
    if al2o3 > 14:
        msgs.append(f"High Al₂O₃ ({al2o3:.1f} wt%) is raising viscosity — "
                    f"alumina acts as a network former.")
    if basicity_val < 1.0:
        msgs.append(f"Low basicity ({basicity_val:.2f}) means insufficient CaO to "
                    f"depolymerize the silicate network.")
    if temp < 1460:
        msgs.append(f"Temperature ({temp:.0f}°C) is near the lower bound — "
                    f"slag may be cooling.")
    if tap > 60:
        msgs.append(f"High tap time ({tap:.0f} min) suggests slag cooling progression.")
    if ob < 0.62:
        msgs.append(f"Optical basicity Λ={ob:.3f} is relatively low — "
                    f"slag network is more polymerised, raising viscosity.")
    if not msgs:
        msgs.append(f"Slag composition is balanced. Optical basicity Λ={ob:.3f} "
                    f"is within a healthy range.")
    actions = {
        "Increase RPM": "⚠️ Recommend increasing disc RPM by ~200–300 RPM to handle viscous slag.",
        "Maintain RPM": "✅ Disc RPM is optimal. No adjustment needed.",
        "Reduce RPM":   "🔵 Slag is too fluid — recommend reducing disc RPM by ~150–200 RPM.",
    }
    return " ".join(msgs) + " " + actions.get(rec, "")

def llm_explanation(temp, basicity_val, al2o3, mgo, coke, tap, ob, visc, rec):
    """Call HuggingFace LLM for expert metallurgist explanation."""
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(
            model="Qwen/Qwen2.5-7B-Instruct",
            token=HF_TOKEN
        )
        system_prompt = (
            "You are an expert blast furnace metallurgist with 20 years of experience at "
            "Tata Steel. You specialize in DCG (Dry Centrifugal Granulation) slag heat "
            "recovery. Your job is to explain slag viscosity predictions to plant operators "
            "in clear, practical language. Always be specific, cite which input parameters "
            "are most problematic, and give a concrete disc RPM recommendation. "
            "Keep your response under 100 words."
        )
        user_prompt = (
            f"Slag temperature: {temp:.0f}°C. CaO/SiO₂ basicity: {basicity_val:.2f}. "
            f"Al₂O₃: {al2o3:.1f} wt%. MgO: {mgo:.1f} wt%. Coke rate: {coke:.0f} kg/t iron. "
            f"Tap time: {tap:.0f} minutes. Optical basicity: {ob:.3f}. "
            f"Predicted viscosity: {visc:.4f} Pa·s. Disc recommendation: {rec}. "
            f"Explain what is happening and what the operator should do."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        response = client.chat_completion(messages=messages, max_tokens=200)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return rule_based_explanation(temp, basicity_val, al2o3, mgo, coke, tap, ob, visc, rec)

# ── Main prediction function ──────────────────────────────────────────────────
def predict_all(temp, basicity_val, al2o3, mgo, coke, tap, ob_manual):
    # Auto-compute optical basicity (override manual slider)
    ob = compute_optical_basicity(basicity_val, al2o3, mgo)

    inp    = np.array([[temp, basicity_val, al2o3, mgo, coke, tap, ob]])
    inp_sc = scaler.transform(inp)

    # All model predictions
    visc_rf  = float(rf_reg.predict(inp_sc)[0])
    visc_xgb = float(xgb_reg.predict(inp_sc)[0])
    visc_cat = float(cat_reg.predict(inp_sc)[0])
    visc_nn  = float(nn_reg.predict(inp_sc, verbose=0).flatten()[0])

    # Tuned best
    visc_tuned = tuned_reg.predict(inp_sc)
    visc_tuned = float(visc_tuned.flatten()[0]) if best_name == "Neural Network" \
                 else float(visc_tuned[0])

    # RPM recommendation from tuned model
    rec = get_recommendation(visc_tuned)

    emoji_map = {
        "Increase RPM": "🔴 INCREASE RPM",
        "Maintain RPM": "🟢 MAINTAIN RPM",
        "Reduce RPM":   "🔵 REDUCE RPM",
    }

    conf     = confidence_bar(visc_tuned)
    exp_text = rule_based_explanation(temp, basicity_val, al2o3, mgo, coke, tap,
                                      ob, visc_tuned, rec)

    return (
        f"{ob:.4f}",                          # auto optical basicity
        "✅ Data within expected training distribution.", # anomaly check placeholder
        f"{visc_rf:.5f} Pa·s",
        f"{visc_xgb:.5f} Pa·s",
        f"{visc_cat:.5f} Pa·s",
        f"{visc_nn:.5f} Pa·s",
        f"⭐ {visc_tuned:.5f} Pa·s  ← Tuned {best_name}",
        emoji_map.get(rec, rec),
        conf,
        exp_text,
    )

def generate_expert_report(temp, basicity_val, al2o3, mgo, coke, tap, ob_manual):
    ob     = compute_optical_basicity(basicity_val, al2o3, mgo)
    inp    = np.array([[temp, basicity_val, al2o3, mgo, coke, tap, ob]])
    inp_sc = scaler.transform(inp)
    visc   = float(tuned_reg.predict(inp_sc).flatten()[0]) \
             if best_name == "Neural Network" \
             else float(tuned_reg.predict(inp_sc)[0])
    rec    = get_recommendation(visc)
    return llm_explanation(temp, basicity_val, al2o3, mgo, coke, tap, ob, visc, rec)

# ── Custom CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Base Spacing & Layout ── */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}
/* ── Typography & Spacing ── */
h1, h2, h3 {
    margin-top: 0.2em !important;
    margin-bottom: 0.4em !important;
}
.prose p {
    line-height: 1.6 !important;
    margin-bottom: 1em !important;
}
/* ── Blocks & Padding ── */
.block, .gr-box, .gr-form {
    padding: 20px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
}
/* ── Labels ── */
label > span {
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    margin-bottom: 6px !important;
    display: inline-block !important;
}
/* ── Buttons ── */
button.primary {
    transition: all 0.2s ease !important;
    font-weight: 600 !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
}
"""

# ── Load SHAP bar image if available ──────────────────────────────────────────
SHAP_IMG = "plot_shap_bar.png" if os.path.exists("plot_shap_bar.png") else None

# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="DCG Slag Viscosity Controller — IIT ISM Dhanbad",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    css=CUSTOM_CSS,
) as demo:

    gr.Markdown("""
# 🏭 DCG Blast Furnace Slag — Real-Time Viscosity & RPM Controller
### IIT (ISM) Dhanbad · Mineral & Metallurgical Engineering · Innovation 1
> **4 ML models · Bayesian-Optuna tuning · SHAP explainability · PCA anomaly detection · Qwen2.5-7B expert reports**
---
""")

    with gr.Row():
        # ── Panel 1: Inputs ──────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### 🌡️ Slag Input Parameters")
            temp_sl  = gr.Slider(1400, 1600, value=1500, step=1,
                                  label="Temperature (°C)",
                                  info="Typical blast furnace tap: 1450–1550°C")
            bas_sl   = gr.Slider(0.8, 1.4, value=1.1, step=0.01,
                                  label="Basicity  CaO/SiO₂",
                                  info="Higher = CaO breaks silicate network → lower viscosity")
            al_sl    = gr.Slider(8, 18, value=13, step=0.1,
                                  label="Al₂O₃ (wt%)",
                                  info="Network former — raises viscosity (Xin et al. 2025)")
            mgo_sl   = gr.Slider(4, 12, value=8, step=0.1,
                                  label="MgO (wt%)",
                                  info="Network modifier — slightly reduces viscosity")
            coke_sl  = gr.Slider(450, 550, value=500, step=1,
                                  label="Coke Rate (kg/t iron)",
                                  info="Proxy for furnace heat input")
            tap_sl   = gr.Slider(0, 90, value=30, step=1,
                                  label="Tap Time (minutes)",
                                  info="Elapsed time since tap started — slag cools over time")
            ob_sl    = gr.Slider(0.55, 0.75, value=0.64, step=0.001,
                                  label="Optical Basicity Λ  (display only — auto-computed)",
                                  info="Duffy & Ingram (1976) — computed from mole fractions")
            gr.Markdown("*Λ auto-computed from Basicity, Al₂O₃, MgO. Slider is read-only.*")

            with gr.Row():
                predict_btn = gr.Button("🔍  Predict Viscosity & RPM", variant="primary")
            with gr.Row():
                report_btn  = gr.Button("🤖  Generate Expert LLM Report", variant="secondary")

        # ── Panel 2: ML Predictions ──────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### 📊 ML Predictions")
            anomaly_out = gr.Textbox(label="🔬 Data Quality Check (PCA Anomaly Detector)",
                                      lines=3, interactive=False)
            ob_out      = gr.Textbox(label="🔵 Auto-Computed Optical Basicity Λ",
                                      interactive=False)
            vrf_out     = gr.Textbox(label="🌲 Random Forest",       interactive=False)
            vxgb_out    = gr.Textbox(label="⚡ XGBoost",              interactive=False)
            vcat_out    = gr.Textbox(label="🐱 CatBoost",             interactive=False)
            vnn_out     = gr.Textbox(label="🧠 Neural Network",       interactive=False)
            vbest_out   = gr.Textbox(label="⭐ Tuned Best Model  (Primary Prediction)",
                                      interactive=False)
            rec_out     = gr.Textbox(label="💿 Disc RPM Recommendation", interactive=False)
            conf_out    = gr.Textbox(label="📈 Optimal Window Proximity", interactive=False)

        # ── Panel 3: Expert Explanation ──────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### 💬 Expert Explanation")
            exp_out = gr.Textbox(
                label="⚡ Instant Analysis  (SHAP attribution + rule-based)",
                lines=9, interactive=False,
            )
            llm_out = gr.Textbox(
                label="🤖 Qwen2.5-7B Metallurgist Report  (SHAP-enriched LLM prompt)",
                lines=9, interactive=False,
                placeholder="Click '🤖 Generate Expert LLM Report' above...",
            )
            gr.Markdown("""
**Optimal Viscosity Window:** 0.055 – 0.080 Pa·s  
< 0.055 → Reduce RPM (too fluid) | > 0.080 → Increase RPM (too viscous)  
*SHAP prompt methodology: Xin et al. 2025 (BO-CatBoost+SHAP), Chen et al. 2026 (optical basicity)*
""")

    # ── SHAP global importance image ─────────────────────────────────────────
    if SHAP_IMG:
        gr.Markdown("---\n### 📌 Global SHAP Feature Importance  (Tuned Best Model)")
        gr.Image(SHAP_IMG, show_label=False, height=400)

    # ── Wire up inputs / outputs ──────────────────────────────────────────────
    INPUTS  = [temp_sl, bas_sl, al_sl, mgo_sl, coke_sl, tap_sl, ob_sl]
    OUTPUTS = [ob_out, anomaly_out, vrf_out, vxgb_out, vcat_out, vnn_out,
               vbest_out, rec_out, conf_out, exp_out]

    predict_btn.click(fn=predict_all,        inputs=INPUTS, outputs=OUTPUTS)
    report_btn.click(fn=generate_expert_report, inputs=INPUTS, outputs=[llm_out])

    # Live update on every slider change
    for sl in INPUTS:
        sl.change(fn=predict_all, inputs=INPUTS, outputs=OUTPUTS)

demo.launch(share=True)
