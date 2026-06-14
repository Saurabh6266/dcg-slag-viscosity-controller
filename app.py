# ============================================================
#  app.py — HuggingFace Space: DCG Slag Viscosity Controller
#  v2.1: + PCA Anomaly Detector  + SHAP-to-LLM Prompt
#  IIT (ISM) Dhanbad | Mineral & Metallurgical Engineering
# ============================================================
import os, json, warnings
import numpy as np
import gradio as gr
import joblib

warnings.filterwarnings("ignore")

# ── Optional heavy imports (graceful degradation if unavailable) ──────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

HF_TOKEN = os.environ.get("HF_TOKEN", "hf_YOUR_TOKEN_HERE")

# ── Load metadata ──────────────────────────────────────────────────────────────
with open("model_metadata.json") as f:
    meta = json.load(f)

FEATURES     = meta["features"]
class_names  = meta["class_names"]
LOW_THRESH   = meta["low_thresh"]
HIGH_THRESH  = meta["high_thresh"]
best_name    = meta["best_model_name"]
feat1        = meta["feat1"]
feat2        = meta["feat2"]
global_shap  = meta.get("shap_importance", {})   # pre-computed global SHAP importance

# ── Load anomaly config ────────────────────────────────────────────────────────
with open("anomaly_config.json") as f:
    anomaly_cfg = json.load(f)

# ── Load models ────────────────────────────────────────────────────────────────
print("Loading models...")
scaler      = joblib.load("scaler.joblib")
le          = joblib.load("label_encoder.joblib")
rf_reg      = joblib.load("model_rf_reg.joblib")
xgb_reg     = joblib.load("model_xgb_reg.joblib")
cat_reg     = joblib.load("model_cat_reg.joblib")
pca_detector = joblib.load("pca_detector.joblib")

if TF_AVAILABLE:
    nn_reg = keras.models.load_model("model_nn_reg.keras")
else:
    nn_reg = None

if best_name == "Neural Network" and TF_AVAILABLE:
    tuned_reg = keras.models.load_model("model_tuned_best.keras")
else:
    tuned_reg = joblib.load("model_tuned_best.joblib")

print(f"✅ All models loaded. Best model: {best_name}")

# ── SHAP Explainer (fast TreeExplainer for tree models) ────────────────────────
shap_explainer = None
if SHAP_AVAILABLE and best_name in ("Random Forest", "XGBoost", "CatBoost"):
    try:
        shap_explainer = shap.TreeExplainer(tuned_reg)
        print("✅ SHAP TreeExplainer ready for real-time explanations.")
    except Exception as e:
        print(f"⚠️ SHAP explainer init failed: {e}. Will use global importance fallback.")

# ── Helper: Optical Basicity ───────────────────────────────────────────────────
def compute_optical_basicity(basicity_val, al2o3, mgo, sio2_base=35.0):
    """Compute Optical Basicity Λ from mole fractions (Duffy & Ingram, 1976)."""
    CaO_w  = basicity_val * sio2_base
    n_CaO   = CaO_w    / 56.08
    n_SiO2  = sio2_base / 60.09
    n_Al2O3 = al2o3     / 101.96
    n_MgO   = mgo       / 40.30
    n_tot   = n_CaO + n_SiO2 + n_Al2O3 + n_MgO
    X_CaO   = n_CaO   / n_tot
    X_SiO2  = n_SiO2  / n_tot
    X_Al2O3 = n_Al2O3 / n_tot
    X_MgO   = n_MgO   / n_tot
    return X_CaO*1.00 + X_SiO2*0.48 + X_Al2O3*0.60 + X_MgO*0.78

# ── Helper: RPM class ──────────────────────────────────────────────────────────
def get_rec(v):
    if v < LOW_THRESH:   return "Reduce RPM"
    if v <= HIGH_THRESH: return "Maintain RPM"
    return "Increase RPM"

# ── Helper: Confidence bar ─────────────────────────────────────────────────────
def confidence_bar(visc):
    if LOW_THRESH <= visc <= HIGH_THRESH:
        mid  = (LOW_THRESH + HIGH_THRESH) / 2
        dist = abs(visc - mid) / (mid - LOW_THRESH)
        pct  = int((1 - dist) * 100)
        return f"{'🟩'*(pct//10)}{'⬜'*(10-pct//10)}  {pct}% — OPTIMAL WINDOW ✅"
    elif visc < LOW_THRESH:
        pct = int((visc / LOW_THRESH) * 100)
        return f"{'🟦'*(pct//10)}{'⬜'*(10-pct//10)}  Slag too fluid ({pct}% toward optimal)"
    else:
        excess = min(visc - HIGH_THRESH, 0.5)
        pct    = max(0, 100 - int((excess / 0.5) * 100))
        return f"{'🟥'*(pct//10)}{'⬜'*(10-pct//10)}  Slag too viscous ({pct}% toward optimal)"

# ── PCA Anomaly Detector ───────────────────────────────────────────────────────
def detect_anomaly(inp_sc: np.ndarray):
    """
    Project input onto 5 PCs and measure Mean Squared Reconstruction Error.
    If MSRE > threshold (mean + 3σ of training set), flag as anomaly.

    Returns:
        is_anomaly (bool), z_score (float), message (str)
    """
    X_recon  = pca_detector.inverse_transform(pca_detector.transform(inp_sc))
    msre     = float(np.mean((inp_sc - X_recon) ** 2))
    mean_e   = anomaly_cfg["mean_error"]
    std_e    = anomaly_cfg["std_error"]
    thresh   = anomaly_cfg["threshold"]
    z_score  = (msre - mean_e) / max(std_e, 1e-12)
    is_anom  = msre > thresh

    if is_anom:
        msg = (f"⚠️  OUT-OF-DISTRIBUTION INPUT  (z = {z_score:+.1f}σ)\n"
               f"    One or more parameters fall outside the training data manifold.\n"
               f"    Prediction shown below is an extrapolation — treat with caution.")
    else:
        msg = (f"✅  Valid slag parameters  (z = {z_score:+.2f}σ — well within training range)")
    return is_anom, z_score, msg

# ── SHAP per-sample explanation ────────────────────────────────────────────────
def get_shap_factors(inp_sc: np.ndarray):
    """
    Returns top-3 (feature_name, shap_value) for the given input.
    Uses real-time TreeExplainer if available; falls back to global importance.
    """
    if shap_explainer is not None:
        try:
            sv = shap_explainer.shap_values(inp_sc)   # shape: (1, n_features)
            if isinstance(sv, list):
                sv = sv[0]
            sv_row = sv[0] if sv.ndim == 2 else sv
            ranked = sorted(zip(FEATURES, sv_row.tolist()),
                            key=lambda x: -abs(x[1]))[:3]
            return ranked, "local"   # SHAP values specific to this prediction
        except Exception:
            pass

    # Fallback: global precomputed SHAP importance (always positive, use as unsigned)
    ranked = sorted(global_shap.items(), key=lambda x: -abs(x[1]))[:3]
    return [(f, float(v)) for f, v in ranked], "global"

def format_shap_for_ui(factors, mode):
    """Format SHAP factors into a readable string for display."""
    tag = "Local SHAP (this exact prediction)" if mode == "local" else "Global SHAP (average importance)"
    lines = [f"📊 {tag}:"]
    for feat, val in factors:
        direction = "↑ raises" if val > 0 else "↓ lowers"
        lines.append(f"   • {feat}: {direction} viscosity by {abs(val):.5f} Pa·s")
    return "\n".join(lines)

# ── Rule-based fallback explanation ───────────────────────────────────────────
def rule_explanation(temp, bas, al2o3, mgo, coke, tap, ob, visc, rec,
                     shap_factors=None, is_anomaly=False):
    msgs = []
    if al2o3 > 14:  msgs.append(f"High Al₂O₃ ({al2o3:.1f} wt%) is raising viscosity — alumina acts as a network former.")
    if bas < 1.0:   msgs.append(f"Low basicity ({bas:.2f}) means insufficient CaO to depolymerise the Si–O network.")
    if temp < 1460: msgs.append(f"Temperature ({temp:.0f}°C) is near the lower safe limit — slag may be cooling.")
    if tap > 60:    msgs.append(f"Tap time ({tap:.0f} min) is high — slag has cooled since tap start.")
    if ob < 0.62:   msgs.append(f"Optical basicity Λ={ob:.3f} is low — slag network is highly polymerised.")
    if not msgs:    msgs.append(f"Slag composition is balanced. Optical basicity Λ={ob:.3f} is healthy.")
    if is_anomaly:  msgs.append("⚠️ Note: input parameters are outside the training distribution.")
    actions = {
        "Increase RPM": "⚠️ Recommend increasing disc RPM by ~200–300 to handle viscous slag.",
        "Maintain RPM": "✅ Disc RPM is in the optimal window. No adjustment needed.",
        "Reduce RPM":   "🔵 Slag is too fluid — recommend reducing disc RPM by ~150–200.",
    }
    return " ".join(msgs) + " " + actions.get(rec, "")

# ── LLM Expert Explanation (SHAP-enriched prompt) ─────────────────────────────
def llm_explanation(temp, bas, al2o3, mgo, coke, tap, ob, visc, rec,
                    shap_factors, shap_mode, is_anomaly):
    """
    Calls Qwen2.5-7B-Instruct with a prompt that includes:
      • All slag parameters
      • Predicted viscosity + disc recommendation
      • Top-3 SHAP feature attributions (local or global)
      • Anomaly flag if applicable
    Falls back to rule_explanation() if the API call fails.
    """
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=HF_TOKEN)

        system_prompt = (
            "You are an expert blast furnace metallurgist with 20 years of experience at "
            "Tata Steel. You specialize in Dry Centrifugal Granulation (DCG) slag heat "
            "recovery systems. Your job is to explain ML-predicted slag viscosity results "
            "to plant operators in clear, practical language — citing the specific input "
            "parameters and SHAP-attributed root causes. Be concrete, actionable, and "
            "keep your response under 120 words."
        )

        # Build the SHAP attribution string for the prompt
        shap_tag = "local SHAP (exact attribution for this prediction)" if shap_mode == "local" \
                   else "global SHAP importance (average feature attribution)"
        shap_lines = "; ".join([
            f"{feat} ({'+' if val>=0 else ''}{val:.5f} Pa·s)"
            for feat, val in shap_factors
        ])

        anomaly_note = (
            f" ⚠️ IMPORTANT: This input has a PCA anomaly score of z={abs(shap_factors[0][1]):.1f}σ "
            f"outside the training distribution — the prediction is an extrapolation."
            if is_anomaly else ""
        )

        user_prompt = (
            f"Slag temperature: {temp:.0f}°C. "
            f"CaO/SiO₂ basicity: {bas:.2f}. "
            f"Al₂O₃: {al2o3:.1f} wt%. MgO: {mgo:.1f} wt%. "
            f"Coke rate: {coke:.0f} kg/t iron. Tap time: {tap:.0f} minutes. "
            f"Optical basicity Λ = {ob:.3f}. "
            f"Predicted viscosity: {visc:.4f} Pa·s. "
            f"Disc RPM recommendation: {rec}. "
            f"Top driving factors ({shap_tag}): {shap_lines}.{anomaly_note} "
            f"Explain what is happening physically in the slag and what the operator should do right now."
        )

        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=250,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        # Graceful fallback — always works even without HF token
        return rule_explanation(temp, bas, al2o3, mgo, coke, tap, ob, visc, rec,
                                shap_factors, is_anomaly)

# ── Main prediction function ───────────────────────────────────────────────────
def predict_all(temp, bas, al2o3, mgo, coke, tap, ob_manual):
    # 1. Auto-compute Optical Basicity from composition
    ob = compute_optical_basicity(bas, al2o3, mgo)

    # 2. Scale input
    inp_raw = [[temp, bas, al2o3, mgo, coke, tap, ob]]
    inp_sc  = scaler.transform(inp_raw)

    # 3. PCA anomaly check
    is_anomaly, z_score, anomaly_msg = detect_anomaly(inp_sc)

    # 4. All model predictions
    v_rf  = float(rf_reg.predict(inp_sc)[0])
    v_xgb = float(xgb_reg.predict(inp_sc)[0])
    v_cat = float(cat_reg.predict(inp_sc)[0])
    v_nn  = float(nn_reg.predict(inp_sc, verbose=0).flatten()[0]) if nn_reg else v_cat

    if best_name == "Neural Network" and TF_AVAILABLE:
        v_best = float(tuned_reg.predict(inp_sc, verbose=0).flatten()[0])
    else:
        v_best = float(tuned_reg.predict(inp_sc)[0])

    # 5. RPM recommendation
    rec   = get_rec(v_best)
    emoji = {
        "Increase RPM": "🔴  INCREASE DISC RPM  (+200–300 RPM)",
        "Maintain RPM": "🟢  MAINTAIN DISC RPM  (Optimal window)",
        "Reduce RPM":   "🔵  REDUCE DISC RPM    (−150–200 RPM)",
    }

    # 6. SHAP attribution for this prediction
    shap_factors, shap_mode = get_shap_factors(inp_sc)
    shap_ui_text = format_shap_for_ui(shap_factors, shap_mode)

    # 7. Rule-based explanation (instant)
    rule_exp = rule_explanation(temp, bas, al2o3, mgo, coke, tap, ob,
                                v_best, rec, shap_factors, is_anomaly)

    # 8. Combine anomaly + SHAP for the expert panel
    expert_panel = anomaly_msg + "\n\n" + shap_ui_text + "\n\n" + rule_exp

    return (
        f"{ob:.4f}",                          # auto optical basicity
        anomaly_msg,                           # data quality check
        f"{v_rf:.5f}  Pa·s",
        f"{v_xgb:.5f}  Pa·s",
        f"{v_cat:.5f}  Pa·s",
        f"{v_nn:.5f}  Pa·s",
        f"⭐  {v_best:.5f}  Pa·s   ← Tuned {best_name}",
        emoji.get(rec, rec),
        confidence_bar(v_best),
        expert_panel,
    )

# ── LLM report (called separately by button) ──────────────────────────────────
def generate_llm_report(temp, bas, al2o3, mgo, coke, tap, ob_manual):
    ob     = compute_optical_basicity(bas, al2o3, mgo)
    inp_sc = scaler.transform([[temp, bas, al2o3, mgo, coke, tap, ob]])

    is_anomaly, z_score, _ = detect_anomaly(inp_sc)
    shap_factors, shap_mode = get_shap_factors(inp_sc)

    if best_name == "Neural Network" and TF_AVAILABLE:
        v_best = float(tuned_reg.predict(inp_sc, verbose=0).flatten()[0])
    else:
        v_best = float(tuned_reg.predict(inp_sc)[0])

    rec = get_rec(v_best)
    return llm_explanation(temp, bas, al2o3, mgo, coke, tap, ob,
                           v_best, rec, shap_factors, shap_mode, is_anomaly)

# ── Gradio CSS ─────────────────────────────────────────────────────────────────
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
    report_btn.click(fn=generate_llm_report, inputs=INPUTS, outputs=[llm_out])

    # Live update on every slider change
    for sl in INPUTS:
        sl.change(fn=predict_all, inputs=INPUTS, outputs=OUTPUTS)

demo.launch()
