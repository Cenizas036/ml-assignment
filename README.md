# ArvyaX ML Internship Assignment

## Setup Instructions

```bash
pip install pandas numpy scikit-learn scipy joblib
```

Place `train.csv` and `test.csv` in the same folder, then run:

```bash
python pipeline.py
```

Output: `predictions.csv`

---

## Approach

### Problem Framing
This is NOT a standard classification problem. The data is noisy, short, and sometimes contradictory — mirroring real-world mental health and wellness signals. The system must understand emotional state, decide on an action, and express uncertainty honestly.

### Feature Engineering

**Text features (TF-IDF):**
- Max 300 features, unigrams + bigrams
- `sublinear_tf=True` to handle very short texts gracefully
- Short texts ("ok", "fine") are handled by the `uncertain_flag`

**Metadata features:**
- `sleep_hours`, `energy_level`, `stress_level`, `duration_min`
- `ambience_type`, `time_of_day`, `previous_day_mood`, `face_emotion_hint`, `reflection_quality`
- All label-encoded + StandardScaled
- Missing values: numeric → median, categorical → "unknown"

**Combined:** TF-IDF sparse matrix + metadata dense matrix stacked via `scipy.sparse.hstack`

---

### Model Choice

| Task | Model | Why |
|------|-------|-----|
| Emotional state (Part 1) | RandomForestClassifier | Handles noise, gives feature importance, class_weight="balanced" handles label imbalance |
| Intensity (Part 2) | GradientBoostingClassifier | Captures ordinal structure better than linear models |

**Intensity — Classification vs Regression:**
Treated as **classification** (5 classes: 1–5). Reason: the labels are discrete and perceptual — "3" and "4" are not exactly equidistant the way temperature would be. Classification gives a cleaner probability distribution for uncertainty modeling. A regression approach (RidgeRegressor → round) is a valid alternative and would score comparably.

---

### Decision Engine (Part 3)

Pure rule-based logic using predicted state + intensity + actual stress/energy/time_of_day.

**What to do logic:**
- High stress / anxious → `box_breathing` or `grounding`
- Sad / reflective → `journaling`
- Night time, low energy → `rest` or `sound_therapy`
- Morning + calm → `light_planning`
- High energy + positive → `deep_work`
- Low energy → `yoga` or `rest`
- Default → `pause`

**When to do it logic:**
- Stress ≥ 8 or intensity = 5 → `now`
- Evening/night → `tonight`
- Morning + low intensity → `within_15_min`
- Moderate distress, daytime → `within_15_min`
- Low intensity → `later_today`

---

### Uncertainty Modeling (Part 4)

Three conditions trigger `uncertain_flag = 1`:
1. Combined confidence < 0.50
2. Journal text length < 20 characters (very short input like "ok" or "fine")
3. Top-2 class probability margin < 0.15 (the model is nearly tied between two classes)

`confidence` = average of max predicted probability from emotional_state model + intensity model.

---

### Feature Understanding (Part 5)

Feature importance comes from RandomForest's built-in impurity-based scores.

Expected findings:
- `face_emotion_hint` and `previous_day_mood` are strong metadata predictors
- `stress_level` and `energy_level` anchor intensity predictions
- TF-IDF bigrams like "feel anxious", "really tired", "so calm" carry the most text weight
- `reflection_quality` is a meta-signal: high quality text → more reliable prediction

---

### Ablation Study (Part 6)

Three conditions compared via 3-fold cross-validation F1:
- Text-only
- Metadata-only
- Combined (text + metadata)

Combined almost always wins. Metadata-only can be surprisingly strong because signals like `face_emotion_hint` and `stress_level` are direct proxies for emotional state. Text adds nuance the metadata cannot capture ("I feel okay but deep down I'm exhausted").

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn scipy joblib

# Run full pipeline
python pipeline.py

# Output
# predictions.csv — id, predicted_state, predicted_intensity, confidence, uncertain_flag, what_to_do, when_to_do
```

---

## Files

| File | Purpose |
|------|---------|
| `pipeline.py` | End-to-end pipeline |
| `predictions.csv` | Final output |
| `README.md` | This file |
| `ERROR_ANALYSIS.md` | 10 failure cases with analysis |
| `EDGE_PLAN.md` | On-device deployment strategy |
