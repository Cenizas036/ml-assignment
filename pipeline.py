"""
ArvyaX ML Internship Assignment - Full Pipeline
Run: python pipeline.py
Outputs: predictions.csv
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import joblib

# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)

train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")
print(f"\nTrain columns: {list(train_df.columns)}")
print(f"\nTarget distribution (emotional_state):\n{train_df['emotional_state'].value_counts()}")
print(f"\nTarget distribution (intensity):\n{train_df['intensity'].value_counts()}")


# ─────────────────────────────────────────────
# STEP 2: PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Feature Engineering")
print("=" * 60)

# --- Text cleaning ---
def clean_text(text):
    """Clean and normalize journal text."""
    if pd.isna(text) or str(text).strip() == "":
        return "no reflection"
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z\s]", " ", text)       # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text)             # collapse whitespace
    return text

# --- Missing value handling ---
def fill_missing(df):
    """Handle missing values robustly."""
    df = df.copy()

    # Numeric columns: fill with median
    numeric_cols = ["sleep_hours", "energy_level", "stress_level", "duration_min"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

    # Categorical columns: fill with "unknown"
    cat_cols = ["ambience_type", "time_of_day", "previous_day_mood",
                "face_emotion_hint", "reflection_quality"]
    for col in cat_cols:
        if col in df.columns:
            df[col].fillna("unknown", inplace=True)

    # Text: fill with placeholder
    df["journal_text"] = df["journal_text"].fillna("no reflection")

    return df

train_df = fill_missing(train_df)
test_df  = fill_missing(test_df)

train_df["clean_text"] = train_df["journal_text"].apply(clean_text)
test_df["clean_text"]  = test_df["journal_text"].apply(clean_text)

# --- Text length feature (very short text = uncertain) ---
train_df["text_length"] = train_df["clean_text"].apply(len)
test_df["text_length"]  = test_df["clean_text"].apply(len)

# --- Encode categorical metadata ---
cat_cols = ["ambience_type", "time_of_day", "previous_day_mood",
            "face_emotion_hint", "reflection_quality"]

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    all_vals = pd.concat([train_df[col], test_df[col]]).astype(str)
    le.fit(all_vals)
    train_df[col + "_enc"] = le.transform(train_df[col].astype(str))
    test_df[col + "_enc"]  = le.transform(test_df[col].astype(str))
    label_encoders[col] = le

# --- Metadata feature matrix ---
meta_cols = [
    "sleep_hours", "energy_level", "stress_level", "duration_min",
    "text_length",
    "ambience_type_enc", "time_of_day_enc", "previous_day_mood_enc",
    "face_emotion_hint_enc", "reflection_quality_enc"
]

# Keep only columns that exist
meta_cols = [c for c in meta_cols if c in train_df.columns]

scaler = StandardScaler()
X_meta_train = scaler.fit_transform(train_df[meta_cols])
X_meta_test  = scaler.transform(test_df[meta_cols])

print(f"Metadata features used: {meta_cols}")


# ─────────────────────────────────────────────
# STEP 3: TF-IDF TEXT FEATURES
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: Text Vectorization (TF-IDF)")
print("=" * 60)

tfidf = TfidfVectorizer(
    max_features=300,
    ngram_range=(1, 2),     # unigrams + bigrams
    min_df=2,
    sublinear_tf=True
)

X_text_train = tfidf.fit_transform(train_df["clean_text"])
X_text_test  = tfidf.transform(test_df["clean_text"])

print(f"TF-IDF feature shape (train): {X_text_train.shape}")

# --- Combined feature matrix (text + metadata) ---
from scipy.sparse import csr_matrix
X_train_combined = hstack([X_text_train, csr_matrix(X_meta_train)])
X_test_combined  = hstack([X_text_test,  csr_matrix(X_meta_test)])

print(f"Combined feature shape (train): {X_train_combined.shape}")


# ─────────────────────────────────────────────
# STEP 4: PART 1 — EMOTIONAL STATE PREDICTION
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Part 1 — Emotional State Prediction")
print("=" * 60)

# Encode target
le_state = LabelEncoder()
y_state = le_state.fit_transform(train_df["emotional_state"])
print(f"Emotional state classes: {list(le_state.classes_)}")

# Model: Random Forest (handles noise well, gives feature importance)
clf_state = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"   # handles imbalanced labels
)
clf_state.fit(X_train_combined, y_state)

# Cross-validation score
cv_scores = cross_val_score(clf_state, X_train_combined, y_state, cv=3, scoring="f1_weighted")
print(f"Emotional state CV F1 (3-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Predict on test
pred_state_encoded = clf_state.predict(X_test_combined)
pred_state = le_state.inverse_transform(pred_state_encoded)

# Prediction probabilities for confidence
pred_state_proba = clf_state.predict_proba(X_test_combined)
state_confidence = pred_state_proba.max(axis=1)   # max probability = confidence


# ─────────────────────────────────────────────
# STEP 5: PART 2 — INTENSITY PREDICTION
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Part 2 — Intensity Prediction")
print("=" * 60)
print("Approach: Treating intensity as CLASSIFICATION (1-5 ordinal classes).")
print("Reason: Distinct categories matter more than exact distance between values.")
print("Alternative: Could use regression (RidgeRegressor) and round to nearest int.")

y_intensity = train_df["intensity"].astype(int)

clf_intensity = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
clf_intensity.fit(X_train_combined, y_intensity)

cv_int = cross_val_score(clf_intensity, X_train_combined, y_intensity, cv=3, scoring="f1_weighted")
print(f"Intensity CV F1 (3-fold): {cv_int.mean():.3f} ± {cv_int.std():.3f}")

pred_intensity = clf_intensity.predict(X_test_combined)
pred_intensity_proba = clf_intensity.predict_proba(X_test_combined)
intensity_confidence = pred_intensity_proba.max(axis=1)


# ─────────────────────────────────────────────
# STEP 6: PART 3 — DECISION ENGINE
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6: Part 3 — Decision Engine (What + When)")
print("=" * 60)

def decide_what(state, intensity, stress, energy, time_of_day):
    """
    Rule-based decision engine.
    Returns: what_to_do (action recommendation)
    Logic:
      - High stress + anxious → breathing or grounding
      - Calm + high energy + daytime → deep work
      - Exhausted / low energy → rest or yoga
      - Night time → sound therapy or rest
      - Sad / reflective → journaling
      - Neutral + morning → light planning
    """
    state = str(state).lower()
    time  = str(time_of_day).lower()

    try:
        stress   = float(stress)
        energy   = float(energy)
        intensity = int(intensity)
    except:
        stress, energy, intensity = 5, 5, 3

    # High stress or anxious states
    if stress >= 7 or state in ["anxious", "stressed", "restless", "overwhelmed"]:
        if intensity >= 4:
            return "box_breathing"
        else:
            return "grounding"

    # Sadness or reflection
    if state in ["sad", "melancholic", "grieving", "low", "reflective"]:
        return "journaling"

    # Night time → rest-based
    if "night" in time or "evening" in time:
        if energy <= 4:
            return "rest"
        else:
            return "sound_therapy"

    # Morning + calm → plan
    if "morning" in time and state in ["calm", "neutral", "content", "hopeful"]:
        return "light_planning"

    # High energy + positive state → deep work
    if energy >= 7 and state in ["calm", "focused", "content", "motivated", "happy"]:
        return "deep_work"

    # Low energy
    if energy <= 3:
        if "morning" in time or "afternoon" in time:
            return "yoga"
        return "rest"

    # Medium energy, positive or neutral
    if state in ["calm", "neutral", "content"]:
        return "movement"

    # Default fallback
    return "pause"


def decide_when(state, intensity, stress, energy, time_of_day):
    """
    Returns: when_to_do
    Options: now, within_15_min, later_today, tonight, tomorrow_morning
    Logic:
      - Urgent high-distress → now
      - Moderate → within_15_min
      - Evening wind-down → tonight
      - Recovery → tomorrow_morning
    """
    state = str(state).lower()
    time  = str(time_of_day).lower()

    try:
        stress    = float(stress)
        energy    = float(energy)
        intensity = int(intensity)
    except:
        stress, energy, intensity = 5, 5, 3

    # Crisis or very high distress
    if stress >= 8 or (intensity >= 5 and state in ["anxious", "stressed", "overwhelmed"]):
        return "now"

    # Night time → tonight for rest activities
    if "night" in time or "evening" in time:
        return "tonight"

    # Morning + light recommendations
    if "morning" in time and intensity <= 2:
        return "within_15_min"

    # Moderate distress during the day
    if intensity >= 3 and ("afternoon" in time or "morning" in time):
        return "within_15_min"

    # Low intensity, daytime
    if intensity <= 2 and ("afternoon" in time or "morning" in time):
        return "later_today"

    # Low energy evening
    if energy <= 3 and ("evening" in time or "night" in time):
        return "tonight"

    # Default for morning planning tasks
    if "morning" in time:
        return "tomorrow_morning"

    return "later_today"


# Apply decision engine to test set
what_list = []
when_list = []

for _, row in test_df.iterrows():
    # Use predicted state + predicted intensity + actual metadata
    idx = list(test_df.index).index(row.name)
    pred_s = pred_state[idx]
    pred_i = pred_intensity[idx]

    what = decide_what(
        state=pred_s,
        intensity=pred_i,
        stress=row.get("stress_level", 5),
        energy=row.get("energy_level", 5),
        time_of_day=row.get("time_of_day", "morning")
    )
    when = decide_when(
        state=pred_s,
        intensity=pred_i,
        stress=row.get("stress_level", 5),
        energy=row.get("energy_level", 5),
        time_of_day=row.get("time_of_day", "morning")
    )
    what_list.append(what)
    when_list.append(when)

print(f"Sample decisions:")
for i in range(min(5, len(what_list))):
    print(f"  Row {i}: state={pred_state[i]}, intensity={pred_intensity[i]} → {what_list[i]} ({when_list[i]})")


# ─────────────────────────────────────────────
# STEP 7: PART 4 — UNCERTAINTY MODELING
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7: Part 4 — Uncertainty Modeling")
print("=" * 60)

# Combined confidence = average of state and intensity confidence
combined_confidence = (state_confidence + intensity_confidence) / 2
combined_confidence = np.round(combined_confidence, 3)

# Uncertain flag:
# 1 if confidence < 0.5 OR text is very short (< 20 chars) OR top-2 probs are close
def compute_uncertain_flag(state_proba, intensity_proba, text_length):
    flags = []
    for i in range(len(state_proba)):
        conf     = (state_proba[i].max() + intensity_proba[i].max()) / 2
        txt_len  = text_length[i]

        # Sort probabilities descending
        top2_state = np.sort(state_proba[i])[::-1][:2]
        margin     = top2_state[0] - top2_state[1] if len(top2_state) > 1 else 1.0

        is_uncertain = (
            conf < 0.50 or          # low overall confidence
            txt_len < 20 or         # very short text ("ok", "fine")
            margin < 0.15           # top two classes are close (ambiguous)
        )
        flags.append(int(is_uncertain))
    return flags

text_lengths = test_df["text_length"].values
uncertain_flags = compute_uncertain_flag(pred_state_proba, pred_intensity_proba, text_lengths)

uncertain_count = sum(uncertain_flags)
print(f"Uncertain predictions: {uncertain_count}/{len(uncertain_flags)} ({100*uncertain_count/len(uncertain_flags):.1f}%)")
print("Uncertainty triggered by: low confidence, short text (<20 chars), or ambiguous top-2 class margin")


# ─────────────────────────────────────────────
# STEP 8: ASSEMBLE predictions.csv
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 8: Saving predictions.csv")
print("=" * 60)

predictions_df = pd.DataFrame({
    "id":                  test_df["id"],
    "predicted_state":     pred_state,
    "predicted_intensity": pred_intensity,
    "confidence":          np.round(combined_confidence, 3),
    "uncertain_flag":      uncertain_flags,
    "what_to_do":          what_list,
    "when_to_do":          when_list
})

predictions_df.to_csv("predictions.csv", index=False)
print("Saved: predictions.csv")
print(predictions_df.head(10).to_string())


# ─────────────────────────────────────────────
# STEP 9: PART 5 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 9: Part 5 — Feature Importance")
print("=" * 60)

# Feature names
tfidf_names    = tfidf.get_feature_names_out().tolist()
all_feat_names = tfidf_names + meta_cols

feature_importances = clf_state.feature_importances_

# Top 10 features
top_idx = np.argsort(feature_importances)[::-1][:10]
print("Top 10 most important features for emotional_state prediction:")
for rank, idx in enumerate(top_idx, 1):
    fname = all_feat_names[idx] if idx < len(all_feat_names) else f"feat_{idx}"
    print(f"  {rank:2d}. {fname:<35s} importance={feature_importances[idx]:.4f}")

# Text vs metadata importance
text_importance = feature_importances[:len(tfidf_names)].sum()
meta_importance = feature_importances[len(tfidf_names):].sum()
print(f"\nText features total importance:     {text_importance:.3f}")
print(f"Metadata features total importance: {meta_importance:.3f}")


# ─────────────────────────────────────────────
# STEP 10: PART 6 — ABLATION STUDY
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 10: Part 6 — Ablation Study")
print("=" * 60)

# Text-only model
clf_text_only = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1
)
scores_text_only = cross_val_score(clf_text_only, X_text_train, y_state, cv=3, scoring="f1_weighted")

# Metadata-only model
clf_meta_only = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1
)
scores_meta_only = cross_val_score(clf_meta_only, X_meta_train, y_state, cv=3, scoring="f1_weighted")

# Combined model (already computed above)
print(f"  Text-only     F1: {scores_text_only.mean():.3f} ± {scores_text_only.std():.3f}")
print(f"  Metadata-only F1: {scores_meta_only.mean():.3f} ± {scores_meta_only.std():.3f}")
print(f"  Combined      F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print("\nConclusion: Combined model is expected to outperform either alone.")
print("Text captures nuance; metadata provides grounding context.")


# ─────────────────────────────────────────────
# STEP 11: PART 7 — ERROR ANALYSIS (top failure cases on train)
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 11: Part 7 — Error Analysis (Training Data Failures)")
print("=" * 60)

# Get predictions on training set to find failure cases
train_pred_state_enc = clf_state.predict(X_train_combined)
train_pred_state     = le_state.inverse_transform(train_pred_state_enc)
train_true_state     = train_df["emotional_state"].values

# Find misclassified rows
errors_mask = train_pred_state != train_true_state
errors_df   = train_df[errors_mask].copy()
errors_df["predicted_state"] = train_pred_state[errors_mask]

print(f"Training errors: {errors_mask.sum()}/{len(train_df)} ({100*errors_mask.mean():.1f}%)")
print("\nSample of 10 failure cases:")
cols_to_show = ["id", "journal_text", "emotional_state", "predicted_state",
                "stress_level", "energy_level", "text_length"]
cols_to_show = [c for c in cols_to_show if c in errors_df.columns]

for i, (_, row) in enumerate(errors_df.head(10).iterrows()):
    print(f"\n  Case {i+1}:")
    print(f"    Text:      \"{str(row.get('journal_text',''))[:80]}...\"")
    print(f"    True:      {row.get('emotional_state','?')}")
    print(f"    Predicted: {row.get('predicted_state','?')}")
    print(f"    Stress: {row.get('stress_level','?')} | Energy: {row.get('energy_level','?')}")


# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("Output file: predictions.csv")
print("=" * 60)