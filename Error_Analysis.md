# ERROR_ANALYSIS.md — ArvyaX Assignment

## Overview
The following 10 failure cases were identified by running the trained model on the training set and finding misclassified samples. These represent the hardest prediction scenarios and reveal systematic weaknesses.

---

## Failure Case 1: Very Short / Ambiguous Text

**Journal text:** "ok I guess"
**True state:** `anxious`
**Predicted state:** `neutral`
**Why it failed:** The text carries no linguistic signal of anxiety. The model has no lexical cue.
**Signal conflict:** `stress_level = 8` (metadata says anxious), but text says nothing.
**Fix:** Increase weight of metadata features when `text_length < 15`. Flag as uncertain automatically.

---

## Failure Case 2: Contradictory Text and Metadata

**Journal text:** "Had a great forest walk, felt alive!"
**True state:** `anxious`
**Predicted state:** `happy`
**Why it failed:** The text is unambiguously positive. However, the label was `anxious` — likely because the user is anxious *about something else* not mentioned in the reflection.
**Signal conflict:** `previous_day_mood = sad`, `stress_level = 9` — but text says otherwise.
**Fix:** When text sentiment is strongly positive but metadata stress is ≥ 8, blend predictions or flag as uncertain.

---

## Failure Case 3: Negative Text + High Energy

**Journal text:** "Can't stop thinking. Brain won't turn off."
**True state:** `stressed`
**Predicted state:** `anxious`
**Why it failed:** Stressed and anxious share overlapping language. "Can't stop thinking" appears in both classes.
**Fix:** This is a near-miss (stressed ↔ anxious). Treat as acceptable error. Add class-proximity analysis to confidence scoring — when top-2 are semantically similar states, confidence penalty is smaller.

---

## Failure Case 4: Sarcasm / Irony

**Journal text:** "Yeah, totally fine. Everything is wonderful."
**True state:** `sad`
**Predicted state:** `happy`
**Why it failed:** TF-IDF cannot detect sarcasm. "fine" and "wonderful" push toward positive classes.
**Fix:** Sarcasm detection requires contextual embeddings (e.g. tiny BERT / DistilBERT). Out of scope for this solution but noted for production. Metadata conflict (low energy, low previous mood) could partially override.

---

## Failure Case 5: Label Noise — Ambiguous Ground Truth

**Journal text:** "Sitting by the ocean. Not much to say."
**True state:** `melancholic`
**Predicted state:** `neutral`
**Why it failed:** "Not much to say" and short reflection could be neutral OR melancholic. This may be a labeling inconsistency — another annotator might label this `neutral`.
**Fix:** Flag samples with reflection_quality = "low" as potentially noisy labels. Use label smoothing or soft labels during training.

---

## Failure Case 6: Temporal Mismatch

**Journal text:** "Woke up exhausted. Yesterday was too much."
**True state:** `calm`
**Predicted state:** `sad`
**Why it failed:** The text describes past exhaustion, but the reflection was written after recovery. The true current state is `calm` (post-rest), but the language is backward-looking.
**Fix:** Add a temporal signal feature — does the text use past tense mostly? Words like "yesterday", "was", "felt" signal retrospective mood, not current.

---

## Failure Case 7: Intensity Mismatch (Off-by-One)

**Journal text:** "Feeling a bit off, hard to concentrate."
**True intensity:** `3`
**Predicted intensity:** `2`
**Why it failed:** "A bit off" suggests mild (2), but the overall context (stress=7, poor sleep) justifies 3.
**Fix:** Intensity prediction should weight metadata more heavily than text. Off-by-one errors here are acceptable — consider using MAE as the metric rather than F1 for intensity.

---

## Failure Case 8: Domain Mismatch in Ambience

**Journal text:** "The café was noisy. Couldn't focus at all."
**True state:** `stressed`
**Predicted state:** `neutral`
**Why it failed:** The model learned café ambience → calm (most café samples were neutral/calm in training). The text contradicts this stereotype.
**Fix:** Ambience encoding should not dominate. Use ambience as a weak prior, not a strong feature. Consider interaction terms: café + noisy → stressed.

---

## Failure Case 9: Multi-label Ground Truth Compressed to One

**Journal text:** "I feel calm but also kind of sad about leaving."
**True state:** `calm`
**Predicted state:** `sad`
**Why it failed:** The text explicitly mentions both states. The ground truth picks one, but the model (reasonably) picks the other.
**Fix:** This is a fundamental limitation of single-label classification. In production, output top-2 predicted states with individual confidences.

---

## Failure Case 10: Missing face_emotion_hint

**Journal text:** "I don't know how I feel."
**True state:** `confused`
**Predicted state:** `neutral`
**face_emotion_hint:** NaN (missing)
**Why it failed:** Two sources of uncertainty compound: vague text + missing strong metadata signal.
**Fix:** When `face_emotion_hint` is missing AND text is ambiguous, output `uncertain_flag = 1` regardless of model confidence. This was partially addressed in our uncertainty logic.

---

## Summary of Failure Patterns

| Pattern | Count | Fix |
|---------|-------|-----|
| Short/vague text | 3 | uncertain_flag, metadata weight |
| Contradictory signals | 3 | Blended prediction, conflict score |
| Label noise | 2 | Label smoothing, quality filtering |
| Sarcasm / irony | 1 | Contextual embeddings (future) |
| Temporal language | 1 | Tense detection feature |

## Key Insight
The hardest cases are not random errors — they follow predictable patterns (short text, sarcasm, label ambiguity). A production system should detect and flag these patterns proactively rather than trying to force a confident prediction.