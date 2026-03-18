# EDGE_PLAN.md — On-Device / Mobile Deployment

## Overview
This plan describes how the ArvyaX system would run on-device (mobile / offline) with no internet connection.

---

## Why On-Device?

- Mental health data is sensitive — it should never leave the device
- Offline-first means the app works anywhere (forest, mountain, no signal)
- Latency must be <200ms for a responsive UX
- App stores increasingly favor privacy-first architecture

---

## Model Architecture for Edge

### Replace TF-IDF + RandomForest with a Lightweight Pipeline:

| Component | Edge Version | Size | Latency |
|-----------|-------------|------|---------|
| Text encoding | MiniLM-L3 (sentence-transformers) | ~17MB | ~30ms |
| Classifier | XGBoost (ONNX export) | ~1MB | <5ms |
| Decision engine | Pure rule-based Python/Swift | ~0KB | <1ms |
| Total | — | ~20MB | ~35ms |

**Alternative (even smaller):**
- Replace MiniLM-L3 with a custom TF-IDF (vocab=500) → reduce to <5MB total
- Trade: slightly lower accuracy on ambiguous texts

---

## Platform Options

### Android
- Use **TensorFlow Lite** or **ONNX Runtime for Android**
- Export the classifier to `.tflite` or `.onnx`
- Text: TFLite tokenizer or pre-compute TF-IDF in Java/Kotlin
- Decision engine: pure Kotlin logic

### iOS
- Use **CoreML** (Apple's on-device ML framework)
- Export via `coremltools` from scikit-learn / PyTorch
- Text: NSLinguisticTagger for basic preprocessing

### Cross-platform (React Native / Flutter)
- ONNX Runtime has React Native and Flutter bindings
- One model file works on both platforms

---

## Export Pipeline

```python
# Export RandomForest to ONNX
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [("float_input", FloatTensorType([None, X_train_combined.shape[1]]))]
onnx_model = convert_sklearn(clf_state, initial_types=initial_type)
with open("model_state.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Also save TF-IDF vocab and scaler
import joblib
joblib.dump(tfidf,  "tfidf.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(le_state, "label_encoder.joblib")
```

---

## Model Size Optimization

| Technique | Savings | Tradeoff |
|-----------|---------|----------|
| Reduce TF-IDF vocab: 300 → 100 | 60% smaller | Slight accuracy drop |
| Reduce RF trees: 200 → 50 | 75% smaller | Wider confidence intervals |
| Quantize ONNX model (INT8) | 50-75% smaller | <1% accuracy drop |
| Remove least-important features (keep top 20) | 80% smaller | Moderate accuracy drop |

Target: **< 5MB total model bundle** for seamless mobile install.

---

## Latency Budget (Target: <200ms end-to-end)

| Step | Time |
|------|------|
| Text cleaning | ~1ms |
| TF-IDF transform | ~5ms |
| Metadata encoding | ~1ms |
| ONNX inference | <10ms |
| Decision logic | <1ms |
| **Total** | **~18ms** |

This comfortably fits within a 200ms budget even on mid-range phones.

---

## Handling Offline Scenarios

- **No inference needed when offline** — the model runs entirely locally
- **Data sync:** Journal entries stored locally in SQLite. Synced to cloud when online (optional, opt-in)
- **Model updates:** Delta-compressed model updates delivered on WiFi only

---

## Privacy Guarantees

- Journal text **never leaves the device**
- All inference runs in a sandboxed process
- No logging of raw text in production
- Differential privacy can be added for federated learning (future)

---

## Tradeoffs Summary

| Approach | Accuracy | Size | Latency | Privacy |
|----------|----------|------|---------|---------|
| Cloud API (GPT-4) | Very high | 0MB local | 1-3s | Low |
| Server-side our model | High | 0MB local | 200ms | Medium |
| On-device ONNX | Medium-high | ~5MB | ~20ms | Very high |
| On-device TF-IDF only | Medium | ~1MB | ~5ms | Very high |

**Recommendation:** Ship on-device ONNX (5MB). Give users opt-in cloud enhancement for ambiguous/uncertain predictions.