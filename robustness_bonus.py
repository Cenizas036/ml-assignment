"""
ArvyaX — Part 9: Robustness + Bonus: Supportive Message Generator
Run AFTER pipeline.py has generated predictions.csv

Usage:
    python robustness_bonus.py

Output:
    predictions_with_messages.csv
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PART 9 — ROBUSTNESS HANDLER
# ─────────────────────────────────────────────

class RobustnessHandler:
    """
    Handles three robustness challenges:
    1. Very short text ("ok", "fine", "idk")
    2. Missing values in metadata
    3. Contradictory inputs (e.g. happy text + high stress)
    """

    # Words that are too short/vague to trust
    VAGUE_WORDS = {"ok", "fine", "idk", "okay", "good", "bad", "meh",
                   "yeah", "yep", "nope", "nothing", "dunno", "no", "yes",
                   "alright", "whatever", "hmm", "hm", "eh"}

    def __init__(self):
        pass

    # ── 1. Very Short Text ──────────────────────────────

    def is_short_text(self, text: str) -> bool:
        """Returns True if text is too short to trust."""
        if pd.isna(text):
            return True
        text = str(text).strip().lower()
        words = text.split()
        if len(text) < 15:
            return True
        if len(words) <= 2 and all(w in self.VAGUE_WORDS for w in words):
            return True
        return False

    def handle_short_text(self, text: str) -> dict:
        """
        When text is too short:
        - Fall back to metadata-only prediction
        - Set uncertain_flag = 1
        - Reduce confidence by 40%
        """
        return {
            "action": "fallback_to_metadata",
            "uncertain_flag": 1,
            "confidence_penalty": 0.40,
            "note": f"Short/vague text detected: '{text}' → relying on metadata signals"
        }

    # ── 2. Missing Values ───────────────────────────────

    def handle_missing_values(self, row: dict) -> dict:
        """
        Imputation strategy per column:
        - sleep_hours: 6.5 (population average)
        - energy_level: 5 (neutral midpoint)
        - stress_level: 5 (neutral midpoint)
        - time_of_day: 'morning' (most common)
        - previous_day_mood: 'neutral'
        - face_emotion_hint: 'unknown'
        - ambience_type: 'unknown'
        - reflection_quality: 'medium'
        """
        defaults = {
            "sleep_hours":       6.5,
            "energy_level":      5,
            "stress_level":      5,
            "duration_min":      20,
            "time_of_day":       "morning",
            "previous_day_mood": "neutral",
            "face_emotion_hint": "unknown",
            "ambience_type":     "unknown",
            "reflection_quality":"medium",
        }

        filled = {}
        missing_cols = []

        for col, default in defaults.items():
            val = row.get(col, None)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                filled[col] = default
                missing_cols.append(col)
            else:
                filled[col] = val

        return {
            "filled_row": filled,
            "missing_cols": missing_cols,
            "confidence_penalty": 0.05 * len(missing_cols),  # -5% per missing field
            "note": f"Imputed {len(missing_cols)} missing fields: {missing_cols}"
        }

    # ── 3. Contradictory Inputs ─────────────────────────

    def detect_contradiction(self, text: str, stress: float, energy: float,
                              previous_mood: str, face_hint: str) -> dict:
        """
        Detects when text sentiment conflicts with metadata signals.

        Contradiction types:
        A) Positive text + high stress metadata → hidden anxiety
        B) Negative text + low stress + high energy → venting, not crisis
        C) Calm face hint + high stress score → suppression
        D) Previous mood: very sad + text: very happy → overcorrection
        """
        text_lower = str(text).lower()

        positive_words = {"great", "amazing", "happy", "wonderful", "calm",
                          "peaceful", "good", "fine", "lovely", "nice", "joy"}
        negative_words = {"tired", "exhausted", "anxious", "stressed", "sad",
                          "heavy", "dull", "empty", "lost", "overwhelmed", "off"}

        text_positive = sum(1 for w in positive_words if w in text_lower)
        text_negative = sum(1 for w in negative_words if w in text_lower)

        try:
            stress = float(stress)
            energy = float(energy)
        except:
            stress, energy = 5.0, 5.0

        contradictions = []

        # Type A: text seems positive but stress is very high
        if text_positive > text_negative and stress >= 7:
            contradictions.append("positive_text_high_stress")

        # Type B: text seems negative but stress is low
        if text_negative > text_positive and stress <= 3:
            contradictions.append("negative_text_low_stress")

        # Type C: face shows calm but stress is high
        if str(face_hint).lower() in ["calm_face", "neutral_face"] and stress >= 8:
            contradictions.append("calm_face_high_stress")

        # Type D: previous mood very sad but text is very positive
        if str(previous_mood).lower() in ["sad", "depressed", "overwhelmed"] \
                and text_positive >= 3:
            contradictions.append("mood_swing_overcorrection")

        return {
            "has_contradiction": len(contradictions) > 0,
            "contradiction_types": contradictions,
            "confidence_penalty": 0.15 * len(contradictions),  # -15% per contradiction
            "recommended_action": "blend_predictions" if contradictions else "use_model_output",
            "note": f"Contradictions found: {contradictions}" if contradictions else "No contradictions"
        }

    # ── Combined Robustness Check ───────────────────────

    def full_check(self, row: dict) -> dict:
        """Run all three checks and return combined penalty + flags."""
        text = row.get("journal_text", "")

        short_result    = self.handle_short_text(text)
        missing_result  = self.handle_missing_values(row)
        conflict_result = self.detect_contradiction(
            text=text,
            stress=row.get("stress_level", 5),
            energy=row.get("energy_level", 5),
            previous_mood=row.get("previous_day_mood", "neutral"),
            face_hint=row.get("face_emotion_hint", "unknown")
        )

        is_short = self.is_short_text(text)
        total_penalty = (
            (short_result["confidence_penalty"] if is_short else 0) +
            missing_result["confidence_penalty"] +
            conflict_result["confidence_penalty"]
        )

        return {
            "is_short_text":     is_short,
            "missing_cols":      missing_result["missing_cols"],
            "has_contradiction": conflict_result["has_contradiction"],
            "contradiction_types": conflict_result["contradiction_types"],
            "total_penalty":     min(total_penalty, 0.60),  # cap penalty at 60%
        }


# ─────────────────────────────────────────────
# BONUS — SUPPORTIVE MESSAGE GENERATOR
# ─────────────────────────────────────────────

class SupportiveMessageGenerator:
    """
    Generates human-like supportive messages based on:
    - predicted_state
    - predicted_intensity
    - what_to_do
    - when_to_do
    - contextual signals

    No LLM used — pure template + rule logic.
    """

    # Opening lines based on emotional state
    OPENINGS = {
        "anxious":      [
            "It sounds like your mind is running a little fast right now.",
            "You seem to be carrying some tension at the moment.",
            "There's a restless energy in your reflection today.",
        ],
        "stressed":     [
            "It feels like there's a lot on your plate right now.",
            "You've got a heavy load today — that's okay.",
            "Your reflection hints at some real pressure building up.",
        ],
        "sad":          [
            "It sounds like things feel a bit heavy right now.",
            "There's a quiet sadness in your words today.",
            "Sometimes things just feel low — and that's valid.",
        ],
        "calm":         [
            "You seem to be in a grounded, settled place.",
            "There's a gentle calm in what you've shared.",
            "Your reflection suggests you're in a good space right now.",
        ],
        "happy":        [
            "There's a lovely energy in your reflection today!",
            "You seem to be in a really positive place.",
            "It sounds like things are going well for you right now.",
        ],
        "neutral":      [
            "You seem to be in a steady, balanced state.",
            "Nothing too high or low — just present.",
            "Your reflection feels grounded and even.",
        ],
        "restless":     [
            "You seem a little unsettled — your mind wants to move.",
            "There's a fidgety energy in your words today.",
            "It sounds like you're between states right now.",
        ],
        "melancholic":  [
            "There's a thoughtful, bittersweet tone in your reflection.",
            "You seem to be sitting with some deeper feelings today.",
            "Your words carry a quiet weight — that's okay to feel.",
        ],
        "focused":      [
            "You seem sharp and ready today.",
            "There's a clear, directed energy in your reflection.",
            "Your mind appears tuned in and engaged.",
        ],
        "overwhelmed":  [
            "It sounds like everything is hitting at once right now.",
            "You're carrying a lot — more than feels manageable.",
            "Your reflection suggests you're stretched quite thin.",
        ],
    }

    # Action descriptions
    ACTION_DESCRIPTIONS = {
        "box_breathing":   "Try a box breathing exercise — 4 counts in, hold 4, out 4, hold 4. Repeat 4 times.",
        "journaling":      "Take 10 minutes to write freely — no structure, no goals. Just let it out on paper.",
        "grounding":       "Try the 5-4-3-2-1 grounding technique: name 5 things you see, 4 you hear, 3 you can touch.",
        "deep_work":       "Your mind is in a good place for focused work. Block 90 minutes, close distractions, and go deep.",
        "yoga":            "A gentle yoga flow — even 15 minutes — will help move stale energy and restore your body.",
        "sound_therapy":   "Put on calming binaural beats or nature sounds and simply rest with your eyes closed for 20 minutes.",
        "light_planning":  "Take 10 minutes to gently plan your day. No pressure — just a loose sketch of intentions.",
        "rest":            "Your body is asking for rest. Give yourself permission to do nothing for a while.",
        "movement":        "A short walk or gentle movement will shift your energy more than you'd expect.",
        "pause":           "Just pause. Step away from screens, take three slow breaths, and let yourself arrive in the moment.",
    }

    # Timing phrases
    TIMING_PHRASES = {
        "now":              "Do this right now, before moving on to anything else.",
        "within_15_min":    "Try to do this within the next 15 minutes while it's fresh.",
        "later_today":      "Set a reminder for later today — don't let it slip.",
        "tonight":          "Save this for tonight, as part of your wind-down.",
        "tomorrow_morning": "Start tomorrow morning with this — it'll set the right tone for the day.",
    }

    # Intensity qualifiers
    INTENSITY_QUALIFIERS = {
        1: "just a slight",
        2: "a mild",
        3: "a moderate",
        4: "a fairly strong",
        5: "an intense",
    }

    def generate(self, predicted_state: str, predicted_intensity: int,
                  what_to_do: str, when_to_do: str,
                  uncertain_flag: int = 0) -> str:
        """Generate a complete supportive message."""

        state   = str(predicted_state).lower().strip()
        action  = str(what_to_do).lower().strip()
        timing  = str(when_to_do).lower().strip()

        try:
            intensity = int(predicted_intensity)
            intensity = max(1, min(5, intensity))
        except:
            intensity = 3

        # Pick opening line
        openings = self.OPENINGS.get(state, [
            "Your reflection today carries something worth paying attention to."
        ])
        # Use intensity as index selector (deterministic, no randomness needed)
        opening = openings[intensity % len(openings)]

        # Intensity qualifier
        qualifier = self.INTENSITY_QUALIFIERS.get(intensity, "a moderate")

        # Action description
        action_desc = self.ACTION_DESCRIPTIONS.get(
            action,
            f"Take a moment for some {action.replace('_', ' ')}."
        )

        # Timing phrase
        timing_phrase = self.TIMING_PHRASES.get(
            timing,
            "Try to do this when you get a moment."
        )

        # Uncertainty disclaimer
        uncertain_note = ""
        if uncertain_flag == 1:
            uncertain_note = " (I'm not fully certain about this read — trust your instincts.)"

        # Assemble message
        message = (
            f"{opening} "
            f"There seems to be {qualifier} {state} energy here.{uncertain_note} "
            f"Let's work with that. "
            f"{action_desc} "
            f"{timing_phrase}"
        )

        return message


# ─────────────────────────────────────────────
# APPLY TO predictions.csv
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Part 9: Robustness + Bonus: Supportive Messages")
    print("=" * 60)

    # Load base predictions
    try:
        preds = pd.read_csv("predictions.csv")
    except FileNotFoundError:
        print("ERROR: predictions.csv not found. Run pipeline.py first.")
        exit(1)

    # Load test data for metadata
    try:
        test_df = pd.read_csv("test.csv")
    except FileNotFoundError:
        print("ERROR: test.csv not found.")
        exit(1)

    # Merge
    merged = preds.merge(test_df, on="id", how="left")

    handler   = RobustnessHandler()
    generator = SupportiveMessageGenerator()

    robustness_notes   = []
    supportive_messages = []
    adjusted_confidence = []
    adjusted_uncertain  = []

    for _, row in merged.iterrows():
        row_dict = row.to_dict()

        # Run robustness check
        check = handler.full_check(row_dict)

        # Adjust confidence
        orig_conf = float(row.get("confidence", 0.5))
        new_conf  = max(0.05, orig_conf - check["total_penalty"])
        adjusted_confidence.append(round(new_conf, 3))

        # Adjust uncertain flag
        new_flag = 1 if (row.get("uncertain_flag", 0) == 1 or
                         check["is_short_text"] or
                         check["has_contradiction"]) else 0
        adjusted_uncertain.append(new_flag)

        # Build robustness note
        notes = []
        if check["is_short_text"]:
            notes.append("short_text")
        if check["missing_cols"]:
            notes.append(f"missing:{','.join(check['missing_cols'][:2])}")
        if check["has_contradiction"]:
            notes.append(f"conflict:{check['contradiction_types'][0]}")
        robustness_notes.append("|".join(notes) if notes else "ok")

        # Generate supportive message
        msg = generator.generate(
            predicted_state=row.get("predicted_state", "neutral"),
            predicted_intensity=row.get("predicted_intensity", 3),
            what_to_do=row.get("what_to_do", "pause"),
            when_to_do=row.get("when_to_do", "later_today"),
            uncertain_flag=new_flag
        )
        supportive_messages.append(msg)

    # Build final output
    output_df = preds.copy()
    output_df["confidence"]        = adjusted_confidence
    output_df["uncertain_flag"]    = adjusted_uncertain
    output_df["robustness_note"]   = robustness_notes
    output_df["supportive_message"] = supportive_messages

    output_df.to_csv("predictions_with_messages.csv", index=False)

    print(f"\nSaved: predictions_with_messages.csv")
    print(f"Rows: {len(output_df)}")
    print(f"\nSample outputs:\n")

    for i, row in output_df.head(5).iterrows():
        print(f"ID {row['id']} | {row['predicted_state']} (intensity {row['predicted_intensity']}) "
              f"| conf={row['confidence']} | uncertain={row['uncertain_flag']}")
        print(f"  → Do: {row['what_to_do']} | When: {row['when_to_do']}")
        print(f"  → Note: {row['robustness_note']}")
        print(f"  → Message: {row['supportive_message']}")
        print()

    # Summary stats
    short_count    = sum(1 for n in robustness_notes if "short_text" in n)
    conflict_count = sum(1 for n in robustness_notes if "conflict" in n)
    missing_count  = sum(1 for n in robustness_notes if "missing" in n)
    uncertain_count = sum(adjusted_uncertain)

    print("=" * 60)
    print("Robustness Summary:")
    print(f"  Short/vague text detected:    {short_count}")
    print(f"  Contradictory signals:        {conflict_count}")
    print(f"  Missing value imputation:     {missing_count}")
    print(f"  Total uncertain predictions:  {uncertain_count}/{len(output_df)}")
    print("=" * 60)