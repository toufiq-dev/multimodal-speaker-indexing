# Failure Analysis — First End-to-End Run (Jamuna TV "রাজনীতি", 53.2 min, Kaggle T4)

This document records the measured failure of the first full-show run so the
thesis can report negative results honestly and so regressions are detectable.

## Measured output pathology (`result.json`, 191 segments)

| Metric | Value | Healthy expectation |
|---|---|---|
| Unique texts | **34 of 191 cues** (82% duplicates) | ~191 unique |
| Worst duplication | one block repeated **×32** consecutively, spanning 310 s→569 s | 0 duplicates |
| Avg / max cue length | **2,836 / 6,260 chars** | 30–80 chars per cue |
| Punctuation | **0 danda, 0 commas, 0 question marks** in 541,756 chars | present |
| Confidence | exactly two values: 165 × 0.5, 26 × 0.6 | distributed, calibrated |
| Speakers | Speaker_1..4 + `face_cluster_0`; zero real names resolved | 5 named speakers |
| Coverage | 95% timeline; min cue 0.02 s | no sub-frame cues |

## Causal chain (verified against notebook execution log)

1. **PEFT 0.12 rejects the adapter config** (`alora_invocation_tokens`
   TypeError) → adapter JSON hand-edited inside the notebook.
2. **transformers 4.44 Whisper IndexError** on empty `suppress_tokens` →
   three successive rewrites of `asr_lora.py` inside `/kaggle/working`,
   introducing double-escaped regexes (`r"<\\|[^|]+\\|>"`) that *mangled*
   special tokens instead of removing them.
3. The hot-patched file became the repo's code — the notebook was the
   codebase; nothing reproducible remained.
4. **LoRA chunk engine violated the word-timestamp contract**: fixed 20 s
   windows, zero overlap, no VAD, `words=[]` always → greedy same-speaker
   merging produced 80–140 s mega-segments.
5. **Fusion join amplified the violation**: every short diarization turn
   overlapping a mega-segment received that segment's ENTIRE text, once per
   overlap → 34 unique texts fanned out to 191 cues (the ×32 run above).
6. **Identity resolution failed end-to-end**:
   - face registry directory existed but contained zero photos → all faces
     UNKNOWN → DBSCAN → single `face_cluster_0`;
   - NER names matched to speakers by LIST POSITION against speakers ordered
     by an untrained gating network's constant output → wrong by construction;
   - host self-intro ("আমি রফতান আঞ্জুমান নিকোল") was never used as an anchor.
7. **Text quality**: unpunctuated training data + `_clean_text` DELETING
   repeated-character runs instead of collapsing them ate legitimate Bangla
   graphemes.

## Fix mapping

| Failure | Fix | Regression test |
|---|---|---|
| Duplicated cues | finals built from turns: word-midpoint containment or proportional split | `tests/test_fusion_segments.py` |
| Arbitrary word↔speaker assignment | midpoint containment replaces IoU | `tests/test_word_assignment.py` |
| Mangled special tokens / eaten graphemes | corrected regexes; collapse runs to one instance | `tests/test_text_cleaning.py` |
| Constant 0.5 confidence | untrained gate removed; deterministic cascade | `tests/test_identity_resolution.py` |
| Positional name matching | greedy co-occurrence matching; host anchor priority | `tests/test_identity_resolution.py` |
| Empty registry undetected | `validate_registry()` pre-flight check in `evaluation/dataset.py` | `tests/test_metrics_and_dataset.py` |
| Notebook-as-codebase | experiment tracking (`evaluation/tracking.py`); slim clone-and-run notebook plan | manual process |
| No evaluation | WER/CER/cpWER/WDER/DER/JER/face accuracy/health assertions | whole suite |

## Process lessons

- Cache stage artifacts (diarization RTTM, raw ASR words) before fusion;
  fusion bugs then cost seconds, not a 30-minute re-transcription.
- Debug with a 3–5 minute slice before committing to the full episode.
- Pin the stack (transformers ≥ 4.46, peft ≥ 0.13) — both crash loops were
  version friction, not logic errors.
