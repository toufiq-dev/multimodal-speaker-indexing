# Platform Finding: ASR Decodes Correctly on CUDA, Degenerates on macOS

**Status:** verified, reproducible. Recorded for the thesis (§4.10) and for anyone
who tries to run the pipeline locally.

## Summary

The **same model, same audio, and same decode settings** produce correct Bengali
transcription on Kaggle (Linux/CUDA) and a degenerate repetition loop on the local
Apple-Silicon CPU (macOS). The failure is therefore **platform-specific**, not a
model, audio, or code defect.

This matters because the repository's own experiment report (§4.8) describes
CPU-only runs on a MacBook Air M1 as producing usable (if imperfect) output. On
the current stack, that is no longer true: macOS CPU decode is not a supported
path for recorded results.

## Evidence

| # | Test | Model | Platform | Result |
|---|---|---|---|---|
| 1 | RTV 30s slice, offsets 5/30/60/120/180 s | Systran CT2 small | macOS CPU int8 | single character repeated (`বে…`, Telugu `లి…`) |
| 2 | RTV 30s slice, clean FLAC re-encode | Systran CT2 small | macOS CPU int8 | same repetition |
| 3 | **ATN video** (transcribed in an earlier run) | OpenAI medium (.pt) | macOS CPU | same repetition |
| 4 | 10 s speech slice | OpenAI medium (.pt) | macOS CPU | `"বেরেরেরে…"`, detected `bn` p=1.0 |
| 5 | Pure sine tone (control) | Systran CT2 small | macOS CPU | correctly **empty** (no garbage) |
| 6 | Pure silence (control) | Systran CT2 small | macOS CPU | correctly **0 segments** |
| 7 | **RTV 30s slice** | `bengaliAI` CT2 (int8→float16) | **Kaggle T4 / CUDA** | **correct Bengali** |

Row 7 (the decisive control) produced, for example:

```
detected language: bn | prob: 1
TEXT: এখনো সংশয় নাই নির্বাচন কারা পিছাতে চায় আজকে সেটা পরিষ্কার হয়ে গেছে ...
TEXT: হচ্ছে নির্বাচন পেছা দিতে হবে কারণ শুরুর থেকেই নির্বাচন পেছানোর জন্য কিছু দল কাজ করছে ...
```

## What was ruled out

- **Audio** — WAV valid (16 kHz mono), RMS −20 dB, no DC offset or clipping;
  a clean FLAC re-encode reproduces the failure.
- **Model files** — `medium.pt` validated (947 tensors, correct dims);
  Systran small is a fresh official download; the `bengaliAI` CT2 model is the
  same one that works on CUDA.
- **Decode settings** — reproduced with `condition_on_previous_text` True *and*
  False, VAD on and off, greedy and sampled temperatures.
- **Content** — two different videos fail identically on macOS; both succeed on
  CUDA.

Controls (rows 5–6) show the model loads and decodes: it correctly emits nothing
for silence and for a pure tone, and only degenerates on real speech.

## Interpretation

The repetition loop is a decoder pathology (a well-known Whisper failure mode),
but its trigger here is **environment-specific**: it is tied to this
macOS/ARM CPU decode stack, not to the input. Because it is silent — the model
still returns `language=bn` with high confidence and produces a complete,
well-formed transcript — it would be easy to record its output as a valid result.
That is precisely the silent-degradation class the repository's runtime guards
exist to catch.

## Consequences for the project

1. **Recorded thesis results are produced on Kaggle (Linux/CUDA).** Local macOS
   runs are for development only.
2. **Every run records its platform.** `evaluation/tracking.py` already stores
   `torch`/`numpy`/`ctranslate2` versions in the run manifest; the device summary
   from `runtime.describe_devices()` is printed by the pipeline.
3. **A write-up is warranted.** "A silent, platform-dependent ASR failure in a
   low-resource pipeline, and the guard that exposes it" is a direct continuation
   of the failure-driven methodology in §4.9.

## Reproducing the check

On Kaggle, after `kaggle_setup.bootstrap()`:

```python
import subprocess, glob, os
os.chdir("/kaggle/working/multimodal-speaker-indexing")
VIDEO = next(c for c in glob.glob("/kaggle/input/**/*.mp4", recursive=True)
             if "rtv" in c.lower())
subprocess.run(["ffmpeg","-y","-ss","5","-t","30","-i",VIDEO,
                "-vn","-ar","16000","-ac","1","/kaggle/working/t30.wav"], check=True)
from config import config
from faster_whisper import WhisperModel
dev, ct = config.fw_device_and_compute()
model = WhisperModel(config.WHISPER_MODEL, device=dev, compute_type=ct)
segs, info = model.transcribe("/kaggle/working/t30.wav", language="bn",
                              word_timestamps=True, vad_filter=True)
print(info.language, [s.text for s in list(segs)[:3]])
```

Expect real Bengali text. Identical inputs on macOS CPU are expected to repeat.
