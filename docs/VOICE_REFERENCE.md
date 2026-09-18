# Voice reference registry

Faces answer *who is on screen*. In an overlapping turn the camera frames both
speakers, so a face match — however confident — cannot say which of them is
talking. A voiceprint is derived from the speech itself, so this adds the one
evidence source the failing subset actually needs.

It is the mirror image of the face registry:

| | face registry | voice registry |
|---|---|---|
| files | `data/registry/<Name>.jpg` | `data/registry/<Name>.mp3` |
| embedding | ArcFace (insightface) | speaker verification (pyannote) |
| aggregation | per-speaker majority vote | per-cluster trimmed centroid |
| gates | `FACE_SIM_THRESHOLD` / `_MARGIN` | `VOICE_SIM_THRESHOLD` / `_MARGIN` |

**Off by default.** With `VOICE_POLICY` unset the pipeline is byte-for-byte the
previous behaviour: the voice stage is a no-op and no voice artefacts are
written. Nothing about v9–v14 changes.

---

## 1. The three policies

| `VOICE_POLICY` | effect |
|---|---|
| `off` (default) | voiceprints are never computed. Current behaviour. |
| `fallback` | voice may name only the speakers the **face pass left unnamed**, ahead of the text heuristics. Strictly additive: it cannot overwrite a face match. |
| `override` | as `fallback`, and additionally may **replace** a face-assigned name when the voiceprint is stronger by more than `VOICE_OVERRIDE_MARGIN`. |

Both are inserted as step **1b** of the fusion cascade — after the calibrated
face pass, before the textual heuristics, because a measured acoustic match
outranks a co-occurrence guess but must not silently overrule a cited face
match under `fallback`.

### The gates are not guessed

`VOICE_SIM_THRESHOLD=0.45` and `VOICE_SIM_MARGIN=0.05` are **starting points,
not calibrated values**. They are in a different embedding space from ArcFace,
so the face thresholds do not transfer. §5 measures them on your episode before
any result is claimed.

---

## 2. Data handling — read this before running

The five reference clips are real people's voices:

| file | duration |
|---|---:|
| `Dr_Md_Tawohidul_Haque.mp3` | 697 s |
| `Dr_Abdul_Noor_Tushar.mp3` | 283 s |
| `Barrister_A_S_M_Shahriar_Kabir.mp3` | 192 s |
| `MA_Aziz.mp3` | 158 s |
| `Abu_Hena_Razzaki.mp3` | 151 s |

`.gitignore` excludes `*.mp3` **on purpose**. Do not un-ignore them: a voice
recording is a biometric and the face photos are already treated as the only
committable registry asset. `git push` therefore does **not** carry the audio —
it must travel as a Kaggle dataset.

Of the five, only **three are in this episode** (`Dr Abdul Noor Tushar`,
`Barrister A S M Shahriar Kabir`, `Dr Md Tawohidul Haque`); `MA Aziz` and
`Abu Hena Razzaki` are enrolled but absent. That is a free impostor test: any
cluster the matcher assigns to one of those two is a measured false positive,
and `voice_verify.py` reports it as `wrong`.

---

## 3. Prerequisite: get the audio onto Kaggle

The clips live in `data/registry/` locally but are not in git. Either:

**a. Upload them as a dataset** — Kaggle → Datasets → New Dataset, upload the
five `.mp3` files, name it e.g. `msi-voice-refs`, then Add Input → attach it to
the notebook. `scripts/kaggle_run.py` copies **both** photos and audio out of
`/kaggle/input` into the registry (it previously copied images only, so the
clips would have been dropped silently).

**b. Already inside an attached dataset.** If the audio is in the same dataset
as the photos, nothing to do.

Verify with one line:

```bash
!ls -la /kaggle/working/registry/rtv_goll_table/
```

You want five `.mp3` alongside the `.jpg`s. `kaggle_run.py` reports the counts
as `registry : N photo(s), M voice reference(s)` and warns if the policy is on
with no audio.

---

## 4. Model access

The embedder is tried in order and the first that loads wins:

1. `pyannote/wespeaker-voxceleb-resnet34-LM`
2. `pyannote/embedding`

Both are gated. (1) is first because `pyannote/speaker-diarization-3.1` already
downloads it, so any account whose diarization works has already accepted its
licence. Override with `VOICE_MODEL=<hf-id>` if needed.

If neither loads, the stage prints the Hugging Face error and the run continues
**face-only** — a missing supporting evidence source never fails an episode.

---

## 5. The run sequence

Each step is one Kaggle cell. Copy the whole line.

### 5.0 Get the code onto the box

```bash
!cd /kaggle/working/multimodal-speaker-indexing && git pull -q
```

### 5.1 `v15` — control, confirms the refactor is inert

```bash
!cd /kaggle/working/multimodal-speaker-indexing && python scripts/kaggle_run.py --id rtv_goll_table_ep_v15
```

Same configuration as v14. Compare it against v14: this is the regression check
for the fusion refactor (the cascade is now driven directly instead of through
`run_fusion_pipeline`, so the diagnostics are not recomputed) and for the new
`diarization.json` write. **If v15 does not reproduce v14 within the v12–v13
spread (0.049 coverage-matched), stop — the refactor, not the voice feature, is
the variable.**

It also produces `diarization.json`, which every later step needs.

### 5.2 `v16` — instrument run: compute voice evidence, default gates

```bash
!cd /kaggle/working/multimodal-speaker-indexing && VOICE_POLICY=fallback python scripts/kaggle_run.py --id rtv_goll_table_ep_v16
```

This is where the embeddings are produced. It writes:

```
voiceprints.npz            raw cluster + enrolment embeddings (for §5.3)
voice_diagnostics.json     the full similarity matrix, per-cluster
voice_decisions.json       what the voice pass actually did per cluster
diarization.json           the turns the clusters came from
```

Score it like any run, but treat its **identity** result as provisional: the
gates are still the uncalibrated defaults. Its value is the artefacts.

### 5.3 Calibrate — no GPU, no audio, milliseconds

```bash
!cd /kaggle/working/multimodal-speaker-indexing && python scripts/voice_verify.py --run-dir /kaggle/working/output/rtv_goll_table_ep_v16 --gt-dir data/gt/rtv_goll_table --save /kaggle/working/output/rtv_goll_table_ep_v16/voice_sweep.json
```

It derives the cluster→name truth by overlapping the run's clusters with the
annotated transcript (the run supplies the segmentation, the **annotation**
supplies the identity — nothing from the hypothesis decides the right answer),
then re-scores every threshold offline.

Read three numbers:

- **`max off-diagonal`** in the cross-matrix — the impostor ceiling. No
  threshold at or below it can ever separate that pair of enrolled people. If
  it sits near the genuine matches, the honest conclusion is that voice cannot
  distinguish them on this registry.
- **`coherence`** per cluster — a low value means that cluster contains more
  than one voice, i.e. a **diarization** error. No gate here repairs that; it is
  Step 2 work.
- The **sweep** — `wrong` counts clusters named *incorrectly*. The recommended
  row names the most clusters with **zero** misnaming. A row that buys one more
  `correct` at the cost of one `wrong` is not obviously a gain: in a
  time-weighted metric the two turns are rarely the same length.

### 5.4 `v17` — the measured test

Substitute the gate values printed as RECOMMENDED:

```bash
!cd /kaggle/working/multimodal-speaker-indexing && VOICE_POLICY=fallback VOICE_SIM_THRESHOLD=<T> VOICE_SIM_MARGIN=<M> python scripts/kaggle_run.py --id rtv_goll_table_ep_v17
```

Then the same with `VOICE_POLICY=override` **only if** the sweep shows a cluster
whose voiceprint names someone the face pass named differently — that is the
case `override` exists for, and the case `fallback` cannot reach:

```bash
!cd /kaggle/working/multimodal-speaker-indexing && VOICE_POLICY=override VOICE_SIM_THRESHOLD=<T> VOICE_SIM_MARGIN=<M> python scripts/kaggle_run.py --id rtv_goll_table_ep_v18
```

### 5.5 Score every run

```bash
!cd /kaggle/working/multimodal-speaker-indexing && python -m evaluation.score_run --hyp /kaggle/working/output/rtv_goll_table_ep_v17/subtitles.srt --tag v17 --compare data/gt/rtv_goll_table/metrics_v9_c1.json
```

Repeat per run id. Compare against `metrics_v9_c1.json` (v9 re-scored with the
coverage-matched metrics), not `metrics_v9.json` — the latter predates them and
those delta cells come back `null`.

### 5.6 Per C2, repeat the winner

The v12/v13 spread is 0.049 coverage-matched and 0.128 on the overlapping
subset. **Any voice delta smaller than that is noise.** One run of the winning
configuration cannot establish an effect; run it twice.

---

## 6. Reading the artefacts

`voice_diagnostics.json`

| key | meaning |
|---|---|
| `enrolled` | per name: windows embedded, windows kept, seconds, **coherence** |
| `enrolled_cross` | every pair of enrolled speakers — the impostor distribution |
| `clusters.<SPEAKER_xx>` | turns used, seconds used vs skipped, windows, **coherence**, best/runner-up/margin, `scores` (all similarities), decision + reason |
| `summary.rejected_reasons` | `below_threshold` / `ambiguous_margin` / `low_coherence` / `no_voiceprint` |

`voice_decisions.json` — per cluster: `filled`, `agreed_with_face`,
`overrode_face`, `kept_face`, `skipped_name_taken`. The transcript alone cannot
distinguish "voice agreed with the face" from "voice was ignored"; this can.

`voiceprints.npz` — the raw vectors. This is what makes §5.3 possible offline.

---

## 7. Known limits (stated, not buried)

- **Cluster-level only.** Voice names a diarization cluster; it does not split
  an overlapping turn into two streams. It can fix *which* speaker a cluster is,
  never *which words within a mixed turn* belong to whom. That remains Step 3.
- **Gated model.** Needs `HF_TOKEN` and accepted licences, same as diarization.
- **Embeddings are unverified locally.** The macOS venv has drifted (pyannote
  **4.0.7** installed, **3.3.2** pinned); both export `Model`/`Inference` and
  both accept `{"waveform", "sample_rate"}` and resample internally, and the
  code path was checked against both, but the model call itself has only been
  exercised on Kaggle. Everything around it is unit-tested (`tests/test_voice.py`).
- **Enrolment quality is a real risk.** The reference clips are broadcast audio
  and may contain other speakers. The trimmed centroid and the reported
  `coherence` exist for that, but a clip that is mostly someone else will still
  enrol the wrong voice — check `coherence` in the enrolment table before
  trusting any match.
