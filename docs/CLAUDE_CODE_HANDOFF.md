# Handoff: Bengali Talk-Show Speaker Indexing — accurate speaker-labelled transcription

You are picking up an MSc thesis project. Read this whole document before touching
anything; it contains measured results, environment traps, and nine iterations of
history that are expensive to rediscover. The immediate goal is **accurate
speaker-labelled transcription of Bengali talk-show video**, and the current
system is not accurate enough. Treat everything below as evidence, not opinion.

---

## 1. Situation and goal

- **Degree:** MSc Data Science / CSE, Daffodil International University (DIU),
  Bangladesh. Supervisor: Prof. Dr. Sheak Rashed Haider Noori.
- **Thesis status:** an 18-credit thesis; a pre-defense committee called it "a
  project, not a thesis" and proposed evaluating it at 9 credits. It must be
  reframed as research (see §9).
- **Task:** given a multi-speaker Bengali talk-show video, produce a transcript in
  which every segment is attributed to the correct **named** speaker.
- **Target episode (the only one with hand-checked ground truth so far):**
  - URL: https://youtu.be/t2l5UlEBpz0 (RTV, "Goll Table"), duration **409 s**.
  - Speakers, per the student (who has watched it):
    - **Dr Abdul Noor Tushar** — speaks first and by far the most.
    - **Barrister A S M Shahriar Kabir** — also speaks, later.
    - **Dr Md Tawohidul Haque** — the **host**; interjects and interrupts.
    - They frequently **talk over each other** and interrupt.
    - **Zahed Ur Rahman does NOT appear in this episode.** (His reference photo
      sits in the registry from a different programme.)
- **Known transcription problem:** words go missing (e.g. "Begum Zia" comes out
  as "Begum"), and overlapping speech is mistranscribed and mislabelled.

## 2. Repository and entry points

- Repo: `github.com/toufiq-dev/multimodal-speaker-indexing` (branch `main`).
- Thesis report (separate repo/dir): `~/Developer/multimodal-speaker-indexing-Thesis-Report`
  (`thesis/chapter*.md`, built with `build_thesis_docx.py`).
- Pipeline stages: media (ffmpeg) → diarization (pyannote 3.1) → ASR
  (faster-whisper, CT2) → vision (InsightFace, 8 FPS, face tracking) → NER
  (BanglaBERT) → deterministic identity cascade → RAG index.
- Entry points:
  - `scripts/kaggle_run.py` — one-command Kaggle runner (env check, bootstrap,
    video discovery, pipeline, result summary + face diagnostics).
  - `scripts/run_episode.py` — drives one episode, writes outputs, health-checks.
  - `scripts/export_transcript.py` — result.json → readable .txt/.html.
  - `main.py` — the underlying pipeline.
- Key modules: `engines/{media,diarization,transcription,vision,active_speaker,nlp,fusion,clustering,rag}.py`,
  `evaluation/{metrics,dataset,tracking,ablations,baselines}.py`, `config.py`, `models.py`.

## 3. Kaggle environment (hard-won; ignore at your peril)

Every item below cost hours of debugging.

1. **Accelerator GPU T4×2, Internet ON, `HF_TOKEN` as a Kaggle *Secret* attached
   to the notebook**, whose account has accepted the gated licence for
   `pyannote/speaker-diarization-3.1`.
2. **NumPy must be 1.26.x.** `insightface==0.7.3` and `onnxruntime-gpu==1.19.2`
   are compiled against the NumPy 1.x C ABI; Kaggle ships NumPy 2.x and the
   kernel may already have imported it, so **pip installing 1.26.4 requires a
   kernel restart to take effect**. Symptom if ignored: `RecursionError` inside
   `numpy/_core` from `faster_whisper/feature_extractor.py`.
3. **cuDNN 9**: CTranslate2 `dlopen`s cuDNN, and glibc fixes the library search
   path at process start, so `LD_LIBRARY_PATH` must contain the torch wheel's
   `nvidia/cudnn/lib` **before the process launches**. Setting it in-process is
   too late. Symptom: `Unable to load any of {libcudnn_cnn.so.9...}`.
4. **/tmp is per-session.** The CT2 ASR model is converted into
   `/tmp/msi_scratch/models/bengaliAI_ct2` and disappears with a new session;
   `kaggle_run.py` re-converts automatically.
5. **`git pull` does not update already-imported modules.** Purge `sys.modules`
   for repo modules before importing, or the old code keeps running.
6. **Do not debug ASR on macOS.** Verified: the same audio, model and decode
   settings produce a **degenerate repetition loop** on Apple-Silicon CPU (the
   model still reports `language=bn`, so output looks plausible) and correct
   Bengali on CUDA. See `docs/PLATFORM_ASR_FINDING.md`.
7. Inputs: the video and the face-registry photos live in **two Kaggle Datasets**
   (`msi-video`, `msi-registry`), mounted under
   `/kaggle/input/datasets/<username>/<slug>/`.

## 4. Run commands

```bash
# one command; safe to re-run; prints environment, run, speaker table and diagnostics
!cd /kaggle/working/multimodal-speaker-indexing && git pull -q && \
    python scripts/kaggle_run.py --id rtv_goll_table_ep_vN

# re-print a previous run's summary without re-running
python scripts/kaggle_run.py --id <id> --report-only
```

Per-episode outputs land in `/kaggle/working/output/<id>/`:
`result.json`, `subtitles.srt`, `health.json`, `run_*.json`,
**`fusion_diagnostics.json`** (per-speaker face evidence) and
**`face_tracks.json`** (per-track frames, mouth motion, identity — the data
needed to calibrate anything motion-related).

## 5. Iteration history — what was tried and what actually happened

| run | change | measured outcome |
|---|---|---|
| v1 | baseline cascade | 78 segs. Labels: Shahriar 0.781, Tawohidul 0.696, `face_cluster_1`, and the Bengali clause `কলকাতা দল গেছিলাম তখন` as a "speaker". **Tushar absent.** Identity wrong. |
| v2 | registry gates `FACE_SIM_MARGIN=0.05`, `FACE_MIN_FRAME_FRACTION=0.4` (guessed) | **Regression.** The gates rejected *every* face match; labels became running-speech fragments (`গিয়েছি তদ্বির করতে`, `বলে দিচ্ছি`, `যখন`). |
| v3 | corroboration requirement for anchor names | Garbage reduced but `আর` ("and") still emitted; Tushar still misattributed. |
| v4 | diagnostics added (`fusion_diagnostics.json`) | No person names at all; **first real diagnosis**: Tushar matched **1 face in 616**; per-speaker presence 4–13%; the 7 reference photos are mutually well separated. |
| v5 | tracked 8 FPS pass, track-level identity (mean embedding), per-turn speaking-track override | Tushar finally appeared (6 segs) and votes rose to 78, but Shahriar still dominated. |
| v6 | **`FACE_SIM_THRESHOLD` 0.65 → 0.40, calibrated** | **The breakthrough.** Tushar became dominant (51 segs / 165 s). Side effects: a false positive (`Zahed Ur Rahman`) and the host's name over some of Shahriar's turns. |
| v7 | prominence gate (`ASD_MIN_TRACK_PRESENCE=0.30`) + global-support gate | Zahed removed. Remaining: same-shaped errors plus missing words. |
| v8 | word preservation (orphan words → nearest turn); per-turn override disabled | Words no longer dropped; labels reverted to per-speaker. |
| v9 | override re-enabled with calibrated motion gate (`ASD_MIN_MOUTH_MOTION=0.03`) | 93 cues: Tushar 59, Shahriar 21, Tawohidul 11, `face_cluster_3` 2. **Student reports no material improvement.** |

**Lesson that matters more than any single fix:** three regressions were caused
by *guessing* a threshold (v2, v3, v5) and one breakthrough came from
*measuring* one (v6). Measured numbers first, always.

## 6. Measured facts (use these; do not re-derive blindly)

**Reference photos are not the problem.** Pairwise ArcFace cosine over all 21
pairs of the 7 registry photos: **max 0.189**; Tushar ↔ Shahriar = **−0.076**.
They are fully separated (acceptance is 0.40). So no threshold can fix a
wrong-name problem caused by two people looking alike — that is not what is
happening.

**Genuine matches are weak because the reference photos differ from the video.**
Matching the on-screen face at 12 timestamps against the registry:
Tushar at t=5/10/20/60/90/120/240 s scores **0.45–0.61** (impostors 0.04–0.19);
Shahriar at t=300 s scores 0.645; the host at t=2 s and t=360 s scores 0.62–0.63.
The original threshold (0.65) sat **above the entire genuine distribution**,
which is why the speaker's own face was rejected for five iterations.

**Per-speaker face evidence (v6–v9, 4846 face occurrences, 130 tracks):**

| diarization cluster | faces | dominant identity | votes |
|---|---|---|---|
| SPEAKER_00 | 397 | Dr Md Tawohidul Haque | Tawohidul 227, Shahriar 62, Tushar 54 |
| SPEAKER_01 | 331 | Barrister Shahriar | Shahriar 87, Tawohidul 36, Tushar 9 |
| SPEAKER_02 | 2400 | Dr Abdul Noor Tushar | Tushar 1347, Zahed 222, Shahriar 126, Tawohidul 70 |
| SPEAKER_03 | 1719 | Barrister Shahriar | Shahriar 755, Tushar 116, Zahed 109, Tawohidul 25 |

**Clusters are mixed.** SPEAKER_03 contains 116 Tushar matches — i.e. some of
Tushar's speech was clustered with Shahriar. **This is a diarization error and
it is a major source of remaining wrong names.**

**Mouth motion separates talking from merely-present faces** (per-track mean,
8 FPS): silent panellist (Zahed) median 0.0150, **max 0.0233**; talking faces
0.038–0.074 (medians 0.038–0.061). Current gate `ASD_MIN_MOUTH_MOTION = 0.03`.

**Current configuration** (`config.py`): `FACE_SIM_THRESHOLD=0.40`,
`FACE_SIM_MARGIN=0.10`, `FACE_MIN_FRAME_FRACTION=0.0`, `VISION_ASD_FPS=8`,
`ASD_TRACK_IOU=0.3`, `ASD_MIN_MOUTH_MOTION=0.03`, `ASD_MIN_TRACK_PRESENCE=0.30`,
`ENABLE_TURN_OVERRIDE=1`, `WHISPER_VAD_FILTER=1`. Thresholds are
env-overridable (e.g. `FACE_SIM_THRESHOLD=0.45`, `WHISPER_VAD_FILTER=0`).

## 7. Remaining failure modes, ranked by expected impact

1. **No ground truth exists**, so nothing can be quantified or improved
   reliably. Every iteration so far has been judged by eye. This is the single
   biggest blocker — fix it first.
2. **Diarization merges/splits speakers** (mixed clusters, above), so correct
   face evidence lands on the wrong cluster label.
3. **Overlapping speech.** Whisper produces ONE text stream from mixed audio; it
   cannot say which words came from which voice. No post-processing fixes this.
   Needs speaker-attributed ASR (source separation + per-speaker transcription,
   or a target-speaker ASR model).
4. **ASR drops words** (e.g. "Begum Zia" → "Begum"). Likely Whisper VAD
   discarding quiet/overlapped speech before decoding; also int8 quantisation.
5. **The registry contains a person who is not in the episode** (Zahed), and
   reference photos are stylistically unlike the broadcast frames (0.45–0.61).

## 8. Recommended research plan (do it in this order)

**P0 — Build ground truth. This is the prerequisite for everything.**
Annotate ≥5 minutes (ideally the whole 409 s) of the target episode:
speaker turns with the true name, plus verbatim text. The schema already exists
in `evaluation/dataset.py` (RTTM + `speaker_map.json` + `transcript.json`).
Without this you cannot report speaker-name accuracy, DER, JER or WER, and you
cannot tell whether a change helped.

**P1 — Fix the ASR, measured.**
- Re-convert the CT2 model at **float16** (`--quantization float16`) and compare
  WER on the ground truth; int8 costs accuracy.
- Run the `WHISPER_VAD_FILTER=0` ablation — it likely recovers the dropped words
  in overlaps (watch for hallucinations).
- Compare model checkpoints: `bengaliAI/tugstugi_bengaliai-asr_whisper-medium`
  vs a large-v3 variant vs the LoRA adapter. Report WER/CER per checkpoint.

**P2 — Fix diarization, measured.**
- Report DER/JER (`evaluation/metrics.py:der_jer`) before changing anything.
- Try pyannote 4.x / `speaker-diarization-community-1`; try
  `NUM_SPEAKERS`/`MIN_SPEAKERS`/`MAX_SPEAKERS` hints; check overlap detection.
- The mixed clusters above are the target: measure per-cluster impurity.

**P3 — Overlap attribution (the hard, novel part).**
Investigate speaker-attributed ASR: separate the mixture per diarization speaker
(e.g. SepFormer/Conv-TasNet, or pyannote + SpeechBrain separation) and transcribe
each stream, or use a target-speaker ASR model. Evaluate on the overlapping
regions against the ground truth. This is the most defensible *research*
contribution in the whole project.

**P4 — Use face evidence to repair clusters (partially implemented).**
`engines/active_speaker.py` + `fusion._turn_identities_from_tracks` already
select the face whose mouth is moving (gate calibrated at 0.03) and override the
cluster label when the identity is globally supported. It is implemented and
tested but its effect on real accuracy is unmeasured — measure it against
ground truth (with the override on vs off) rather than eyeballing.

**P5 — Statistics and write-up.**
Report per-episode numbers, not pooled; use the existing metrics harness
(`evaluation/metrics.py`, `evaluation/ablations.py`); add more episodes (even
10-minute segments from different channels) so claims generalise.

## 9. Thesis framing (required by the committee)

The work is currently judged as engineering, not research. Two assets already
exist that make it a thesis:

1. **A measured negative finding.** "The visible face is the speaking face" is
   invalid in multi-camera broadcast: on this episode the speaker's face cleared
   the similarity threshold **1 time in 616 detections** while a silent
   co-panelist cleared it 55 times; and the original threshold sat above the
   entire genuine-match distribution. That is quantitative and publishable.
2. **A dataset/benchmark opportunity.** The supervisor's group publishes
   datasets and benchmarks (Data in Brief, low-resource Bangla NLP). A small,
   annotated Bengali talk-show benchmark — the one P0 produces — is the natural
   contribution, and it is also what makes every later claim measurable.

Chapter 4 already contains a case study (`§4.10`, including the v2 regression and
the v4 diagnostics) written in this failure-driven style; extend it with P0–P4
results. Do not report claims that are not backed by the annotation.

## 10. What to deliver

Working towards: a run on the target episode whose `subtitles.srt` is correct
speaker-for-speaker against the annotated ground truth, with reported
speaker-name accuracy, DER/JER and WER before/after each intervention — and a
thesis chapter that states the negative finding, the method, and the measured
improvement honestly, including the overlap limitation.

Where to start: **P0.** Nothing else can be trusted until it exists.
