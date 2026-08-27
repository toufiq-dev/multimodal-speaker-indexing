# System Architecture — Multimodal Bangla Talk-show Speaker Indexing

## High-level pipeline

```mermaid
flowchart TD
    A[Video file] --> B[engines/media.py<br/>ffmpeg audio extraction 16 kHz mono]
    B --> C[engines/diarization.py<br/>pyannote 3.1, SINGLE pass,<br/>num_speakers hint]
    B --> D[ASR engine - choose ONE]

    D -->|primary| E[engines/transcription.py<br/>faster-whisper long-form decode<br/>word_timestamps=True VAD beam search]
    D -->|fallback| F[engines/asr_lora.py<br/>HF Whisper + LoRA merged fp16<br/>chunked, words=[]]

    A --> G[engines/vision.py<br/>InsightFace frames @ VISION_FPS<br/>registry match -> DBSCAN fallback]
    B --> H[engines/nlp.py<br/>BanglaBERT NER on intro window<br/>+ 'আমি <নাম>' host anchor]

    C --> I[engines/fusion.py]
    E --> I
    F --> I
    G --> I
    H --> I

    I --> J[Identity resolution cascade:<br/>ground truth > registry face ><br/>host anchor > co-occurrence NER ><br/>face cluster > generic Speaker_N]
    I --> K[Final segment builder:<br/>word-midpoint containment, or<br/>proportional split fallback]
    J --> L[result.json + subtitles.srt]
    K --> L

    L --> M[evaluation/<br/>WER CER cpWER WDER DER JER<br/>face accuracy fusion health]
```

## Data contracts (the invariants that were violated in v1)

Every arrow above carries typed objects (`models.py`). The two contracts that
must never be broken again:

### Contract 1 — ASR → Fusion: word timestamps
`TranscribedSegment.words: List[WordToken]` is the ONLY mechanism by which
fusion can align text to diarization turns precisely. The primary engine
(faster-whisper, `word_timestamps=True`) always satisfies it. The LoRA chunk
engine emits `words=[]`; fusion therefore treats its output with a
conservative proportional-split path instead of pasting whole blocks.

### Contract 2 — Fusion output: every token exactly once
A word belongs to the turn containing its **midpoint**; a word-less block's
tokens are distributed across same-speaker overlapping turns **in proportion
to overlap duration**. Duplicate text rate across finals must be ≤ 5%
(asserted by `evaluation.metrics.assert_no_regression`).

## Identity resolution cascade (deterministic, no learned gate)

| Priority | Evidence | Source |
|---|---|---|
| 0 | Annotated ground-truth label | `ground_truth_labels` (evaluation mode) |
| 1 | Registered-face recognition ≥ threshold | `data/registry/*.jpg` via InsightFace |
| 2 | Host self-intro anchor `আমি <Name>` spoken by that speaker | `engines/nlp.extract_anchor_names_from_text` |
| 3 | Greedy name↔speaker co-occurrence matching (NEVER positional) | NER names × transcript evidence |
| 4 | Face cluster label (visual-only identity) | DBSCAN over unknown embeddings |
| 5 | Generic `Speaker_N` | fallback |

The untrained Xavier-init `GatingNetwork` was removed entirely: it emitted a
near-constant ≈0.5 (measured: 165/191 segments at exactly 0.5 confidence),
silently ordering speakers in the old resolution cascade.

## Module map

| Module | Responsibility | Key fix in this revision |
|---|---|---|
| `config.py` | Env-aware config, device routing, cleaning rules | collapse-not-delete regexes; `fw_device_and_compute()` routes MPS→CPU int8 for CTranslate2 |
| `engines/transcription.py` | faster-whisper long-form + word timestamps | midpoint-containment word→turn assignment replaces meaningless IoU |
| `engines/asr_lora.py` | LoRA fallback (chunked) | correctly-escaped regexes; fp16 merge (not 4-bit); UNKNOWN chunks never merged into mega-segments |
| `engines/diarization.py` | pyannote single pass | speaker-count hints via `config.diarization_kwargs()`; two-pass scheme deleted |
| `engines/vision.py` | frame sampling, face registry, clustering | frame time `i/fps` (was `(i+1)/fps`, 1 s offset); registry globs `.jpeg/.webp`; ctx_id=-1 CPU on macOS |
| `engines/nlp.py` | NER intro names + host anchor | greedy anchor capture with stopword trimming; empty-text guard; injectable NER pipe |
| `engines/fusion.py` | identity resolution + final segments | full rebuild — see header docstring |
| `evaluation/` | metrics, dataset manifest, baselines, ablations, tracking | new |

## Known limitations

- LoRA fallback still lacks word timestamps; its speaker alignment quality is
  bounded by chunk length. The recommended production path is merging the
  adapter into the fp16 base once, exporting with `ct2-transformers-converter`,
  and running through `engines/transcription.py`.
- No punctuation restoration stage yet (`ENABLE_PUNCTUATION_RESTORE` flag is a
  placeholder); the Bengali fine-tuned Whisper models emit unpunctuated text.
