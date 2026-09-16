# Benchmark: RTV "Goll Table" — Bengali talk-show speaker-attributed transcription

Episode `rtv_goll_table`, 409 s, https://youtu.be/t2l5UlEBpz0.

Three speakers, all with registry photos in `data/registry/`:

| Canonical label | Role | Turns | Speaking time |
|---|---|---:|---:|
| Dr Abdul Noor Tushar | guest, dominant | 57 | 183.3 s |
| Barrister A S M Shahriar Kabir | guest | 22 | 155.0 s |
| Dr Md Tawohidul Haque | host, interjects | 20 | 55.3 s |

**Zahed Ur Rahman is not in this episode**, though his photo sits in the
registry; any prediction of that name is a false positive by construction.

## Files

| file | what it is |
|---|---|
| `groundtruth_raw.srt` | the hand-edited draft, read-only, never modified |
| `v9_system.srt` | the untouched v9 system output being graded, read-only |
| `transcript.json` | normalised annotation, 99 turns, with per-turn reliability |
| `reference.rttm` | all 99 turns |
| `reference_reliable.rttm` | the 70 turns whose timestamps support time-based scoring |
| `speaker_map.json` | canonical identities and registry photos |
| `reliability.json` | repair log, tier counts, and every excluded span |
| `v9_baseline_metrics.json` | the measured v9 baseline |

Regenerate with `python -m evaluation.normalize_gt` then
`python -m evaluation.score_v9`. Both are deterministic.

## How the annotation was built, and what that costs

The draft was produced by **editing the system's own v9 subtitles**, not by
annotating the audio from scratch. Two consequences govern every number
derived from it:

1. **The timestamps are v9's timestamps.** All 88 numbered cues match v9 cue
   boundaries to the millisecond. So DER/JER computed here contain no
   independent boundary information: they measure *speaker confusion on fixed
   boundaries*. **A true diarization DER is not obtainable from this
   annotation** and needs boundaries re-annotated against the audio.
2. **The text was corrected, not transcribed blind.** Where v9 was wrong but
   plausible, the annotator may have left it. WER/CER are therefore **lower
   bounds** on true ASR error.

## Repairs applied (all logged in `reliability.json`)

| id | issue | resolution |
|---|---|---|
| R1 | one cue lost its index line, so a naive parser read its speaker as `"00"` | index recovered as v9 cue **90** by matching its timestamps (380.444–386.890) against `v9_system.srt` |
| R2 | one cue contained two speaker lines (host + guest) | split into two turns |
| R3 | 9 hand-typed blocks carried no timestamps at all (10 turns) | interpolated between timed neighbours, marked `inferred`, **excluded from all time-based metrics** |

Labels were already consistent: three distinct strings, zero spelling
variants, matching the registry filenames. No label normalisation was needed.
v9 cues 86–90 (a degenerate ASR repetition loop) were deleted by the annotator
and rewritten by hand; cue 90's timestamps survive via R1.

## Timing reliability tiers

| tier | turns | seconds | usable for |
|---|---:|---:|---|
| `reliable` | 70 | 274.7 | DER/JER, time-weighted accuracy, WER |
| `implausible_rate` | 18 | 23.2 | text metrics only |
| `implausible_span` | 1 | 76.3 | text metrics only |
| `inferred` | 10 | 19.3 | text metrics only |

`implausible_rate` = the cue implies a speaking rate above 35 characters/second
(Bengali conversation runs ~10–18); several are 50 ms cues carrying 70–105
characters. `implausible_span` is cue 61: 76.3 s for 31 characters, a grossly
wrong end boundary that swallows the 261–319 s region.

**Spans excluded from DER/JER** are enumerated individually under
`excluded_from_der` in `reliability.json`. The DER scoring region (UEM) is the
union of the `reliable` turns: **261.5 s, 63.9 % of the episode**. Scoring
without that UEM charges the 92 s of hypothesis outside the reference as false
alarm and inflates DER from 0.136 to 0.527; the UEM figure is the correct one.

38 of 99 turns overlap another turn in time, so the overlap phenomenon this
project targets is present and measurable in the annotation.
