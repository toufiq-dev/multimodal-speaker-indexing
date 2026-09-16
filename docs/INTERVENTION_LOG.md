# Intervention log — RTV Goll Table

The deliverable table. One row per **run**, two runs per configuration (C2).
Nothing is a result until it appears here with a run id and the exact command
that produced it.

Scoring command for every row:

```
python -m evaluation.score_run --hyp <run>/subtitles.srt --tag <run id> \
    --compare data/gt/rtv_goll_table/metrics_v9_c1.json
```

`--compare metrics_v9.json` still works but predates the coverage-matched
metrics, so those delta rows come back `null`. Compare against
`metrics_v9_c1.json`, which is v9 re-scored with the current scorer.

## The two constraints that govern the table

**C1 — the headline metric is confounded by word coverage.** Raw
`speaker_name_accuracy` is time-weighted over every reference turn, so a turn
the decoder transcribed as nothing scores zero while still counting in the
denominator. Measured on the real artefacts, by deleting host cues from v9's
own SRT and rescoring — the labels are untouched throughout:

| host cues deleted | raw accuracy | coverage-matched | turns covered | attributed-time |
|---:|---:|---:|---:|---:|
| 0/11 | 0.8899 | 0.8899 | 70/70 | 0.8899 |
| 3/11 | 0.8317 | 0.8625 | 68/70 | 0.8756 |
| 6/11 | 0.8070 | 0.8589 | 66/70 | 0.8723 |
| 11/11 | 0.8030 | 0.8546 | 66/70 | 0.8921 |

Deleting 3 of 11 host cues costs **5.8 points of raw accuracy and zero points
of identity quality**. Coverage-matched absorbs most of it (2.7 points) and
attributed-time nearly all (1.4). Neither is immune: a deleted cue can vacate
part of a turn that other cues keep "covered", and an overlapping neighbour
with the wrong name can then fill the vacated time. Read all three.

Note the mechanism precisely: **blanking a cue's text does not move raw
accuracy at all** (measured: 0.8899 at every blanking fraction). Raw accuracy
only moves when the cue *disappears*, which is what the pipeline does when a
turn transcribes to nothing. So C1 bites through dropped cues, not empty ones.

**C2 — the decoder is not reproducible at fixed configuration.** v9 and v10 ran
the same code and config on the same audio: 924 vs 866 words, 5-gram repeat
0.0641 vs 0.0000, WER 0.3675 vs 0.3436. The suspected mechanism is now
addressable rather than merely observed: Whisper re-decodes any window that
trips the compression-ratio or log-prob check by **sampling** at rising
temperatures, and that is the only stochastic step in the decode.
`WHISPER_TEMPERATURE_FALLBACK=0` decodes greedily at temperature 0 only. The
`nofallback` ablation row tests whether the spread collapses. **Suspected, not
demonstrated** — it needs the measurement below.

## Metrics every row must carry

| column | where it comes from | why |
|---|---|---|
| coverage-matched accuracy | `coverage_matched_accuracy.reliable_uem.accuracy` | C1 headline |
| raw accuracy | `speaker_name_accuracy.reliable_uem` | comparable to published v9/v10 |
| attributed-time accuracy | `attributed_time_accuracy.reliable_uem.accuracy` | coverage-free view of the cascade |
| word recall (aligned) | `asr.word_recall` | LCS match against reference; a repetition loop cannot inflate it |
| word count ratio | `asr.word_recall_proxy` | the old "word recall"; kept only for continuity |
| WER | `asr.corpus_wer` | lower bound, see caveats |
| 5-gram repeat | `repetition.hypothesis.ngram_repeat_ratio` | looping |
| overlap / non-overlap | both accuracy blocks | the primary target is the overlapping subset |

## Baselines

| run id | config | cov-matched | raw | attr-time | word recall | count ratio | WER | repeat | overlap / non-overlap (raw) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `v9_c1` | default decode | 0.8899 | 0.8899 | 0.8899 | 0.6873 | 0.8163 | 0.3675 | 0.0641 | 0.7803 / 0.9682 |
| `v10_c1` | same as v9 | 0.8851 | 0.8777 | 0.8851 | 0.6643 | 0.7650 | 0.3436 | 0.0000 | 0.7710 / 0.9538 |

v9's three accuracies coincide because the annotation was edited from v9's own
subtitles: its cues tile the reference exactly, so coverage is 70/70 turns and
274.7/274.7 s. The metrics only separate on runs that are not v9.

**v10's SRT was recovered** and is archived as `v10_system.srt`; `v10_c1` is
that file scored with the current scorer. Every future run must have its SRT
archived beside its metrics JSON the same way.

### C1 validated on the v9 -> v10 pair

The two runs shared identical diarization and face evidence, so the whole gap
is decode-driven. Decomposed:

| subset | raw delta | coverage-matched delta |
|---|---:|---:|
| reliable UEM | **-0.0122** | **-0.0048** |
| non-overlapping | **-0.0144** | **-0.0004** |
| overlapping | -0.0093 | -0.0093 |

**The non-overlapping "regression" was 97 % word coverage** — 0.0004 of identity
change, reported for a year as 1.4 points. The overlapping delta is identical
under both metrics, so *that* one is real: v10's 21 overlapping turns really
were labelled 0.9 points worse. Reporting raw accuracy alone conflates the two
and points the work at the wrong stage.

## Interventions

| run id | intervention | cov-matched | raw | attr-time | word recall | WER | repeat | overlap / non-overlap | command |
|---|---|---|---|---|---|---|---|---|---|
| v12 | word-merge fix (per-segment), default decode | 0.8359 | 0.8242 | 0.8359 | — | — | — | 0.6427 / 0.9772 | `kaggle_run.py --id rtv_goll_table_ep_v12` |
| v13 | word-merge fix (per-segment), default decode | 0.8851 | 0.8777 | 0.8851 | 0.6458 | 0.3825 | 0.0185 | 0.7710 / 0.9678 | `kaggle_run.py --id rtv_goll_table_ep_v13` |

## The noise floor, measured (v12 vs v13)

Same code, same configuration, same decode, two runs:

| | v12 | v13 | spread |
|---|---:|---:|---:|
| coverage-matched accuracy | 0.8359 | 0.8851 | **0.0492** |
| overlapping subset | 0.6427 | 0.7710 | **0.1283** |
| non-overlapping subset | 0.9772 | 0.9678 | 0.0094 |
| Shahriar Kabir accuracy | 0.6473 | 0.8775 | **0.2302** |

**This spread exceeds every delta measured so far, including v9→v10 (0.0122).**
The overlapping subset — the primary target — is the noisiest number available.

The spread is not diffuse. Comparing the two runs turn by turn on the
ground-truth grid, they **agree on 85 of 89 turns**. Four disagree, and one of
them dominates: 342.48–355.03 s (12.5 s), where v12 labels the host and v13
labels Shahriar (the reference agrees with v13). In a 114.4 s overlapping
subset, that single turn is ~11 % of the time — the 12.8-point swing is
essentially one turn flipping.

Cause is unfixed: the decode is stochastic (temperature fallback samples at
rising temperature), so the word set differs, so per-turn decisions differ.
Until the decode is seeded, no ASR or fusion intervention can be evaluated.
Reproducibility is now the blocking step, ahead of VAD and overlap work.

## Split-word artifacts: the per-segment merge did not work

Measured on the real subtitles (scripts in the session scratchpad):

| file | cues | cues starting with a combining mark | cues with a split fragment | fragment sites |
|---|---:|---:|---:|---:|
| ground truth | 89 | 0 | 0 | 0 |
| v9_system | 93 | 7 | 13 | 14 |
| v10_system | 94 | 7 | 15 | 16 |
| v12 | 88 | 8 | 14 | 16 |
| v13 | 92 | 8 | 16 | 20 |

The counts are unchanged, and the timestamps give the reason:
**30.00 s, 60.38 s, 86.62 s, 162.52 s — Whisper's 30-second window edges.** The
artifacts are words straddling a window boundary, whose tail is emitted as the
first token of the *next* segment. A merge that resets at each segment boundary
treats that tail as a word start, so it repairs nothing.

Fixed by merging over the whole decode, plus an unconditional rule that a token
opening with a Bengali dependent sign never starts a word (it only ever attaches
to a preceding letter). Unfixable residue: the very first word of the stream at
0.03 s ("্যাপ"), whose head was lost by the decoder — there is no preceding word
to merge it into.

## Caveats carried by every row

- **WER is a lower bound.** The ground-truth text was edited from v9's own
  output, so it is biased toward v9's word choices. Comparisons between *new*
  runs are the valid signal; any run's absolute WER advantage over v9 is
  understated and v9's own WER is flattered.
- **DER measures speaker confusion, not boundary quality.** The ground-truth
  timestamps are v9's. A true DER needs boundaries re-annotated against audio.
- **Single-stream Whisper cannot attribute simultaneous words.** If Step 3
  (overlap attribution) proves out of reach, the measured ceiling gets reported
  and the step stops. It does not get replaced with threshold tuning.
- **macOS decode is degenerate** (`docs/PLATFORM_ASR_FINDING.md`): it produces a
  repetition loop that still reports `language=bn`. `scripts/ablate_asr.py`
  refuses to run without CUDA for that reason. No measurement may come from it.
