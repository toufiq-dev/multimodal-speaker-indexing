"""Fusion engine for multimodal speaker identity resolution.

Rebuilt after the failed end-to-end run. Three structural changes:

1. ``create_final_segments`` no longer pastes whole transcribed blocks into
   every overlapping diarization turn (this produced 157/191 duplicated cues
   in the first full-show test). With word timestamps, finals are built from
   exactly the words whose midpoint falls inside the turn. Without words
   (LoRA fallback), block text is split PROPORTIONALLY across overlapping
   turns so every character is emitted exactly once.

2. The untrained Xavier-init GatingNetwork is REMOVED from the decision
   path (it output ~0.5 for everything -> meaningless confidences). Identity
   resolution now uses deterministic, interpretable evidence scores:
   registry faces > host self-intro anchor ("আমি <নাম>") > NER-name/speaker
   co-occurrence matching > face clusters > generic Speaker_N.

3. NER names are matched to speakers by CO-OCCURRENCE EVIDENCE (how often
   the name appears in a speaker's own transcript) via greedy bipartite
   matching -- never by list position.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from typing import Iterable, List, Dict, Tuple, Optional

from config import config
from models import (
    DiarizationSegment,
    TranscribedSegment,
    FaceOccurrence,
    FinalSegment,
)


# Bengali guest-introduction anchors ("সঙ্গে আছেন <Name>" etc.) are used by
# downstream tooling; host-anchor extraction itself lives in engines.nlp.
_GUEST_ANCHOR_RE = re.compile(
    r"(?:সঙ্গে আছেন|সাথে আছেন|হলেন|উপস্থিত)\s+([\u0980-\u09FF ]{2,40}?)(?=\s(?:সঙ্গে|সাথে)|$|[,।])"
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _index_by_time(items: Iterable, key) -> Tuple[list, List[float]]:
    """Sort items by a scalar time key and return (items, keys) for bisect.

    Turn-vs-point containment was previously a nested scan: every diarization
    turn rescanned every word and every face. On a full show that is
    ~10^3 turns x ~10^4 words (and a comparable face count) per call. Sorting
    once and binary-searching the range makes it O(n log n + matches).
    """
    ordered = sorted(items, key=key)
    return ordered, [key(i) for i in ordered]


class GatingFusion:
    """Deterministic multimodal identity resolver + final-segment builder.

    The class keeps its historical name for API compatibility, but contains
    no learned component anymore.
    """

    def __init__(self, ordered_names: Optional[List[str]] = None):
        self.ordered_names = ordered_names or []
        #: Per-speaker face evidence from the last resolve_identities() call,
        #: persisted by callers for threshold calibration.
        self.last_diagnostics: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Evidence collection
    # ------------------------------------------------------------------
    def _aggregate_faces_per_speaker(
        self,
        diarization: List[DiarizationSegment],
        faces: List[FaceOccurrence],
    ) -> Dict[str, List[FaceOccurrence]]:
        """Collect all faces observed during each speaker's talking time."""
        speaker_faces: Dict[str, List[FaceOccurrence]] = {seg.speaker_id: [] for seg in diarization}
        ordered_faces, face_times = _index_by_time(faces, lambda f: f.frame_time)
        for dia_seg in diarization:
            lo = bisect_left(face_times, dia_seg.start)
            hi = bisect_right(face_times, dia_seg.end)
            if lo < hi:
                speaker_faces[dia_seg.speaker_id].extend(ordered_faces[lo:hi])
        return speaker_faces

    def _best_registry_face_per_speaker(
        self, speaker_faces: Dict[str, List[FaceOccurrence]]
    ) -> Dict[str, Tuple[str, float]]:
        """Per speaker: (registry_name, confidence) by majority vote over faces.

        The previous rule took the single highest-similarity face across ALL of
        a speaker's frames and named the whole speaker from it. That is fragile:
        one ambiguous frame — or a camera cut to a listener — could name an
        entire speaker, and it produced a constant confidence per speaker.

        Now each detected face votes for its identity, and a name is accepted
        only when the evidence is unambiguous:

          * the face clears ``FACE_SIM_THRESHOLD`` (vision already set this), and
          * its top-1 similarity leads the runner-up by ``FACE_SIM_MARGIN``
            (two enrolled people who look alike produce a small gap), and
          * the winning identity appears in at least
            ``FACE_MIN_FRAME_FRACTION`` of the speaker's detected faces.

        The reported confidence is the MEAN similarity of the winning identity's
        faces, so it varies with evidence instead of being a copied constant.
        """
        out: Dict[str, Tuple[str, float]] = {}
        for spk, occs in speaker_faces.items():
            if not occs:
                continue
            votes: Dict[str, List[float]] = {}
            for f in occs:
                fid = f.resolved_face_id
                if fid == "UNKNOWN" or fid.startswith("face_cluster_"):
                    continue
                if f.face_confidence < config.FACE_SIM_THRESHOLD:
                    continue
                if (f.face_confidence - f.runner_up_confidence) < config.FACE_SIM_MARGIN:
                    # Ambiguous: the runner-up is too close to call.
                    continue
                votes.setdefault(fid, []).append(f.face_confidence)

            if not votes:
                continue

            name = max(votes, key=lambda n: len(votes[n]))
            sims = votes[name]
            if len(sims) / len(occs) < config.FACE_MIN_FRAME_FRACTION:
                # Present too rarely to attribute the whole speaker to it.
                continue
            out[spk] = (name, round(sum(sims) / len(sims), 3))
        return out

    def _best_cluster_per_speaker(
        self, speaker_faces: Dict[str, List[FaceOccurrence]]
    ) -> Dict[str, Tuple[str, int]]:
        """Per speaker: dominant (cluster_label, hit_count) among clustered faces."""
        counts: Dict[str, Dict[str, int]] = {}
        for spk, occs in speaker_faces.items():
            for f in occs:
                fid = f.resolved_face_id
                if fid and fid.startswith("face_cluster_") and fid != "face_cluster_noise":
                    counts.setdefault(spk, {})
                    counts[spk][fid] = counts[spk].get(fid, 0) + 1
        out: Dict[str, Tuple[str, int]] = {}
        for spk, c in counts.items():
            label = max(c.items(), key=lambda kv: kv[1])[0]
            out[spk] = (label, c[label])
        return out

    def registry_face_diagnostics(
        self, speaker_faces: Dict[str, List[FaceOccurrence]]
    ) -> Dict[str, Dict]:
        """Per-speaker face evidence, for calibrating the rejection gates.

        Reports how many faces were detected per diarization speaker, which
        identities they voted for, the mean similarity per identity, and the
        top-1 / runner-up gap on the best face. Persisted as
        ``fusion_diagnostics.json`` so ``FACE_SIM_MARGIN`` and
        ``FACE_MIN_FRAME_FRACTION`` are chosen from measured gaps instead of
        guesses — guessing them once removed every real name from the output.
        """
        diag: Dict[str, Dict] = {}
        for spk, occs in speaker_faces.items():
            votes: Dict[str, int] = {}
            sims: Dict[str, List[float]] = {}
            best_sim, best_runner = 0.0, 0.0
            for f in occs:
                fid = f.resolved_face_id
                if fid == "UNKNOWN" or fid.startswith("face_cluster_"):
                    continue
                votes[fid] = votes.get(fid, 0) + 1
                sims.setdefault(fid, []).append(f.face_confidence)
                if f.face_confidence > best_sim:
                    best_sim, best_runner = f.face_confidence, f.runner_up_confidence
            winner = max(votes, key=lambda n: votes[n]) if votes else None
            diag[spk] = {
                "n_faces": len(occs),
                "votes": votes,
                "mean_sim": {k: round(sum(v) / len(v), 3) for k, v in sims.items()},
                "winner": winner,
                "winner_presence": (
                    round(votes[winner] / len(occs), 3) if winner and occs else 0.0),
                "best_sim": round(best_sim, 3),
                "best_runner_up": round(best_runner, 3),
                "best_margin": round(best_sim - best_runner, 3),
            }
        return diag

    @staticmethod
    def _host_anchor_evidence(
        diarization: List[DiarizationSegment],
        transcribed: List[TranscribedSegment],
        known_names: Optional[set] = None,
    ) -> List[Tuple[str, str, int]]:
        """Find 'আমি <Name>' self-introductions.

        Returns [(speaker_id, name, hit_count)] sorted by hit count desc.
        This is the primary textual anchor for identifying the HOST, since a
        presenter says "আমি X" while speaking -- direct first-person evidence.
        Name capture/trimming logic lives in engines.nlp (single source).

        ``known_names`` (registry identities + NER output) lets a capture be
        accepted on corroboration alone; without it, a capture must look like a
        name, because an unvalidated clause from running speech must never
        become a speaker label.
        """
        from engines.nlp import extract_anchor_names_from_text  # lazy: avoids heavy import at module load

        hits: Dict[Tuple[str, str], int] = {}
        for tseg in transcribed:
            spk = tseg.speaker_id
            if not spk or spk == "UNKNOWN":
                continue
            for name in extract_anchor_names_from_text(tseg.text, known_names):
                hits[(spk, name)] = hits.get((spk, name), 0) + 1
        return sorted(
            ((spk, name, n) for (spk, name), n in hits.items()),
            key=lambda x: -x[2],
        )

    @staticmethod
    def _cooccurrence_scores(
        names: List[str],
        diarization: List[DiarizationSegment],
        transcribed: List[TranscribedSegment],
    ) -> Dict[Tuple[str, str], float]:
        """Evidence score[(speaker, name)] = weighted mentions of the name in
        that speaker's own transcript (intro window weighted higher)."""
        scores: Dict[Tuple[str, str], float] = {}
        lowered_names = [(n, n.replace(" ", "")) for n in names]
        for tseg in transcribed:
            spk = tseg.speaker_id
            if not spk or spk == "UNKNOWN":
                continue
            compact = tseg.text.replace(" ", "")
            weight = 2.0 if tseg.start < config.NLP_INTRO_SECONDS else 1.0
            for name, compact_name in lowered_names:
                if compact_name and compact_name in compact:
                    key = (spk, name)
                    scores[key] = scores.get(key, 0.0) + weight
        return scores

    # ------------------------------------------------------------------
    # Identity resolution (deterministic cascade)
    # ------------------------------------------------------------------
    def resolve_identities(
        self,
        diarization: List[DiarizationSegment],
        transcribed: List[TranscribedSegment],
        faces: List[FaceOccurrence],
        ground_truth_labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Tuple[str, float]]:
        """Map each diarization speaker_id -> (resolved_name, confidence).

        Cascade (first match wins):
          0. Ground-truth annotation overrides (evaluation mode).
          1. Registered face recognition above threshold.
          2. Host anchor ("আমি <Name>") spoken by that very speaker.
          3. Greedy name<->speaker matching on co-occurrence evidence.
          4. Face cluster labels.
          5. Generic Speaker_N.
        """
        resolved: Dict[str, Tuple[str, float]] = {}
        used_names: set = set()

        speaker_ids = list(dict.fromkeys(s.speaker_id for s in diarization))

        # Pass 0: explicit ground-truth labels (annotated evaluation data).
        if ground_truth_labels:
            for spk in speaker_ids:
                gt = ground_truth_labels.get(spk)
                if gt:
                    resolved[spk] = (gt, 1.0)
                    used_names.add(gt)

        speaker_faces = self._aggregate_faces_per_speaker(diarization, faces)

        # Face evidence per speaker, recorded for calibration. Callers persist
        # this (run_episode writes fusion_diagnostics.json) so the margin and
        # presence thresholds can be chosen from measured gaps rather than
        # guessed — guessing them once removed every real name from the output.
        self.last_diagnostics = self.registry_face_diagnostics(speaker_faces)

        # Corroborated names: registry identities seen in the video plus the NER
        # output. A textual anchor matching one of these is trusted; anything
        # else must look like a name on its own.
        registry_names = {
            f.resolved_face_id for f in faces
            if f.resolved_face_id != "UNKNOWN"
            and not f.resolved_face_id.startswith("face_cluster_")
        }
        known_names = set(self.ordered_names) | registry_names

        # Pass 1: registry faces.
        reg = self._best_registry_face_per_speaker(speaker_faces)
        for spk in speaker_ids:
            if spk in resolved:
                continue
            if spk in reg and reg[spk][1] >= config.FACE_SIM_THRESHOLD:
                name, conf = reg[spk]
                if name not in used_names:
                    resolved[spk] = (name, round(conf, 3))
                    used_names.add(name)

        # Pass 2: host self-intro anchor.
        for spk, name, hit_count in self._host_anchor_evidence(
                diarization, transcribed, known_names):
            if spk in resolved or name in used_names or hit_count < 1:
                continue
            conf = min(0.95, 0.6 + 0.1 * hit_count)
            resolved[spk] = (name, conf)
            used_names.add(name)

        # Pass 3: co-occurrence greedy matching (NEVER positional).
        candidates = [n for n in self.ordered_names
                      if n and n not in used_names]
        if candidates and len(resolved) < len(speaker_ids):
            scores = self._cooccurrence_scores(candidates, diarization, transcribed)
            pairs = sorted(scores.items(), key=lambda kv: -kv[1])
            for (spk, name), sc in pairs:
                if spk in resolved or name in used_names or sc <= 0:
                    continue
                conf = min(0.9, 0.5 + 0.05 * sc)
                resolved[spk] = (name, round(conf, 3))
                used_names.add(name)

        # Pass 4: face clusters (keep as visual-only identity).
        clusters = self._best_cluster_per_speaker(speaker_faces)
        for spk in speaker_ids:
            if spk in resolved:
                continue
            if spk in clusters:
                label, hits = clusters[spk]
                if label not in used_names:
                    resolved[spk] = (label, min(0.5, 0.1 * hits))
                    used_names.add(label)

        # Pass 5: generic labels.
        counter = 1
        for spk in speaker_ids:
            if spk in resolved:
                continue
            while f"Speaker_{counter}" in used_names:
                counter += 1
            resolved[spk] = (f"Speaker_{counter}", 0.5)
            used_names.add(f"Speaker_{counter}")
            counter += 1

        return resolved

    # ------------------------------------------------------------------
    # Final segment construction
    # ------------------------------------------------------------------
    @staticmethod
    def _iter_words(transcribed: List[TranscribedSegment]):
        for tseg in transcribed:
            for w in tseg.words:
                yield w

    def _finals_from_words(
        self,
        diarization: List[DiarizationSegment],
        transcribed: List[TranscribedSegment],
        resolved_names: Dict[str, Tuple[str, float]],
    ) -> List[FinalSegment]:
        """Exact construction: a turn's text = the words whose midpoint lies
        inside the turn. Kills the duplication bug class by construction."""
        finals: List[FinalSegment] = []
        # Index words by midpoint once instead of regenerating the full word
        # stream inside the per-turn loop.
        ordered_words, word_mids = _index_by_time(
            self._iter_words(transcribed), lambda w: (w.start + w.end) / 2.0)

        for dia_seg in sorted(diarization, key=lambda d: d.start):
            lo = bisect_left(word_mids, dia_seg.start)
            hi = bisect_right(word_mids, dia_seg.end)
            words_in_turn = ordered_words[lo:hi]
            text = _norm(" ".join(w.word for w in words_in_turn))
            if not text:
                continue
            speaker_name, confidence = resolved_names.get(
                dia_seg.speaker_id, ("UNKNOWN", 0.0))
            finals.append(FinalSegment(
                start=dia_seg.start,
                end=dia_seg.end,
                speaker=speaker_name,
                text=text,
                confidence=confidence,
            ))
        return finals

    def _finals_from_proportional_split(
        self,
        diarization: List[DiarizationSegment],
        transcribed: List[TranscribedSegment],
        resolved_names: Dict[str, Tuple[str, float]],
    ) -> List[FinalSegment]:
        """Word-less fallback (e.g. LoRA chunk path): split each transcribed
        block's text across overlapping same-speaker turns in proportion to
        overlap duration. Each block's characters are emitted exactly once."""
        text_parts: List[List[str]] = [[] for _ in diarization]
        order = sorted(range(len(diarization)), key=lambda i: diarization[i].start)

        for tseg in transcribed:
            overlaps: List[Tuple[int, float]] = []
            for idx in order:
                d = diarization[idx]
                if d.speaker_id != tseg.speaker_id:
                    continue
                ov = min(tseg.end, d.end) - max(tseg.start, d.start)
                if ov > 0:
                    overlaps.append((idx, ov))
            if not overlaps:
                continue
            total_ov = sum(ov for _, ov in overlaps)
            tokens = tseg.text.split()
            n = len(tokens)
            cursor = 0
            last_i = len(overlaps) - 1
            for j, (idx, ov) in enumerate(overlaps):
                if j == last_i:
                    take = tokens[cursor:]
                else:
                    take_n = int(round(n * (ov / total_ov))) if total_ov > 0 else 0
                    take = tokens[cursor:cursor + take_n]
                    cursor += len(take)
                if take:
                    text_parts[idx].extend(take)

        finals: List[FinalSegment] = []
        for idx in order:
            text = _norm(" ".join(text_parts[idx]))
            if not text:
                continue
            d = diarization[idx]
            speaker_name, confidence = resolved_names.get(d.speaker_id, ("UNKNOWN", 0.0))
            finals.append(FinalSegment(
                start=d.start, end=d.end,
                speaker=speaker_name, text=text, confidence=confidence,
            ))
        return finals

    def create_final_segments(
        self,
        diarization: List[DiarizationSegment],
        transcribed: List[TranscribedSegment],
        resolved_names: Dict[str, Tuple[str, float]],
    ) -> List[FinalSegment]:
        """Create FinalSegments directly from diarization turns."""
        has_words = any(tseg.words for tseg in transcribed)
        if has_words:
            return self._finals_from_words(diarization, transcribed, resolved_names)
        return self._finals_from_proportional_split(
            diarization, transcribed, resolved_names)


def run_fusion_pipeline(
    diarization: List[DiarizationSegment],
    transcribed: List[TranscribedSegment],
    faces: List[FaceOccurrence],
    ordered_names: Optional[List[str]] = None,
    ground_truth_labels: Optional[Dict[str, str]] = None,
) -> List[FinalSegment]:
    """Run full fusion pipeline.

    Args:
        diarization: Speaker diarization segments.
        transcribed: Transcribed segments with speaker assignments.
        faces: Face occurrences with confidence and lip-sync.
        ordered_names: Names extracted from NLP intro (optional).
        ground_truth_labels: Optional annotated mapping speaker_id -> real
            name (evaluation mode / registry-less datasets).

    Returns:
        List of FinalSegment with resolved speaker names.
    """
    fusion = GatingFusion(ordered_names=ordered_names)
    resolved = fusion.resolve_identities(
        diarization, transcribed, faces, ground_truth_labels=ground_truth_labels)
    return fusion.create_final_segments(diarization, transcribed, resolved)


def fusion_diagnostics(
    diarization: List[DiarizationSegment],
    faces: List[FaceOccurrence],
    ordered_names: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Per-speaker face evidence, for choosing the registry rejection gates.

    Run the pipeline once, read ``fusion_diagnostics.json``, and set
    ``FACE_SIM_MARGIN`` / ``FACE_MIN_FRAME_FRACTION`` from the observed
    ``best_margin`` and ``winner_presence`` distributions — never by guessing.
    """
    fusion = GatingFusion(ordered_names=ordered_names)
    speaker_faces = fusion._aggregate_faces_per_speaker(diarization, faces)
    return fusion.registry_face_diagnostics(speaker_faces)
