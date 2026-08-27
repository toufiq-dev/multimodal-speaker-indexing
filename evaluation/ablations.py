"""Systematic ablation experiment matrix.

Defines an experiment as a named configuration of component swaps, executes
it against a prepared set of stage artifacts (diarization, transcription,
faces), and reports metrics per cell. Designed so every cell is reproducible:
artifacts are cached once and reused across all cells.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

from models import DiarizationSegment, FaceOccurrence, TranscribedSegment, FinalSegment
from evaluation.metrics import (
    wer, cer, cpwer, der_jer, speaker_name_accuracy,
    fusion_health_metrics,
)
from evaluation.baselines import (
    heuristic_fusion, text_only, run_identity_pipeline,
)


@dataclass
class ExperimentCell:
    name: str
    use_faces: bool = True
    use_nlp_names: bool = True
    transcription_mode: str = "word"      # "word" | "chunk" | "oracle_text"
    naming: str = "auto"                  # "auto" | "heuristic" | "gt"
    ground_truth_labels: Dict[str, str] = field(default_factory=dict)

    def tag(self) -> str:
        return (f"{self.name}|asr={self.transcription_mode}"
                f"|faces={'Y' if self.use_faces else 'N'}"
                f"|nlp={'Y' if self.use_nlp_names else 'N'}"
                f"|name={self.naming}")


def default_matrix() -> List[ExperimentCell]:
    """The standard ablation matrix for the thesis:
    full system, each single-modality removal, both baselines, oracle bound.
    """
    cells = [
        ExperimentCell("full_system"),
        ExperimentCell("no_faces", use_faces=False),
        ExperimentCell("no_nlp", use_nlp_names=False),
        ExperimentCell("no_faces_no_nlp", use_faces=False, use_nlp_names=False),
        ExperimentCell("chunk_asr", transcription_mode="chunk",
                       use_faces=False, use_nlp_names=False),
        ExperimentCell("heuristic_baseline", naming="heuristic"),
        ExperimentCell("text_only_baseline", naming="heuristic",
                       transcription_mode="oracle_text"),
        ExperimentCell("oracle_names", naming="gt"),
    ]
    return cells


def run_cell(
    cell: ExperimentCell,
    diarization: List[DiarizationSegment],
    transcribed_word: List[TranscribedSegment],
    transcribed_chunk: Optional[List[TranscribedSegment]],
    faces: List[FaceOccurrence],
    ordered_names: List[str],
    reference_turns_named: List[DiarizationSegment],
    reference_texts_by_speaker: Dict[str, str],
) -> Dict:
    """Execute one matrix cell and compute its metric bundle."""
    if cell.naming == "heuristic":
        finals = heuristic_fusion(diarization, transcribed_word)
    elif cell.transcription_mode == "oracle_text" and cell.naming == "heuristic":
        # Text-only baseline: bypass diarization entirely.
        finals = text_only(transcribed_word)
    else:
        tsrc = (transcribed_word if cell.transcription_mode == "word"
                else (transcribed_chunk or transcribed_word))
        finals = run_identity_pipeline(
            diarization, tsrc,
            faces=faces if cell.use_faces else [],
            ordered_names=ordered_names if cell.use_nlp_names else [],
            ground_truth_labels=cell.ground_truth_labels if cell.naming == "gt" else None,
        )

    hyp_by_spk: Dict[str, str] = {}
    for fs in finals:
        hyp_by_spk.setdefault(fs.speaker, "")
        hyp_by_spk[fs.speaker] += " " + fs.text

    c_min_wer, _perms = cpwer(reference_texts_by_speaker, hyp_by_spk)
    metrics = {
        "cell": cell.tag(),
        "WER_concat_best_perm": round(c_min_wer, 4) if c_min_wer != float("inf") else None,
        "CER_mean": round(
            sum(cer(reference_texts_by_speaker[s], hyp_by_spk.get(s, ""))
                 for s in reference_texts_by_speaker)
            / max(len(reference_texts_by_speaker), 1), 4),
        **der_jer(reference_turns_named,
                  [DiarizationSegment(fs.start, fs.end, fs.speaker)
                   for fs in finals]),
        "speaker_name_accuracy": round(
            speaker_name_accuracy(reference_turns_named, finals), 4),
        **{f"health_{k}": v for k, v in fusion_health_metrics(finals).items()},
    }
    return {"metrics": metrics, "final_segments": finals}


def run_matrix(
    cells: List[ExperimentCell],
    **kwargs,
) -> List[Dict]:
    return [run_cell(c, **kwargs) for c in cells]


def save_results(results: List[Dict], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for r in results:
        payload.append({
            "metrics": r["metrics"],
            "final_segments": [asdict(f) for f in r["final_segments"]],
        })
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return out
