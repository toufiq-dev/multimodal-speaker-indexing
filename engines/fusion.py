"""Fusion engine for multimodal speaker identity resolution."""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from config import config
from models import (
    DiarizationSegment,
    TranscribedSegment,
    FaceOccurrence,
    FinalSegment,
)


class GatingNetwork(nn.Module):
    """MLP for audio-visual speaker-face association probability."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def train_step(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Single training step."""
        self.train()
        optimizer.zero_grad()
        outputs = self(features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss.item()


class GatingFusion:
    """Fuses diarization, transcription, vision, and NLP into final speaker segments."""

    def __init__(self, ordered_names: Optional[List[str]] = None):
        self.ordered_names = ordered_names or []
        self.gating_net = GatingNetwork()
        self._trained = False

    def _get_audio_confidence(
        self,
        dia_seg: DiarizationSegment,
        transcribed: List[TranscribedSegment],
    ) -> float:
        """Check if any transcribed segment matches this speaker."""
        for tseg in transcribed:
            if tseg.speaker_id == dia_seg.speaker_id:
                if not (tseg.end <= dia_seg.start or tseg.start >= dia_seg.end):
                    return 1.0
        return 0.0

    def _aggregate_faces_per_speaker(
        self,
        diarization: List[DiarizationSegment],
        faces: List[FaceOccurrence],
    ) -> Dict[str, List[FaceOccurrence]]:
        """Collect all faces overlapping each speaker's segments."""
        speaker_faces: Dict[str, List[FaceOccurrence]] = {seg.speaker_id: [] for seg in diarization}

        for face in faces:
            for dia_seg in diarization:
                if face.frame_time >= dia_seg.start and face.frame_time <= dia_seg.end:
                    speaker_faces[dia_seg.speaker_id].append(face)

        return speaker_faces

    def _get_best_face_for_speaker(
        self,
        speaker_id: str,
        speaker_faces: Dict[str, List[FaceOccurrence]],
    ) -> Optional[FaceOccurrence]:
        """Select face with highest average confidence for this speaker."""
        faces = speaker_faces.get(speaker_id, [])
        if not faces:
            return None

        # Group by resolved_face_id and compute average confidence
        face_groups: Dict[str, List[FaceOccurrence]] = {}
        for face in faces:
            face_groups.setdefault(face.resolved_face_id, []).append(face)

        best_face = None
        best_avg_conf = -1.0

        for face_id, group in face_groups.items():
            avg_conf = sum(f.face_confidence for f in group) / len(group)
            if avg_conf > best_avg_conf:
                best_avg_conf = avg_conf
                # Return representative face (highest individual confidence)
                best_face = max(group, key=lambda f: f.face_confidence)

        return best_face

    def _extract_features(
        self,
        dia_seg: DiarizationSegment,
        transcribed: List[TranscribedSegment],
        best_face: Optional[FaceOccurrence],
    ) -> Tuple[float, float, float]:
        """Extract [audio_conf, face_conf, lip_sync] for gating network."""
        audio_conf = self._get_audio_confidence(dia_seg, transcribed)

        if best_face:
            face_conf = best_face.face_confidence
            lip_sync = best_face.lip_sync_score
        else:
            face_conf = 0.0
            lip_sync = 0.0

        return audio_conf, face_conf, lip_sync

    def train_gating_network(
        self,
        diarization: List[DiarizationSegment],
        transcribed: List[TranscribedSegment],
        faces: List[FaceOccurrence],
        labels: List[float],
        epochs: int = 50,
        lr: float = 1e-3,
    ):
        """Train gating network with supervised labels (1=match, 0=mismatch)."""
        if len(diarization) != len(labels):
            raise ValueError("Labels must match diarization segments")

        speaker_faces = self._aggregate_faces_per_speaker(diarization, faces)

        features_list = []
        for dia_seg in diarization:
            best_face = self._get_best_face_for_speaker(dia_seg.speaker_id, speaker_faces)
            feats = self._extract_features(dia_seg, transcribed, best_face)
            features_list.append(feats)

        features = torch.tensor(features_list, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.gating_net.parameters(), lr=lr)
        criterion = nn.BCELoss()

        for _ in range(epochs):
            self.gating_net.train_step(features, labels_tensor, optimizer, criterion)

        self._trained = True

    def resolve_identities(
        self,
        diarization: List[DiarizationSegment],
        transcribed: List[TranscribedSegment],
        faces: List[FaceOccurrence],
    ) -> Dict[str, Tuple[str, float]]:
        """
        Map each diarization speaker_id to a resolved name and confidence.

        Returns:
            Dict mapping speaker_id -> (resolved_name, confidence)
        """
        speaker_faces = self._aggregate_faces_per_speaker(diarization, faces)

        # Get best face per speaker
        speaker_best_face: Dict[str, Optional[FaceOccurrence]] = {}
        for dia_seg in diarization:
            speaker_best_face[dia_seg.speaker_id] = self._get_best_face_for_speaker(
                dia_seg.speaker_id, speaker_faces
            )

        # Get gating probabilities
        speaker_probs: Dict[str, float] = {}
        for dia_seg in diarization:
            best_face = speaker_best_face.get(dia_seg.speaker_id)
            feats = self._extract_features(dia_seg, transcribed, best_face)
            x = torch.tensor([feats], dtype=torch.float32)
            with torch.no_grad():
                self.gating_net.eval()
                prob = self.gating_net(x).item()
            speaker_probs[dia_seg.speaker_id] = prob

        # Sort speakers by probability (highest first)
        sorted_speakers = sorted(speaker_probs.items(), key=lambda x: -x[1])

        name_idx = 0
        resolved: Dict[str, Tuple[str, float]] = {}
        used_names = set()

        # First pass: high-confidence registered faces (prob > 0.5 or high face_conf)
        for spk_id, prob in sorted_speakers:
            best_face = speaker_best_face.get(spk_id)
            face_conf = best_face.face_confidence if best_face else 0.0
            face_name = best_face.resolved_face_id if best_face else "UNKNOWN"

            # Assign registered name if either gating prob high OR face confidence high
            if face_name != "UNKNOWN" and not face_name.startswith("face_cluster_"):
                if prob > 0.5 or face_conf > config.FACE_SIM_THRESHOLD:
                    if face_name not in used_names:
                        resolved[spk_id] = (face_name, max(prob, face_conf))
                        used_names.add(face_name)

        # Second pass: ordered_names from NLP
        for spk_id, prob in sorted_speakers:
            if spk_id in resolved:
                continue
            if name_idx < len(self.ordered_names):
                name = self.ordered_names[name_idx]
                if name not in used_names:
                    resolved[spk_id] = (name, prob)
                    used_names.add(name)
                    name_idx += 1

        # Third pass: face clusters
        for spk_id, prob in sorted_speakers:
            if spk_id in resolved:
                continue
            best_face = speaker_best_face.get(spk_id)
            if best_face and best_face.resolved_face_id.startswith("face_cluster_"):
                name = best_face.resolved_face_id
                if name not in used_names:
                    resolved[spk_id] = (name, prob)
                    used_names.add(name)

        # Fourth pass: registered faces with low prob but high face_conf (fallback)
        for spk_id, prob in sorted_speakers:
            if spk_id in resolved:
                continue
            best_face = speaker_best_face.get(spk_id)
            if best_face and best_face.resolved_face_id != "UNKNOWN" and not best_face.resolved_face_id.startswith("face_cluster_"):
                face_conf = best_face.face_confidence
                if face_conf > config.FACE_SIM_THRESHOLD:
                    name = best_face.resolved_face_id
                    if name not in used_names:
                        resolved[spk_id] = (name, face_conf)
                        used_names.add(name)

        # Final pass: generic Speaker_N
        speaker_counter = 1
        for spk_id, prob in sorted_speakers:
            if spk_id not in resolved:
                while f"Speaker_{speaker_counter}" in used_names:
                    speaker_counter += 1
                name = f"Speaker_{speaker_counter}"
                resolved[spk_id] = (name, 0.5)
                used_names.add(name)
                speaker_counter += 1

        return resolved

    def create_final_segments(
        self,
        diarization: List[DiarizationSegment],
        transcribed: List[TranscribedSegment],
        resolved_names: Dict[str, Tuple[str, float]],
    ) -> List[FinalSegment]:
        """Create FinalSegment by merging transcribed text per diarization segment."""
        final_segments: List[FinalSegment] = []

        for dia_seg in diarization:
            matching_texts = []
            for tseg in transcribed:
                if tseg.speaker_id != dia_seg.speaker_id:
                    continue
                if tseg.end <= dia_seg.start or tseg.start >= dia_seg.end:
                    continue
                matching_texts.append(tseg.text)

            if not matching_texts:
                continue

            combined_text = " ".join(matching_texts).strip()
            if not combined_text:
                continue

            speaker_name, confidence = resolved_names.get(dia_seg.speaker_id, ("UNKNOWN", 0.0))

            final_segments.append(FinalSegment(
                start=dia_seg.start,
                end=dia_seg.end,
                speaker=speaker_name,
                text=combined_text,
                confidence=confidence,
            ))

        return final_segments


def run_fusion_pipeline(
    diarization: List[DiarizationSegment],
    transcribed: List[TranscribedSegment],
    faces: List[FaceOccurrence],
    ordered_names: Optional[List[str]] = None,
) -> List[FinalSegment]:
    """
    Run full fusion pipeline.

    Args:
        diarization: Speaker diarization segments.
        transcribed: Transcribed segments with speaker assignments.
        faces: Face occurrences with confidence and lip-sync.
        ordered_names: Names extracted from NLP intro (optional).

    Returns:
        List of FinalSegment with resolved speaker names.
    """
    fusion = GatingFusion(ordered_names=ordered_names)
    resolved = fusion.resolve_identities(diarization, transcribed, faces)
    final_segments = fusion.create_final_segments(diarization, transcribed, resolved)
    return final_segments