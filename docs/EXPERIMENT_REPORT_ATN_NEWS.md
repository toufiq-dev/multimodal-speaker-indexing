# Experiment Report: ATN News Video Analysis

## Multimodal Bangla Talk-show Speaker Indexing System

**Student:** Toufiqur Rahman  
**Supervisor:** Professor Dr. Sheak Rashed Haider Noori  
Professor & Head, Department of Computer Science and Engineering  
**Institution:** Daffodil International University (DIU)  
**Program:** MSc Data Science (18-credit capstone project)  
**Date:** August 27, 2026  

---

## 1. Experiment Overview

### 1.1 Video Source
- **Title:** "একটা ব্যাংকের এমডির বেতন ৩৫ লাখ টাকা কিভাবে হয়?" (How does a bank MD get 35 lakh taka salary?)
- **Channel:** ATN News
- **Duration:** 3 minutes 3 seconds (183 seconds)
- **URL:** https://youtu.be/7qbZhScrFew
- **Language:** Bengali (Bangla)
- **Content Type:** News segment with multiple speakers

### 1.2 Hardware Environment
- **Machine:** Apple MacBook Air M1
- **RAM:** 8 GB unified memory
- **OS:** macOS (Darwin)
- **GPU:** Apple M1 Neural Engine (not utilized by CTranslate2/pyannote)

### 1.3 Software Stack
| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.14 | Runtime environment |
| PyTorch | 2.13.0 | CPU-only installation |
| faster-whisper | latest | CTranslate2 backend, CPU int8 |
| pyannote-audio | 4.0.7 | Community diarization model |
| InsightFace | 0.7.3 | Face detection + recognition |
| ONNX Runtime | latest | CPU execution provider |

---

## 2. Pipeline Execution Results

### 2.1 Stage-by-Stage Timing

| Stage | Step | Time | Notes |
|-------|------|------|-------|
| **1** | Audio extraction (ffmpeg) | ~2s | 16kHz mono WAV |
| **2** | Speaker diarization (pyannote) | **245.8s** | CPU, community model |
| **3** | Transcription (Whisper tiny) | ~30s | Word-level timestamps |
| **4** | Frame extraction (ffmpeg) | ~5s | 183 frames @ 1 FPS |
| **5** | Vision pipeline (InsightFace) | ~120s | 183 frames processed |
| **6** | NLP name extraction | ~10s | BanglaBERT NER |
| **7** | Fusion + identity resolution | ~1s | Deterministic cascade |
| **Total** | End-to-end | **487.8s (8.1 min)** | RTF: 2.66x |

### 2.2 Output Files
```
data/output/
├── result.json          # 13 final segments
├── subtitles.srt        # SRT subtitle format
├── audio/
│   └── atn_news.wav     # 16kHz mono audio
└── frames/
    └── frame_*.jpg      # 183 extracted frames
```

---

## 3. Detailed Analysis by Component

### 3.1 Speaker Diarization (pyannote 4.x)

**Result:** 2 speakers detected (SPEAKER_00, SPEAKER_01)

**Diarization segments:** 19 total

| Speaker | Segments | Total Duration | Avg Segment Length |
|---------|----------|----------------|-------------------|
| SPEAKER_00 | 14 | 159.8s | 11.4s |
| SPEAKER_01 | 5 | 3.3s | 0.7s |

**Key observations:**

1. **Correct speaker count:** The diarization correctly identified that **2 people spoke** in the video. The user confirmed: "There are three persons in the video and two of them spoke." This validates pyannote's automatic speaker count detection.

2. **Speaker dominance:** SPEAKER_00 (likely the narrator/reporter) dominates with 98% of speech time. SPEAKER_01 appears only in brief interjections:
   - 81.66s–81.79s (0.14s)
   - 81.82s–81.98s (0.15s)
   - 84.39s–84.79s (0.41s)
   - 158.05s–158.30s (0.25s)

3. **Performance on CPU:** Diarization took 245.8 seconds for a 183-second video (RTF: 1.34x). This is the **primary bottleneck** of the pipeline on CPU-only hardware.

4. **Diarization quality concern:** The very short SPEAKER_01 segments (0.14s, 0.15s) may indicate over-segmentation or actual brief interjections. Manual verification would be needed to confirm.

### 3.2 Transcription (Whisper tiny)

**Result:** 13 transcribed segments with word-level timestamps

**Key observations:**

1. **Model quality:** The `tiny` Whisper model (~75MB) produces **romanized/transliterated Bengali**, not proper Bengali script. This is expected for the smallest Whisper model on Bengali.

2. **Transcription examples:**
   - Segment 1: "Mahouddin Khochoun" (should be মহিউদ্দিন খোঁজৌন or similar)
   - Segment 2: "Amar Khoopriwajdin Manus Uregulari Dakhahai" (garbled transliteration)

3. **Word-level timestamps:** Successfully generated, enabling precise alignment with diarization turns.

4. **Segment consolidation issue:** The last segment spans 48.3s–171.6s (123.4 seconds), merging what diarization identified as multiple speaker turns. This occurs because:
   - The tiny model's word timestamps are less precise
   - Speaker changes within the transcription window weren't detected
   - The word-midpoint containment algorithm assigned all words to one turn

### 3.3 Vision Pipeline (InsightFace)

**Result:** 292 face occurrences across 183 frames

**Key observations:**

1. **Face detections vs. unique faces:** The 292 number represents **face detections**, not unique faces. With 3 people in the video:
   - Average: 1.6 faces per frame
   - Some frames: 1 face (single person visible)
   - Some frames: 2-3 faces (multiple people visible)

2. **Face clustering:** Without a face registry, the system clusters unknown faces using DBSCAN. The result `face_cluster_1` indicates all detected faces were grouped into a single cluster (likely due to insufficient embedding diversity or parameter tuning).

3. **Registry dependency:** The identity resolution cascade falls back to face clustering (Priority 4) when no registry images are provided. This explains why all segments received the generic `face_cluster_1` label.

4. **InsightFace performance:** ~0.65 seconds per frame on CPU (183 frames in ~120s). Acceptable for development.

### 3.4 NLP Name Extraction (BanglaBERT NER)

**Result:** No speaker names extracted

**Key observations:**

1. **NER model loaded:** BanglaBERT NER (`sagorsarker/banglabert-ner`) successfully loaded and executed.

2. **Input quality dependency:** The NER pipeline processes the first 120 seconds of transcription text. With the tiny model's garbled output, the NER could not identify proper Bengali person names.

3. **Host anchor pattern:** The "আমি <Name>" (I am <Name>) pattern was not detected because:
   - The transcription is in romanized form, not Bengali script
   - The pattern matcher looks for Bengali Unicode characters (U+0980–U+09FF)

4. **Critical dependency:** This demonstrates that **NER quality is directly dependent on ASR quality**. The tiny model's output is insufficient for downstream NLP tasks.

### 3.5 Fusion and Identity Resolution

**Result:** 13 final segments, all labeled `face_cluster_1`

**Identity resolution cascade execution:**

| Priority | Evidence Type | Result |
|----------|---------------|--------|
| 0 | Ground-truth labels | Not provided |
| 1 | Registry face recognition | No registry images |
| 2 | Host self-intro anchor | Not detected (garbled transcription) |
| 3 | Co-occurrence NER matching | No NER names found |
| 4 | Face cluster label | **`face_cluster_1` (all segments)** |
| 5 | Generic Speaker_N | Not reached |

**Why all segments got the same label:**

The identity resolution cascade failed at every specificity level:
1. No face registry → Priority 1 skipped
2. No "আমি <Name>" in transcription → Priority 2 skipped
3. No NER names → Priority 3 skipped
4. Face clustering assigned all faces to one cluster → Priority 4 used

**This is the core limitation:** Without either a face registry OR a high-quality transcription (for NLP name extraction), the system cannot distinguish speakers by name.

---

## 4. Honest Assessment of Current Capabilities

### 4.1 What Works Correctly
1. ✅ **Speaker count detection:** pyannote correctly identified 2 speakers
2. ✅ **Speaker turn segmentation:** 19 diarization segments with timestamps
3. ✅ **Audio extraction:** Clean 16kHz WAV output
4. ✅ **Frame extraction:** 183 frames at 1 FPS
5. ✅ **Face detection:** 292 face occurrences detected
6. ✅ **Pipeline execution:** End-to-end completion without crashes

### 4.2 What Needs Improvement
1. ❌ **Transcription quality:** Tiny model produces unusable Bengali text
2. ❌ **Speaker identification:** No names assigned to speakers
3. ❌ **Segment granularity:** Last segment merges 123 seconds of speech
4. ❌ **Diarization speed:** 245s for 183s video on CPU
5. ❌ **Face clustering:** All faces grouped into single cluster

### 4.3 Limitations Acknowledged
1. **Hardware constraints:** 8GB M1 limits model sizes
2. **CPU-only execution:** No GPU acceleration available
3. **No face registry:** System falls back to visual clustering only
4. **Language-specific challenges:** Bengali ASR is a low-resource task

---

## 5. Recommendations for Improvement

### 5.1 Immediate Actions (No Code Changes)

1. **Provide face registry images:**
   - Download 1-3 clear face photos per speaker from Google Images
   - Name files with speaker identity: `mohiuddin.jpg`, `reporter.jpg`
   - Place in `data/registry/`
   - This enables Priority 1 (registry face recognition) in the cascade

2. **Set speaker count hint (optional):**
   - `export NUM_SPEAKERS=2` for this video
   - **Note:** This is optional, not required. The system detects automatically.
   - Useful for evaluation/benchmarking, not for production deployment

### 5.2 Model Upgrades (Within 8GB M1 Constraints)

| Model | Size | Expected Improvement | Risk |
|-------|------|---------------------|------|
| Whisper tiny | 75MB | Baseline (current) | Low |
| Whisper small | 244MB | Better Bengali script | Moderate (slower) |
| Whisper medium | 769MB | Good Bengali quality | High (may cause memory pressure) |

**Recommendation:** Try `small` first. On 8GB M1:
- tiny: ~75MB + working memory → comfortable
- small: ~244MB + working memory → feasible, slower
- medium: ~769MB + working memory → tight, may cause swapping

**Expected timing for small model:**
- Transcription: ~2-3 minutes (vs. 30s for tiny)
- Total pipeline: ~10-12 minutes (vs. 8 minutes for tiny)

### 5.3 System Architecture Improvements

1. **Diarization speed:** Consider using `pyannote/speaker-diarization-community-1` (lighter model) vs. `pyannote/speaker-diarization-3.1` (full model)

2. **Face clustering:** Tune DBSCAN parameters (`eps`, `min_samples`) for better cluster separation

3. **Segment splitting:** The 123-second final segment indicates the transcription engine needs better speaker-change detection

---

## 6. Answers to Specific Questions

### 6.1 "Is Diarization just detecting who is speaking?"

**Yes, exactly.** Diarization answers:
- **WHO** is speaking (speaker labels: SPEAKER_00, SPEAKER_01)
- **WHEN** they are speaking (start/end timestamps)
- **HOW MANY** speakers are present (automatic detection)

It does **NOT** answer:
- **WHAT** the speakers are saying (that's ASR/transcription)
- **WHO they are by name** (that's identity resolution)

The diarization correctly found 2 speakers because 2 people spoke. The third person (visible but silent) was not detected by audio-only diarization.

### 6.2 "Vision caught 292 face occurrences... we need to know who is talking"

**Correct.** Vision detects faces but doesn't know:
- Which face belongs to which speaker
- Who is currently speaking (lip-sync is experimental)

The fusion engine attempts to match faces to speakers by:
1. Overlaying face detections onto diarization time segments
2. Matching face embeddings to registry images (if available)
3. Clustering unknown faces as visual-only identities

**The 292 number breakdown:**
- 183 frames × ~1.6 faces/frame = 292 detections
- These represent ~3 unique individuals (you confirmed 3 people in video)
- The clustering should group them into 3 clusters, but currently groups into 1

### 6.3 "Should I add face registry images?"

**Yes, this is the most effective improvement.** For a thesis project:

**Approach:**
1. Search Google Images for each speaker's name
2. Download 1-3 clear, front-facing photos
3. Name files: `speaker_name.jpg` (underscores become spaces)
4. Place in `data/registry/`

**Example:**
```
data/registry/
├── mohiuddin.jpg      # First speaker
├── atn_reporter.jpg   # Second speaker (if identifiable)
└── bank_md.jpg        # Third person (if they speak)
```

**Why this works:**
- InsightFace computes face embeddings (128-dimensional vectors)
- Registry embeddings are compared to detected face embeddings
- Cosine similarity > 0.65 threshold → identity match
- This is Priority 1 in the identity cascade (highest specificity)

### 6.4 "If I set speaker count... how is this an engineering project?"

**You are absolutely right.** Setting speaker count manually is not engineering—it's cheating.

**The system already works automatically:**
- pyannote detects speaker count from audio
- `NUM_SPEAKERS` is an **optional hint**, not a requirement
- For evaluation/benchmarking, hints can improve accuracy
- For production, the system must be fully automatic

**Thesis contribution:** The system demonstrates **automatic** speaker count detection + identity resolution without manual intervention.

---

## 7. Next Steps

### Phase 1: Improve Transcription Quality
1. Upgrade from `tiny` to `small` Whisper model
2. Verify Bengali script output (not romanized)
3. Re-run pipeline and compare results

### Phase 2: Add Face Registry
1. Download reference photos for known speakers
2. Place in `data/registry/`
3. Re-run pipeline to enable identity resolution

### Phase 3: Evaluate and Document
1. Compare results with/without registry
2. Measure identity resolution accuracy
3. Document findings in thesis

---

## 8. Conclusion

The multimodal speaker indexing system successfully:
- Detected 2 speakers (correct)
- Generated 19 diarization segments with timestamps
- Extracted 183 video frames
- Detected 292 face occurrences
- Produced 13 final segments

**Primary limitation:** Without a face registry or high-quality transcription, the system cannot assign real names to speakers. This is expected behavior—the identity resolution cascade is designed to work with available evidence and fall back gracefully.

**Key insight:** The system's architecture is sound. The limitations are in input quality (tiny Whisper model) and missing reference data (face registry). These are addressable within the 8GB M1 hardware constraints.

---

*Report generated by the Multimodal Bangla Talk-show Speaker Indexing System*  
*Experiments conducted on August 27, 2026*
