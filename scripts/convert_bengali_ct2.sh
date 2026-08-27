#!/bin/bash
# Convert bengaliAI Whisper-medium to CTranslate2 (int8) for faster-whisper word_timestamps=True
# Run on Kaggle T4×2 or local: bash scripts/convert_bengali_ct2.sh
set -e
MODEL="bengaliAI/tugstugi_bengaliai-asr_whisper-medium"
OUT="models/bengaliAI_ct2"
echo "Converting $MODEL → $OUT (int8, word timestamps preserved)..."
~/.venv/bin/ct2-transformers-converter --model "$MODEL" --output_dir "$OUT" --quantization int8 --force
echo "✓ Converted: $OUT"
ls -lh "$OUT" | head -n 20
# Test load
python -c "from faster_whisper import WhisperModel; m=WhisperModel('$OUT', device='cpu', compute_type='int8'); print('✓ Load OK', m)"
