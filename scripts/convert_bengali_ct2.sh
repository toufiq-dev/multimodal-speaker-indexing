#!/bin/bash
# Convert the bengaliAI Whisper-medium checkpoint to CTranslate2 for
# faster-whisper (word_timestamps=True).
#
#   bash scripts/convert_bengali_ct2.sh [output_dir]
#
# faster-whisper cannot consume a Transformers checkpoint directly: it needs a
# CT2 directory (model.bin + CT2 config.json + vocabulary.json). Point
# WHISPER_MODEL at $OUT afterwards.
set -euo pipefail

MODEL="${WHISPER_MODEL_SRC:-bengaliAI/tugstugi_bengaliai-asr_whisper-medium}"
OUT="${1:-models/bengaliAI_ct2}"
QUANT="${CT2_QUANTIZATION:-int8}"

# Invoke through the active interpreter. The previous hardcoded
# ~/.venv/bin/ct2-transformers-converter does not exist on Kaggle or Colab and
# aborted the script under `set -e`.
echo "Converting $MODEL → $OUT (quantization=$QUANT)..."
python -m ctranslate2.converters.transformers \
    --model "$MODEL" \
    --output_dir "$OUT" \
    --quantization "$QUANT" \
    --copy_files tokenizer.json preprocessor_config.json \
    --force

echo "✓ Converted: $OUT"
ls -lh "$OUT"

# Verify it loads AND is multilingual: an English-only conversion does not make
# faster-whisper reject language="bn", it silently decodes Bangla as English.
python - "$OUT" <<'PY'
import sys
from faster_whisper import WhisperModel
out = sys.argv[1]
m = WhisperModel(out, device="cpu", compute_type="int8")
assert m.model.is_multilingual, (
    f"{out} converted as English-only — language='bn' would be silently "
    f"downgraded to 'en'. Re-convert with the multilingual tokenizer.")
print(f"✓ Load OK, multilingual=True → set WHISPER_MODEL={out}")
PY
