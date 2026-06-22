#!/usr/bin/env bash
# Convert a merged HF model -> GGUF and quantize it for llama.cpp (CPU serving).
#
# Pipeline:
#   merged fp16 HF dir  --convert_hf_to_gguf.py-->  model-f16.gguf
#   model-f16.gguf      --llama-quantize-------->   model-<TYPE>.gguf  (per type)
#
# Then serve any of them with llama.cpp's OpenAI-compatible server and score
# with the usual OpenAI backend (see README "Serving matrix"):
#   ./llama.cpp/build/bin/llama-server -m out/gguf/model-Q4_K_M.gguf --port 8000 -c 4096
#
# Usage:
#   bash scripts/export_gguf.sh <merged_dir> [out_dir] [quant types...]
#   bash scripts/export_gguf.sh out/merged-fp16 out/gguf Q4_K_M Q5_K_M Q8_0
#
# llama.cpp location is auto-detected/cloned. Override with LLAMA_CPP=/path.
set -euo pipefail

MERGED_DIR="${1:?usage: export_gguf.sh <merged_dir> [out_dir] [quant types...]}"
OUT_DIR="${2:-out/gguf}"
shift || true
shift || true
QUANTS=("$@")
if [ "${#QUANTS[@]}" -eq 0 ]; then
  QUANTS=(Q4_K_M Q5_K_M Q8_0)
fi

LLAMA_CPP="${LLAMA_CPP:-llama.cpp}"

# --- ensure llama.cpp is present and built --------------------------------- #
if [ ! -d "$LLAMA_CPP" ]; then
  echo "Cloning llama.cpp -> $LLAMA_CPP"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"
fi

QUANT_BIN=""
for cand in "$LLAMA_CPP/build/bin/llama-quantize" "$LLAMA_CPP/llama-quantize"; do
  [ -x "$cand" ] && QUANT_BIN="$cand" && break
done
if [ -z "$QUANT_BIN" ]; then
  echo "Building llama.cpp (llama-quantize)..."
  cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$LLAMA_CPP/build" --target llama-quantize llama-server -j >/dev/null
  QUANT_BIN="$LLAMA_CPP/build/bin/llama-quantize"
fi

# The HF->GGUF converter needs a few python deps; install on demand.
python -c "import gguf" 2>/dev/null || \
  pip install --quiet -r "$LLAMA_CPP/requirements.txt"

# --- convert to f16 GGUF --------------------------------------------------- #
mkdir -p "$OUT_DIR"
F16="$OUT_DIR/model-f16.gguf"
if [ ! -f "$F16" ]; then
  echo "Converting $MERGED_DIR -> $F16"
  python "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGED_DIR" --outfile "$F16" --outtype f16
fi

# --- quantize each requested type ------------------------------------------ #
for q in "${QUANTS[@]}"; do
  outfile="$OUT_DIR/model-${q}.gguf"
  echo "Quantizing -> $outfile ($q)"
  "$QUANT_BIN" "$F16" "$outfile" "$q"
done

echo
echo "Done. GGUF artifacts (size on disk):"
du -h "$OUT_DIR"/*.gguf | sed 's/^/  /'
echo
echo "Serve one, e.g.:"
echo "  $LLAMA_CPP/build/bin/llama-server -m $OUT_DIR/model-${QUANTS[0]}.gguf --port 8000 -c 4096"
