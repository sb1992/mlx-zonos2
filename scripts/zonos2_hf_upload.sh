#!/usr/bin/env bash
# =============================================================================
# zonos2-mlx -> HuggingFace upload (STAGED — RUN MANUALLY, NOT BY THE BUILD)
# =============================================================================
# RUN MANUALLY after `hf auth login` — this script is NOT executed by the build.
# It assembles the upload folder layout, then UPLOADS via the fast/resumable Xet
# path (`hf upload-large-folder`). The Python `HfApi.upload_file` >2 GB stall is
# known — the CLI large-folder path is the correct tool.
#
# Prereqs:
#   hf auth login                       # one-time, outward-facing (your creds)
#   uv pip install 'huggingface_hub[hf_xet]'   # the hf_xet extra = fast Xet xfer
#
# Usage:
#   bash scripts/zonos2_hf_upload.sh           # stage + print the upload command
#   bash scripts/zonos2_hf_upload.sh --upload  # stage + ACTUALLY upload
# =============================================================================
set -euo pipefail

REPO_ID="shraey/zonos2-mlx"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS="${ROOT}/weights"
STAGE="${ROOT}/outputs/hf_upload/zonos2-mlx"

BF16_DIR="${WEIGHTS}/zonos2-bf16"
INT8_DIR="${WEIGHTS}/zonos2-int8"
INT4_DIR="${WEIGHTS}/zonos2-int4"

# --- sanity: the three tiers + the tier-independent assets must exist ---------
for f in \
  "${BF16_DIR}/zonos2-bf16.safetensors" \
  "${BF16_DIR}/config.json" \
  "${BF16_DIR}/dac_44khz/model.safetensors" \
  "${BF16_DIR}/speaker_encoder/model.safetensors" \
  "${INT8_DIR}/zonos2-int8.safetensors" \
  "${INT8_DIR}/config.json" \
  "${INT8_DIR}/quant_config.json" \
  "${INT4_DIR}/zonos2-int4.safetensors" \
  "${INT4_DIR}/config.json" \
  "${INT4_DIR}/quant_config.json" \
; do
  [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done

# --- assemble the upload folder layout ---------------------------------------
# Symlink the big safetensors (don't duplicate ~30 GB on disk); copy the small
# json/config so the staged tree is self-describing. hf resolves symlinks.
echo "Staging upload tree at: ${STAGE}"
rm -rf "${STAGE}"
mkdir -p "${STAGE}/zonos2-bf16" "${STAGE}/zonos2-int8" "${STAGE}/zonos2-int4"

# bf16 tier (trunk + the tier-independent dac + speaker encoder live here)
ln -sf "${BF16_DIR}/zonos2-bf16.safetensors" "${STAGE}/zonos2-bf16/zonos2-bf16.safetensors"
cp -f  "${BF16_DIR}/config.json"             "${STAGE}/zonos2-bf16/config.json"
cp -Rf "${BF16_DIR}/dac_44khz"               "${STAGE}/zonos2-bf16/dac_44khz"
cp -Rf "${BF16_DIR}/speaker_encoder"         "${STAGE}/zonos2-bf16/speaker_encoder"

# int8 tier (trunk + quant recipe)
ln -sf "${INT8_DIR}/zonos2-int8.safetensors" "${STAGE}/zonos2-int8/zonos2-int8.safetensors"
cp -f  "${INT8_DIR}/config.json"             "${STAGE}/zonos2-int8/config.json"
cp -f  "${INT8_DIR}/quant_config.json"       "${STAGE}/zonos2-int8/quant_config.json"

# int4 tier (trunk + quant recipe)
ln -sf "${INT4_DIR}/zonos2-int4.safetensors" "${STAGE}/zonos2-int4/zonos2-int4.safetensors"
cp -f  "${INT4_DIR}/config.json"             "${STAGE}/zonos2-int4/config.json"
cp -f  "${INT4_DIR}/quant_config.json"       "${STAGE}/zonos2-int4/quant_config.json"

# model card = the repo README (HF renders README.md as the model card)
cp -f "${ROOT}/README.md" "${STAGE}/README.md"

echo "Staged tree:"
find "${STAGE}" -maxdepth 2 \( -type f -o -type l \) | sort | sed "s#${STAGE}#  zonos2-mlx#"

# --- the EXACT upload command (fast/resumable Xet path) ----------------------
echo
echo "===================================================================="
echo "To publish (after \`hf auth login\` + the hf_xet extra), run:"
echo
echo "  hf upload-large-folder ${REPO_ID} --repo-type model ${STAGE}"
echo
echo "===================================================================="

if [ "${1:-}" = "--upload" ]; then
  echo "Uploading…"
  hf upload-large-folder "${REPO_ID}" --repo-type model "${STAGE}"
else
  echo "(staging only — re-run with --upload to actually publish)"
fi
