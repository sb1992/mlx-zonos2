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

# --- assemble the upload folder layout (each tier is SELF-CONTAINED) ----------
# Clean tier folders `bf16/ int8/ int4/`, each runnable on its own. The big
# trunk safetensors are symlinked (don't duplicate ~27 GB locally; hf + Xet
# resolve symlinks). The small DAC codec + ECAPA speaker encoder (~315 MB, only
# present in the bf16 source dir) are copied into EVERY tier so a single-tier
# download is self-sufficient; Xet content-dedups the identical copies.
echo "Staging upload tree at: ${STAGE}"
rm -rf "${STAGE}"

stage_tier () {  # $1=tier (bf16/int8/int4)  $2=source dir  $3=trunk filename
  local tier="$1" src="$2" trunk="$3" dst="${STAGE}/$1"
  mkdir -p "${dst}"
  ln -sf "${src}/${trunk}"    "${dst}/${trunk}"
  cp -f  "${src}/config.json" "${dst}/config.json"
  [ -e "${src}/quant_config.json" ] && cp -f "${src}/quant_config.json" "${dst}/quant_config.json"
  cp -Rf "${BF16_DIR}/dac_44khz"       "${dst}/dac_44khz"
  cp -Rf "${BF16_DIR}/speaker_encoder" "${dst}/speaker_encoder"
}

stage_tier bf16 "${BF16_DIR}" zonos2-bf16.safetensors
stage_tier int8 "${INT8_DIR}" zonos2-int8.safetensors
stage_tier int4 "${INT4_DIR}" zonos2-int4.safetensors

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
