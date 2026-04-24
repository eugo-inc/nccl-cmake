#!/usr/bin/env bash
# validate_nccl_conf.sh — lint a nccl.conf file for the eugo-inc/nccl-cmake loader (v2.29.3).
#
# Loader reference: src/misc/param.cc:setEnvFile. Grammar:
#   line := '#' comment  |  KEY=VALUE  |  empty
#   KEY does NOT tolerate leading/trailing whitespace (strncpy, no trim).
#   VALUE does NOT tolerate leading whitespace after '=' (same reason).
#   key and value each truncated at 1023 bytes (silent).
#   malformed lines silently skipped.
#
# This script rejects conf lines that the loader would silently mis-parse, and warns on truncation
# and known-deprecated var names. Exits non-zero on any rejection.

set -uo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <nccl.conf> [<nccl.conf> ...]" >&2
  exit 2
fi

DEPRECATED='NCCL_IB_CUDA_SUPPORT|NCCL_IB_GDR_LEVEL|NCCL_CHECKS_DISABLE|NCCL_LL_THRESHOLD|NCCL_TREE_THRESHOLD|NCCL_SINGLE_RING_THRESHOLD|NCCL_MAX_NRINGS|NCCL_MIN_NRINGS|NCCL_NVML_DIRECT|NCCL_USE_CMAKE'

rc=0
for f in "$@"; do
  if [ ! -r "$f" ]; then
    echo "ERROR: cannot read $f" >&2; rc=1; continue
  fi
  echo "--- checking $f ---"
  lineno=0
  while IFS= read -r line || [ -n "$line" ]; do
    lineno=$((lineno + 1))
    # Drop trailing CR from CRLF files.
    line="${line%$'\r'}"
    # Empty line: OK.
    [ -z "$line" ] && continue
    # Comment: OK (only recognized at line start per loader).
    [[ "$line" == \#* ]] && continue
    # Must contain '='.
    if [[ "$line" != *=* ]]; then
      echo "REJECT $f:$lineno: no '=' — loader silently skips: $line" >&2; rc=1; continue
    fi
    key="${line%%=*}"
    val="${line#*=}"
    # Key: must be NCCL_* with no whitespace.
    if [[ ! "$key" =~ ^NCCL_[A-Z0-9_]+$ ]]; then
      echo "REJECT $f:$lineno: key '$key' invalid (must match ^NCCL_[A-Z0-9_]+$; no leading/trailing whitespace)" >&2; rc=1; continue
    fi
    # Value: reject leading whitespace (loader doesn't trim).
    if [[ "$val" =~ ^[[:space:]] ]]; then
      echo "REJECT $f:$lineno: value for $key has leading whitespace (loader preserves it literally)" >&2; rc=1; continue
    fi
    # Truncation warning: 1023-byte cap on key AND value.
    if [ ${#key} -gt 1023 ]; then
      echo "WARN   $f:$lineno: key length ${#key} > 1023 — will be truncated" >&2
    fi
    if [ ${#val} -gt 1023 ]; then
      echo "WARN   $f:$lineno: value length ${#val} > 1023 — will be truncated" >&2
    fi
    # Deprecated warning.
    if [[ "$key" =~ ^($DEPRECATED)$ ]]; then
      echo "WARN   $f:$lineno: $key is deprecated/removed — will have no effect" >&2
    fi
  done < "$f"
done

exit "$rc"
