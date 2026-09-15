#!/data/data/com.termux/files/usr/bin/bash
# Cài llama.cpp + python deps, và đảm bảo model gguf có mặt tại models/.
# Idempotent: chạy lại nhiều lần không tải lại model nếu checksum đã khớp.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$REPO_DIR/models"
MODEL_FILE="$MODEL_DIR/qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
EXPECTED_SHA256="6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"

echo "== pkg install llama-cpp =="
pkg install -y llama-cpp

echo "== pip install requests beautifulsoup4 =="
pip install --quiet requests beautifulsoup4

mkdir -p "$MODEL_DIR"

model_checksum_ok() {
    [ -f "$MODEL_FILE" ] || return 1
    local actual
    actual="$(sha256sum "$MODEL_FILE" | cut -d' ' -f1)"
    [ "$actual" = "$EXPECTED_SHA256" ]
}

if model_checksum_ok; then
    echo "== Model đã có và checksum khớp, bỏ qua tải lại: $MODEL_FILE =="
else
    echo "== Tải model (~1.1GB) về $MODEL_FILE =="
    curl -L --fail -o "$MODEL_FILE.part" "$MODEL_URL"
    mv "$MODEL_FILE.part" "$MODEL_FILE"
    if ! model_checksum_ok; then
        echo "Checksum không khớp sau khi tải, kiểm tra lại đường truyền hoặc URL." >&2
        exit 1
    fi
    echo "== Tải xong, checksum khớp =="
fi
