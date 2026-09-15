#!/data/data/com.termux/files/usr/bin/bash
# Cài llama.cpp + python deps, và đảm bảo model gguf có mặt tại models/.
# Idempotent: chạy lại nhiều lần không tải lại model nếu checksum đã khớp.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$REPO_DIR/models"

echo "== pkg install llama-cpp =="
pkg install -y llama-cpp

echo "== pip install requests beautifulsoup4 =="
pip install --quiet requests beautifulsoup4

mkdir -p "$MODEL_DIR"

# Đảm bảo 1 file model tồn tại và đúng checksum, chỉ tải lại khi thiếu/hỏng.
ensure_model() {
    local file="$1" url="$2" expected_sha256="$3"

    checksum_ok() {
        [ -f "$file" ] || return 1
        [ "$(sha256sum "$file" | cut -d' ' -f1)" = "$expected_sha256" ]
    }

    if checksum_ok; then
        echo "== Model đã có và checksum khớp, bỏ qua tải lại: $file =="
        return
    fi

    echo "== Tải model về $file =="
    curl -L --fail -o "$file.part" "$url"
    mv "$file.part" "$file"
    if ! checksum_ok; then
        echo "Checksum không khớp sau khi tải, kiểm tra lại đường truyền hoặc URL." >&2
        exit 1
    fi
    echo "== Tải xong, checksum khớp =="
}

ensure_model \
    "$MODEL_DIR/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
    "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"

ensure_model \
    "$MODEL_DIR/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
    "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
