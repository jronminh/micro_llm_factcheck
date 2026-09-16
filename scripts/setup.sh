#!/data/data/com.termux/files/usr/bin/bash
# Cài llama.cpp + python deps, và đảm bảo model gguf có mặt tại models/.
# Idempotent: chạy lại nhiều lần không tải lại model nếu checksum đã khớp.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$REPO_DIR/models"

echo "== pkg install llama-cpp =="
pkg install -y llama-cpp

echo "== pip install requests beautifulsoup4 nltk =="
pip install --quiet requests beautifulsoup4 nltk

# nltk: chỉ dùng để POS-tag câu hỏi tiếng Anh trong decompose.py (bắt nhãn
# JJS/RBS/WDT/WP - so sánh nhất và mệnh đề quan hệ - việc liệt kê từ khóa
# không làm được vì tiếng Anh biến hình "-est" theo từng từ). Không dùng
# pyvi/underthesea cho tiếng Việt: kéo theo build scipy/cmake từ source, quá
# nặng cho Termux/Android - tiếng Việt là ngôn ngữ phân tích tính, so sánh
# nhất luôn dùng tiểu từ rời "nhất" nên chỉ cần từ khóa, không cần NLP thật.
echo "== tải nltk data (POS tagger + tokenizer tiếng Anh) =="
python3 -c "import nltk; nltk.download('averaged_perceptron_tagger_eng', quiet=True); nltk.download('punkt_tab', quiet=True)"

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

# Khảo sát model nhỏ khác ngoài Qwen2.5 (xem notes/micro-llm-alternatives-to-qwen.md).
# Không có Q4_K_M chính chủ trên Qwen/ggml-org, dùng bản re-quant của
# bartowski/unsloth (2 nguồn quant uy tín, phổ biến trong cộng đồng llama.cpp).
ensure_model \
    "$MODEL_DIR/qwen3-0.6b-q4_k_m.gguf" \
    "https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B-Q4_K_M.gguf" \
    "9acfc1e001311f34b4252001b626f2e466d592a42065f66571bff3790d4e1b14"

ensure_model \
    "$MODEL_DIR/gemma-3-270m-it-q4_k_m.gguf" \
    "https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_K_M.gguf" \
    "b1baabd6b729e4041822220d3e648e00d99cac5df86b10dffb77bcccf0688e39"
