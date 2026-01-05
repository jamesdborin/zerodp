#!/usr/bin/env bash

set -euo pipefail

HOST="${1:-http://127.0.0.1:8000}"

start_profile() {
  curl -sS -X POST "${HOST}/start_profile" >/dev/null
}

stop_profile() {
  curl -sS -X POST "${HOST}/stop_profile" >/dev/null
}

send_chat_request() {
  curl -sS -X POST "${HOST}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${OPENAI_API_KEY:-dummy_key}" \
    -d '{
      "model": "jamesdborin/Qwen3-30B-A3B-4layers",
      "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one short sentence."}
      ],
      "max_tokens": 32
    }'
}

start_profile
trap stop_profile EXIT

send_chat_request
