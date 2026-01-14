python test_offline_profile_sglang.py \
    --model-path Qwen/Qwen3-30B-A3B-FP8 \
    --tp-size 2 \
    --max-len 10000 \
    --max-tokens 1024 \
    --batch-size 64 \
    --prompt-len 8192 \
    --data-parallel-rank 1