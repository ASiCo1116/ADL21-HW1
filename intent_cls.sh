python3.8 test_intent.py --test_file "${1}" \
    --cache_dir cache/intent/ \
    --ckpt_path best_intent.ckpt \
    --pred_file "${2}" \
    --batch_size 64 \
    --hidden_size 256 \
    --num_layers 5 \
    --device 'cuda'