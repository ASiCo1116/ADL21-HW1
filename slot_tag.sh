python3.8 test_slot.py --test_file "${1}" \
    --cache_dir cache/slot/ \
    --ckpt_path best_slot.ckpt \
    --pred_file "${2}" \
    --batch_size 32 \
    --hidden_size 512 \
    --max_len 64 \
    --num_layers 6 \
    --device 'cuda'