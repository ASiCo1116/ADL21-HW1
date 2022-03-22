python3 train_intent.py \
    --data_dir ~/data/ADL/hw1/intent/ \
    --cache_dir ~/data/ADL/hw1/cache/intent/ \
    --ckpt_dir "./ckpt/intent/hidden512_layer5"\
    --lr 1e-3 \
    --batch_size 64 \
    --hidden_size 512 \
    --num_layers 5 \
    --num_epoch 200 \
    --device "cuda:1"