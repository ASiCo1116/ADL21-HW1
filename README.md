# Homework 1 ADL NTU 110-2 Spring

## Environment

```shell
# If you have pipenv, you can build the environemt from pipfile
pipenv install --skip-lock

# Otherwise
pip install -r requirements.txt
```

## Preprocessing

```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent classification

#### Training

```shell
# intent.sh
python3 train_intent.py \
    --data_dir ROOT_TO_INTENT_DATA \
    --cache_dir ROOT_TO_INTENT_CACHE \
    --ckpt_dir ROOT_TO_SAVE_CHECKPOINT\
    --lr LR \
    --batch_size BS \
    --hidden_size HS \
    --num_layers NL \
    --num_epoch NE \
    --device DEVICE
```

#### Testing

```python
# intent_cls.sh
python3.8 test_intent.py --test_file ROOT_TO_TEST_FILE \
    --cache_dir ROOT_TO_INTENT_CACHE \
    --ckpt_path ROOT_TO_CHECKPOINT \
    --pred_file ROOT_TO_SAVE_PREDICTION \
    --batch_size BS \
    --hidden_size HS \
    --num_layers NL \
    --device DEVICE
```

## Slot tagging

#### Training

```shell
# slot.sh
python3 train_slot.py \
    --data_dir ROOT_TO_SLOT_DATA \
    --cache_dir ROOT_TO_SLOT_CACHE \
    --ckpt_dir ROOT_TO_SAVE_CHECKPOINT\
    --lr LR \
    --batch_size BS \
    --hidden_size HS \
    --num_layers NL \
    --num_epoch NE \
    --max_len ML \
    --device DEVICE
```

#### Testing

```python
# slot_tag.sh
python3.8 test_slot.py --test_file ROOT_TO_TEST_FILE \
    --cache_dir ROOT_TO_SLOT_CACHE \
    --ckpt_path ROOT_TO_CHECKPOINT \
    --pred_file ROOT_TO_SAVE_PREDICTION \
    --batch_size BS \
    --hidden_size HS \
    --max_len ML \
    --num_layers NL \
    --device DEVICE
```
