#!/bin/bash

# Runs the StarCoderBase model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export AIXDEBUG=1

CHECKPOINT_PATH=/data3/StarCoderBase/CodeGPT-train.pt
VOCAB_FILE="" #<Specify path to file>/gpt2-vocab.json
MERGE_FILE="" #<Specify path to file>/gpt2-merges.txt
DATA_PATH="" #<Specify path and file prefix>_text_document

GPT_ARGS="
    --num-layers 40 \
    --hidden-size 6144 \
    --num-attention-heads 48 \
    --seq-length 1024 \
    --max-position-embeddings 8192 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --bf16 \
    --multi-query \
    --vocab-size=49152 \
    --no-strict-load \
    --hidden-dropout=0 \
    --attention-dropout=0 \
"

DATA_ARGS="
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

cmd="torchrun pretrain_starcoder.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH.new \
    --load $CHECKPOINT_PATH"

echo $cmd

