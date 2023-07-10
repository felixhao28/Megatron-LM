#!/bin/bash

# Runs the StarCoderBase model

export CUDA_DEVICE_MAX_CONNECTIONS=1
# export AIXDEBUG=1
export CUDA_VISIBLE_DEVCIES="0,1,2,3,4,5,6,7"

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/data3/StarCoderBase/CodeGPT-train_mp
VOCAB_FILE="" #<Specify path to file>/gpt2-vocab.json
MERGE_FILE="" #<Specify path to file>/gpt2-merges.txt
DATA_PATH="" #<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
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
    --tokenizer-type=HuggingFaceTokenizer \
    --tokenizer-model=/data3/StarCoderBase\
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

cmd="torchrun $DISTRIBUTED_ARGS pretrain_starcoder.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH.new \
    --load $CHECKPOINT_PATH"

echo $cmd

$cmd
