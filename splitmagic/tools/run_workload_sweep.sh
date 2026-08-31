#!/bin/bash
set -e

MODEL="$1"

if [ "$MODEL" = "tinystories" ]; then

    BATCH_SIZES=(1)
    SEQ_LENGTHS=(32 64 128)

    for BS in "${BATCH_SIZES[@]}"; do
        for SEQ in "${SEQ_LENGTHS[@]}"; do

            python tools/profile_model_operator_workload.py \
                --model tinystories \
                --batch-size "$BS" \
                --sequence-length "$SEQ"

        done
    done

elif [ "$MODEL" = "resnet18" ]; then

    BATCH_SIZES=(32)
    IMAGE_SIZES=(32 64 224)

    for BS in "${BATCH_SIZES[@]}"; do
        for IMG in "${IMAGE_SIZES[@]}"; do

            python tools/profile_model_operator_workload.py \
                --model resnet18 \
                --batch-size "$BS" \
                --image-size "$IMG" \
                --num-classes 200

        done
    done
fi