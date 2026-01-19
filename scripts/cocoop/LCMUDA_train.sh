#!/bin/bash

cd /data1/ljs/LCMUDA

# custom config
DATA=/data1/ljs/LCMUDA/DATA
TRAINER=LCMUDA


DATASET=officehome
SEED=$1

CFG=vit_b16_c4_ep10_batch1_ctxv1
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=100


DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --source-domains art clipart product \
    --target-domains real_world \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi

