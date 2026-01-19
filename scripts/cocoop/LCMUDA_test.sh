#!/bin/bash

cd /data1/ljs/CoOp

# custom config
DATA=/data1/ljs/CoOp/DATA
TRAINER=LCMUDA
# TRAINER=CoOp

DATASET=officehome
SEED=3

CFG=vit_b16_c4_ep10_batch1_ctxv1
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=100


#DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
DIR=output/evaluation/LCMUDA没跑完/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/officehome/LCMUDA没跑完/${CFG}_${SHOTS}shots/seed${SEED} \
    --source-domains art clipart product  \
    --target-domains real_world \
    --eval-only
fi
#--load-epoch 10 \
#--model-dir output/officehome/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
