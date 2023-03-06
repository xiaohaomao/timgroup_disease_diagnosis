#!/bin/bash


PROJECT_PATH=/home/yhuang/RareDisease/RareDisease
EMBED_FOLDER=$PROJECT_PATH/embedding/SDNEEncoder
GRAPH_ADJ_PATH=$PROJECT_PATH/data/preprocess/HPO_GRAPH.adjlist
GRAPH_FORMAT=adjlist
METHOD=sdne
BATCH_SIZE=128
LR=0.0001

# LR=0.0001: [512,128]: 400; [512,256,128]: 430; [512, 256]: 330
# LR=0.001: [128,32]: loss=1587.5; [256,64]: 800; [512,128]: 500; [512, 256, 128]: 500; [512,256]: 500
#ENCODER_LIST=("[128,32]" "[256,64]" "[512,128]")
ENCODER_LIST=("[512,256,128]")
EPOCH_LIST=(400)
ALPHA_LIST=(0.000001)  # 1e-6
BETA_LIST=(5)
NU1_LIST=(0.00001)   # 1e-5
NU2_LIST=(0.0001)    # 1e-4


if [ ! -e $EMBED_FOLDER ]; then mkdir $EMBED_FOLDER; fi

for ENCODER in ${ENCODER_LIST[@]}; do
  for EPOCH in ${EPOCH_LIST[@]}; do
    for ALPHA in ${ALPHA_LIST[@]}; do
      for BETA in ${BETA_LIST[@]}; do
        for NU1 in ${NU1_LIST[@]}; do
          for NU2 in ${NU2_LIST[@]}; do
            OUTPUT_EMBED_PATH=$EMBED_FOLDER/encoder${ENCODER}_lr${LR}_epoch${EPOCH}_alpha${ALPHA}_beta${BETA}_nu1-${NU1}_nu2-${NU2}.txt
            echo "python -m openne --input $GRAPH_ADJ_PATH --graph-format $GRAPH_FORMAT --output $OUTPUT_EMBED_PATH --method $METHOD --bs $BATCH_SIZE --lr $LR --encoder-list \"${ENCODER}\" --epochs $EPOCH --alpha $ALPHA --beta $BETA --nu1 $NU1 --nu2 $NU2"
            python -m openne --input $GRAPH_ADJ_PATH --graph-format $GRAPH_FORMAT --output $OUTPUT_EMBED_PATH --method $METHOD --bs $BATCH_SIZE --lr $LR --encoder-list "${ENCODER}" --epochs $EPOCH --alpha $ALPHA --beta $BETA --nu1 $NU1 --nu2 $NU2
          done
        done
      done
    done
  done
done












