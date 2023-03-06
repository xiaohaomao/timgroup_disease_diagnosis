#!/bin/bash

#VECTOR_SIZE=256
#X_MAX=20
#MAX_ITER=200
NUM_THREADS=12
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=1
WINDOW_SIZE=1600
BINARY=0

PROJECT_PATH=/home/yhuang/RareDisease/RareDisease
BUILDDIR=$PROJECT_PATH/nodeEmbed/GloVe-1.2/build
CORPUS=$PROJECT_PATH/data/preprocess/GloveTextAncestor   # GloveTextAncestorDup | GloveTextAncestor
GLOVE_FOLDER=$PROJECT_PATH/embedding/GloveEncoder/Ancestor   # PHELIST_ANCESTOR_DUP | PHELIST_ANCESTOR

VOCAB_FILE=$GLOVE_FOLDER/vocab.txt
COOCCURRENCE_FILE=$GLOVE_FOLDER/cooccurrence.bin

$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
if [[ $? -ne 0 ]]; then exit 1; fi
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
if [[ $? -ne 0 ]]; then exit 2; fi

vecSizeList=(32 64 128 256 512)
xMaxList=(10 50 100 500)
maxIterList=(200)
for VECTOR_SIZE in ${vecSizeList[@]}; do
  for X_MAX in ${xMaxList[@]}; do
    for MAX_ITER in ${maxIterList[@]}; do
      SAVING_FOLDER=$GLOVE_FOLDER/GloveEncoder_vec${VECTOR_SIZE}_xMax${X_MAX}_maxIter${MAX_ITER}
      COOCCURRENCE_SHUF_FILE=$SAVING_FOLDER/cooccurrence.shuf.bin
      VECTOR_FILE=$SAVING_FOLDER/vectors
      LOG_FILE=$SAVING_FOLDER/log
      if [ ! -e $SAVING_FOLDER ]; then mkdir $SAVING_FOLDER; fi

      $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
      if [[ $? -ne 0 ]]; then exit 3; fi
      echo $CORPUS
      echo $SAVING_FOLDER
      $BUILDDIR/glove -save-file $VECTOR_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE &> $LOG_FILE
      rm  $COOCCURRENCE_SHUF_FILE
    done
  done
done






