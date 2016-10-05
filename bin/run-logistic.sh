#!/bin/bash

export SPARK_HOME=/root/spark 
export KEYSTONE_MEM=80g 

DATE=`date +%Y%m%d.%H%M%S`
NUM_SLAVES=`wc -l $SPARK_HOME/conf/slaves | cut -d" " -f1`
NUM_PARTS=$((8 * $NUM_SLAVES))
NUM_PARTS=256
NUM_ITER=20
LAMBDA=0.0
echo $NUM_PARTS
echo

bin/run-pipeline.sh pipelines.video.youtube8m.Youtube8MVideoRandomFeatures \
    --trainLocation /youtube8m/video/train/ \
    --testLocation /youtube8m/video/validate/ \
    --numIters $NUM_ITER \
    --lambda $LAMBDA \ 
    --numParts $NUM_PARTS > logisticRegression.$LAMBDA.$NUM_ITER.$DATE.txt
