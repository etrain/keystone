#!/bin/bash

export SPARK_HOME=/root/spark 
export KEYSTONE_MEM=80g 

NUM_SLAVES=`wc -l $SPARK_HOME/conf/slaves | cut -d" " -f1`
NUM_PARTS=$((8 * $NUM_SLAVES))
echo $NUM_PARTS
echo

bin/run-pipeline.sh pipelines.video.youtube8m.Youtube8MVideoRandomFeatures \
    --trainLocation /youtube8m/video/train/ \
    --testLocation /youtube8m/video/validate/ \
    --checkpointDir /mnt/checkpoints \
    --numCosines 2 \
    --numEpochs 1 \
    --numParts $NUM_PARTS > randomfeatures.20blocks.txt
