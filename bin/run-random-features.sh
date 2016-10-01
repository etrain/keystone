#!/bin/bash

export SPARK_HOME=/root/spark 
KEYSTONE_MEM=80g 

bin/run-pipeline.sh pipelines.video.youtube8m.Youtube8MVideoRandomFeatures \
    --trainLocation /youtube8m/video/train/ \
    --testLocation /youtube8m/video/validate/ \
    --checkpointDir /mnt/checkpoints \
    --numCosines 2 \
    --numEpochs 1 \
    --numParts 384 > randomfeatures.20blocks.txt
