#!/bin/bash

LOGDIR=/mnt/log
PROF_DIR=/mnt/profiles
PDF_DIR=/mnt/pdfs
DATE=`date "+%Y%m%d.%H%M%S"`
NUM_WORKERS=`wc -l /root/spark/conf/slaves`
NUM_CORES=$(( `cat /proc/cpuinfo | grep processor | wc -l` / 2 ))
NUM_PARTITIONS=$(( $NUM_WORKERS * $NUM_CORES ))

TRAIN_DIR=/amazon/train
TEST_DIR=/amazon/test

EXP=Amazon

#Do Amazon
for MEM in 80g 40g 20g 10g 5g 2.5g 1g 500m
do
  for STRAT in Greedy EstOnly All
  do
    LOGFILE=$LOGDIR/$EXP.$STRAT.$MEM.$DATE.log
  
    KEYSTONE_MEM=$MEM bash bin/run-pipeline.sh workflow.OptimizerEvaluator \
      --trainLocation $TRAIN_DIR \
      --trainLabels $TRAIN_DIR \
      --testLocation $TEST_DIR \
      --testLabels $TEST_DIR \
      --memSize $MEM \
      --testPipeline $EXP \
      --numPartitions $NUM_PARTITIONS \
      --numWorkers $NUM_WORKERS \
      --cachingStrategy $STRAT \
      --profilesDir $PROF_DIR \
      --pdfDir $PDF_DIR \
      --sampleSizes 2,4 >$LOGFILE 2>&1
  done
done
