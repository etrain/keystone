#!/bin/bash

TRAINPATH=/data/sparks/datasets/amazon/train
TESTPATH=/data/sparks/datasets/amazon/test

LOGDIR=`pwd`/log

if [ ! -d $LOGDIR ]
then
	mkdir -p $LOGDIR
fi

DATE=`date +%Y%m%d.%H%M%S`

#Run the pipeline
KEYSTONE_MEM=4g ./bin/run-pipeline.sh \
  pipelines.text.AmazonReviewsTuningPipeline \
  --trainLocation $TRAINPATH \
  --numConfigs 1 \
  --profile true \
  --testLocation $TESTPATH > $LOGDIR/amazon.$DATE.log
