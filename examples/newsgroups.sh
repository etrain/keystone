#!/bin/bash

TRAINPATH=/Users/sparks/datasets/20news-bydate-train
TESTPATH=/Users/sparks/datasets/20news-bydate-test

LOGDIR=`pwd`/log

if [ ! -d $LOGDIR ]
then
	mkdir -p $LOGDIR
fi

DATE=`date +%Y%m%d.%H%M%S`

#Run the pipeline
KEYSTONE_MEM=4g ./bin/run-pipeline.sh \
  pipelines.text.NewsgroupsTuningPipeline \
  --trainLocation $TRAINPATH \
  --numConfigs 100 \
  --profile false \
  --testLocation $TESTPATH > $LOGDIR/newsgroups.$DATE.log
