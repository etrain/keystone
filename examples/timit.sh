#!/bin/bash

TRAINPATH=/Users/sparks/datasets/timit-test-features.csv
TRAINLABELPATH=/Users/sparks/datasets/timit-test-labels.sparse
TESTPATH=/Users/sparks/datasets/timit-test-features.csv
TESTLABELPATH=/Users/sparks/datasets/timit-test-labels.sparse

LOGDIR=`pwd`/log

if [ ! -d $LOGDIR ]
then
	mkdir -p $LOGDIR
fi

DATE=`date +%Y%m%d.%H%M%S`

#Run the pipeline
KEYSTONE_MEM=12g ./bin/run-pipeline.sh \
  pipelines.text.TimitTuningPipeline \
  --trainDataLocation $TRAINPATH \
  --trainLabelsLocation $TRAINLABELPATH \
  --testDataLocation $TESTPATH \
  --testLabelsLocation $TESTLABELSPATH \
  --numParts 4 \
  --numConfigs 1 \
  --profile true \
  --testLocation $TESTPATH > $LOGDIR/newsgroups.$DATE.log
