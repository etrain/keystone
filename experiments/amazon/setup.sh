#!/bin/bash
HADOOP=/root/mapreduce/bin/hadoop

# Start mapreduce if necessary
/root/mapreduce/bin/start-mapred.sh
/root/ephemeral-hdfs/sbin/start-all.sh

# Get the data from S3
$HADOOP distcp s3n://files.sparks.west/data/amazon/ /amazon/train
/root/ephemeral-hdfs/bin/hadoop dfs -mkdir /amazon/test
/root/ephemeral-hdfs/bin/hadoop dfs -mv /amazon/train/part-aam* /amazon/test/
/root/ephemeral-hdfs/bin/hadoop dfs -mv /amazon/train/part-aal* /amazon/test/
/root/ephemeral-hdfs/bin/hadoop dfs -mv /amazon/train/part-aak* /amazon/test/