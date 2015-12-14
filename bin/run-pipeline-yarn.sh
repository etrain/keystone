#!/bin/bash

# Figure out where we are.
FWDIR="$(cd `dirname $0`; pwd)"

CLASS=$1
shift

KEYSTONE_MEM=${KEYSTONE_MEM:-1g}
export KEYSTONE_MEM

export LD_LIBRARY_PATH=/home/eecs/sparks/matrix-bench/openblas-install/lib/lib

spark-submit \
  --master yarn \
  --num-executors 12 \
  --executor-cores 16 \
  --class $CLASS \
  --driver-class-path $FWDIR/../target/scala-2.10/keystoneml-assembly-0.3.0-SNAPSHOT.jar \
  --driver-library-path /home/eecs/sparks/matrix-bench/openblas-install/lib/lib:$FWDIR/../lib \
  --conf spark.executor.extraLibraryPath=/home/eecs/sparks/matrix-bench/openblas-install/lib/lib:$FWDIR/../lib \
  --conf spark.executor.extraClassPath=$FWDIR/../target/scala-2.10/keystoneml-assembly-0.3.0-SNAPSHOT.jar \
  --conf spark.serializer=org.apache.spark.serializer.JavaSerializer \
  --conf spark.executorEnv.LD_LIBRARY_PATH=/home/eecs/sparks/matrix-bench/openblas-install/lib/lib \
  --conf spark.yarn.executor.memoryOverhead=15300 \
  --conf spark.mlmatrix.treeBranchingFactor=16 \
  --conf spark.driver.maxResultSize=0 \
  --driver-memory $KEYSTONE_MEM \
  --executor-memory 130g \
  target/scala-2.10/keystoneml-assembly-0.3.0-SNAPSHOT.jar \
  "$@"

