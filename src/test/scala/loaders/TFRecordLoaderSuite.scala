package loaders

import org.scalatest.FunSuite
import org.apache.spark.SparkContext
import utils.TestUtils
import workflow.PipelineContext

import org.tensorflow.example.feature._

class TFRecordLoaderSuite extends FunSuite with PipelineContext {
//  test("load a sample of tensorflow frame data") {
//    sc = new SparkContext("local", "test")
//    val dataPath = TestUtils.getTestResourceFileName("youtube8m.sample.tfrecord")
//
//    val features = TFRecordLoader(sc, dataPath, 4, SequenceExample.parseFrom)
//
//    println(features.take(5).mkString(","))
//  }

  test("load a sample of tensorflow frame data") {
    sc = new SparkContext("local", "test")
    val dataPath = TestUtils.getTestResourceFileName("youtube8m.video.sample.tfrecord")

    val features = TFRecordLoader(sc, dataPath, 4, Example.parseFrom)

    println(features.take(5).mkString(","))
  }
}
