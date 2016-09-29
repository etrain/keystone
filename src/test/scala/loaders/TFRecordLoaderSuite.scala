package loaders

import org.scalatest.FunSuite
import org.apache.spark.SparkContext
import utils.TestUtils
import workflow.PipelineContext

class TFRecordLoaderSuite extends FunSuite with PipelineContext {
  test("load a sample of imagenet data") {
    sc = new SparkContext("local", "test")
    val dataPath = TestUtils.getTestResourceFileName("youtube8m.sample.tfrecord")

    val features = TFRecordLoader(sc, dataPath, 4)

    println(features.take(5).mkString(","))
  }
}
