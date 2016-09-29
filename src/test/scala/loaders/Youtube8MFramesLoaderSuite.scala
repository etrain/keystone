package loaders

import org.scalatest.FunSuite
import org.apache.spark.SparkContext
import utils.TestUtils
import workflow.PipelineContext

class Youtube8MFramesLoaderSuite extends FunSuite with PipelineContext {
  test("load a sample of youtube frames data") {
    sc = new SparkContext("local", "test")
    val dataPath = TestUtils.getTestResourceFileName("youtube8m.sample.tfrecord")

    val features = Youtube8MFramesLoader(sc, dataPath, 4)

    println(features.take(5).mkString(","))
  }
}
