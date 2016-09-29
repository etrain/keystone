package loaders

import org.scalatest.FunSuite
import org.apache.spark.SparkContext
import utils.TestUtils
import workflow.PipelineContext

class Youtube8MVideoLoaderSuite extends FunSuite with PipelineContext {
  test("load a sample of youtube video data") {
    sc = new SparkContext("local", "test")
    val dataPath = TestUtils.getTestResourceFileName("youtube8m.video.sample.tfrecord")

    val features = Youtube8MVideoLoader(sc, dataPath, 4)

    assert(features.count == 1376, "Sample file should have 1376 records in it.")
    assert(features.map(_.features.length == 1024).reduce(_ & _), "All features should be a vector length 1024.")
  }
}
