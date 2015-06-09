package loaders

import javax.imageio.ImageIO

import nodes.loaders.BlockedImageLoader
import org.scalatest.FunSuite
import org.apache.spark.SparkContext

import pipelines.{Logging, LocalSparkContext}
import utils.TestUtils

class BlockedImageLoaderSuite extends FunSuite with LocalSparkContext with Logging{
  test("make sure we can get tiff decoder") {
    val reader = ImageIO.getImageReadersByFormatName("TIFF")
    val readers = ImageIO.getReaderFormatNames()
    readers.map(println)
    assert(reader != null)
    assert(reader.hasNext, "No tiff reader")
  }

  test("load a sample of LBL data") {
    sc = new SparkContext("local", "test")
    val dataPath = TestUtils.getTestResourceFileName("images/lbl_sample.tar")

    val dims = (100,100,5)

    val imgs = BlockedImageLoader.apply(sc, dataPath, dims).collect()
    // We should have x image blocks images
    assert(imgs.length === 720)

    val z = imgs(100)._2.toArray.mkString(",")
    println(s"Random image: $z")

    // The biggest block should be x by y by z.
    val biggestX = imgs.map(_._2.metadata.xDim).reduce(math.max)
    val biggestY = imgs.map(_._2.metadata.yDim).reduce(math.max)
    val biggestZ = imgs.map(_._2.metadata.numChannels).reduce(math.max)

    //assert((biggestX-padding, biggestY-padding, biggestZ-padding) === dims)

    assert(imgs.map(_._2.metadata.xDim).forall(_ > 0))
    assert(imgs.map(_._2.metadata.yDim).forall(_ > 0))
    assert(imgs.map(_._2.metadata.numChannels).forall(_ > 0))
  }
}
