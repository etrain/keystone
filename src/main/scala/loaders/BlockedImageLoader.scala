package nodes.loaders

import loaders.ImageLoaderUtils
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import pipelines.Transformer
import utils._

case class BlockId(x: Int, y: Int, z: Int)

object BlockedImageLoader {
  /**
   * Loads images from @dataPath and associates images with the labels provided in @labelPath
   *
   * @param sc SparkContext to use
   * @param dataPath Directory containing tar files (can be a HDFS path). This classes assumes
   *                 that each tar file contains images within a directory. The name of the
   *                 directory is treated as the className.
   * @param blockSize Triple that indicates the maximum block size in terms of (x, y, z) dimensions.
   * @param padding How much padding to add to each block (useful for 3D filters).
   */
  def apply(sc: SparkContext, dataPath: String, blockSize: (Int, Int, Int), padding: Int = 2): RDD[(BlockId, Image)] = {
    val filePathsRDD = ImageLoaderUtils.getFilePathsRDD(sc, dataPath)

    def depthGrabber(fname: String): Int = {
      fname.split("[_\\.]")(2).toInt
    }

    val res = ImageLoaderUtils.loadFiles(filePathsRDD, depthGrabber, LabeledImage.apply)

    BlockedImageCreator(blockSize._1, blockSize._2, blockSize._3, padding)(res)

  }
}

case class BlockedImageCreator(strideX: Int, strideY: Int, strideZ: Int, padding: Int)
    extends Transformer[LabeledImage, (BlockId, Image)] {

  def getSubregion(in: Image, xmin: Int, xmax: Int, ymin: Int, ymax: Int): Image = {
    val xdim = math.min(in.metadata.xDim, xmax) - xmin
    val ydim = math.min(in.metadata.yDim, ymax) - ymin

    val newImage = new ChannelMajorArrayVectorizedImage(
      new Array[Double](xdim*ydim*in.metadata.numChannels), ImageMetadata(xdim, ydim, in.metadata.numChannels))

    var x = 0
    while (x < xdim) {
      var y = 0
      while (y < ydim) {
        var c = 0
        while (c < in.metadata.numChannels) {
          newImage.put(x, y, c, in.get(xmin+x, ymin+y, c))
          c+=1
        }
        y+=1
      }
      x+=1
    }

    newImage
  }

  def generateBlocks(in: LabeledImage): Iterator[(BlockId, (Int, Image))] = {
    val res = for {
      x <- 0 until in.image.metadata.xDim by strideX;
      y <- 0 until in.image.metadata.yDim by strideY
    } yield (
        BlockId(x, y, in.label/strideZ),
        (in.label,
         getSubregion(in.image, x, x + strideX + padding, y, y + strideY + padding)))

    res.toIterator
  }

  def combineImages(part: (BlockId, Iterable[(Int, Image)])): (BlockId, Image) = {
    val res = ImageUtils.combineChannels(part._2.toArray.sortBy(_._1).map(_._2))
    (part._1, res)
  }

  override def apply(in: RDD[LabeledImage]): RDD[(BlockId, Image)] = {
    in.flatMap(generateBlocks).groupByKey().map(combineImages)
  }

  def apply(in: LabeledImage): (BlockId, Image) = ???
}