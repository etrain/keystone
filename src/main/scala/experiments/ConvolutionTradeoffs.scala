package experiments

import breeze.linalg._
import nodes.images.Convolver
import org.apache.spark.{SparkContext, SparkConf}
import pipelines.Logging
import scopt.OptionParser
import utils.{ImageUtils, ImageMetadata, RowMajorArrayVectorizedImage, Image}

object ConvolutionTradeoffs extends Logging{
  val appName = "ConvolutionTradeoffs"

  def generateRandomImage(x: Int, y: Int, z: Int): Image = {
    RowMajorArrayVectorizedImage(DenseVector.rand(x*y*z).toArray, ImageMetadata(x,y,z))
  }

  def time(f: => Unit) = {
    val s = System.nanoTime
    f
    System.nanoTime - s
  }

  def gflops(flop: Long, timens: Long) = flop/timens


  def run(imgSize: Int, numChannels: Int, filterSize: Int, numFilters: Int, numImages: Int, separable: Boolean, usef2: Boolean) = {
    if (usef2) {
      val props = System.getProperties()
      props.setProperty("com.github.fommil.netlib.BLAS","com.github.fommil.netlib.F2jBLAS")
      props.setProperty("com.github.fommil.netlib.LAPACK","com.github.fommil.netlib.F2jLAPACK")
      props.setProperty("com.github.fommil.netlib.ARPACK","com.github.fommil.netlib.F2jARPACK")
    }

    val ops = imgSize.toLong*(numChannels*filterSize*filterSize)*numFilters

    //Randomly create a set of images.
    val images: Array[Image] = (0 until numImages)
      .map(x => generateRandomImage(imgSize, imgSize, numChannels))
      .toArray

    val sepFilters = Array(DenseVector.rand(filterSize), DenseVector.rand(filterSize))

    //Randomly create a set of filters.
    val filters: DenseMatrix[Double] = separable match {
        case true => sepFilters(0).toDenseMatrix.t * sepFilters(1).toDenseMatrix
        case false => DenseMatrix.rand(numFilters, filterSize*filterSize*numChannels)
    }

    //Do standard convolution loop.
    val convolver = new Convolver(filters, imgSize, imgSize, numChannels, None)
    val vanillaTimings = images.map(i => time((convolver(i))))
    println(s"Size of vanillaTimings: ${vanillaTimings.length}")
    logInfo("Stuff")

    vanillaTimings.foreach(i => logInfo(s"Vanilla: $i, GFlops: ${gflops(i, ops)}")) //TODO: gotta do better than this.

    //Do FFT convolution loop.
    //Run the fft version.

    val shivTimings = images.map(i => (time(ImageUtils.conv2D(i, sepFilters(0).toArray, sepFilters(1).toArray))))
    val shivOps = 2*(filterSize*imgSize*imgSize*numChannels)
    shivTimings.foreach(i => logInfo(s"Shiv: $i"))
  }

  //Command line arguments. sizes, channels, num filters, filter size range. separable/not.
  case class ConvolutionTradeoffConfig(
    imgSize: Int = 256,
    numFilters: Int = 100,
    filterSize: Int = 3,
    numChannels: Int = 3,
    numImages: Int = 10,
    separable: Boolean = false,
    usef2: Boolean = false)

  def parse(args: Array[String]): ConvolutionTradeoffConfig = new OptionParser[ConvolutionTradeoffConfig](appName) {
    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[Int]("imgSize") action { (x,c) => c.copy(imgSize=x) } text("size of (square) image")
    opt[Int]("numFilters") action { (x,c) => c.copy(numFilters=x) } text("number of filters")
    opt[Int]("filterSize") action { (x,c) => c.copy(filterSize=x) } text("size of (square) filter")
    opt[Int]("numChannels") action { (x,c) => c.copy(numChannels=x) } text("number of channels per image")
    opt[Int]("numImages") action { (x,c) => c.copy(numImages=x) } text("number of images to test")
    opt[Unit]("separable") action { (_, c) => c.copy(separable=true) } text("are the convolutions separable")
    opt[Unit]("usef2") action { (_, c) => c.copy(usef2=true) } text("use f2jblas?")
  }.parse(args, ConvolutionTradeoffConfig()).get

  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
   * @param args
   */
  def main(args: Array[String]) = {
    val conf = parse(args)
    run(conf.imgSize, conf.numChannels, conf.filterSize, conf.numFilters, conf.numImages, conf.separable, conf.usef2)
  }
}