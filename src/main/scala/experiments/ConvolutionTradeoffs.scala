package experiments

import breeze.linalg._
import breeze.math.Complex
import breeze.signal._
import nodes.images.Convolver
import org.apache.spark.{SparkContext, SparkConf}
import pipelines.Logging
import scopt.OptionParser
import utils.{ImageUtils, ImageMetadata, RowMajorArrayVectorizedImage, Image}

object ConvolutionTradeoffs extends Logging{
  val appName = "ConvolutionTradeoffs"

  def convolve2dFfts(x: DenseMatrix[Complex], m: DenseMatrix[Complex], origRows: Int, origCols: Int) = {
    //This code based (loosely) on the MATLAB Code here:
    //Assumes you've already computed the fft of x and m.
    //http://www.mathworks.com/matlabcentral/fileexchange/31012-2-d-convolution-using-the-fft

    //Step 1: Pure fucking magic.
    val res = ifft(x :* m).map(_.real)

    //Step 2: the output we care about is in the bottom right corner.
    val startr = origRows - 1
    val startc = origCols - 1
    res(startr until x.rows, startc until x.cols).copy
  }

  def getChannelMatrix(in: Image, c: Int): DenseMatrix[Double] = {
    val dat = (0 until in.metadata.xDim).flatMap(x => (0 until in.metadata.yDim).map(y => in.get(x,y,c))).toArray
    new DenseMatrix[Double](in.metadata.xDim, in.metadata.yDim, dat)
  }

  def addChannelMatrix(in: Image, c: Int, m: DenseMatrix[Double]) = {
    var x = 0
    while ( x < in.metadata.xDim ) {
      var y = 0
      while ( y < in.metadata.yDim ) {
        in.put(x, y, c, in.get(x,y,c) + m(x,y))
        y+=1
      }
      x+=1
    }
  }

  def padMat(m: DenseMatrix[Double], nrows: Int, ncols: Int): DenseMatrix[Double] = {
    val res = DenseMatrix.zeros[Double](nrows, ncols)
    res(0 until m.rows, 0 until m.cols) := m
    res
  }

  /**
   * Convolves an n-dimensional image with a k-dimensional
   * @param x
   * @param m
   * @return
   */
  def convolve2dFft(x: Image, m: Array[_ <: Image]): Image = {
    val mx = x.metadata.xDim - m.head.metadata.xDim + 1
    val my = x.metadata.yDim - m.head.metadata.yDim + 1
    val chans = x.metadata.numChannels

    val ressize = mx*my*m.length

    val res = new RowMajorArrayVectorizedImage(Array.fill(ressize)(0.0), ImageMetadata(mx, my, m.length))

    val start = System.currentTimeMillis
    val fftXs = (0 until chans).map(c => fft(getChannelMatrix(x, c)))
    val fftMs = (0 until m.length).map(f => (0 until chans).map(c => fft(padMat(getChannelMatrix(m(f), c), x.metadata.xDim, x.metadata.yDim)))).toArray
    val s1 = System.currentTimeMillis

    //logInfo(s"Length of Xs: ${fftXs.length}, Length of each m: ${fftMs.first.length}, Total ms: ${fftMs.length}")
    var c = 0
    while (c < chans) {
      var f = 0
      while (f < m.length) {
        val convBlock = convolve2dFfts(fftXs(c), fftMs(f)(c), m(f).metadata.xDim, m(f).metadata.yDim)
        addChannelMatrix(res, f, convBlock) //todo - this could be vectorized.
        f+=1
      }
      c+=1
    }
    val s3 = System.currentTimeMillis
    logInfo(s"FFT: ${s1-start}, Mulres: ${s3-s1}")

    res
  }

  def generateRandomImage(x: Int, y: Int, z: Int): Image = {
    RowMajorArrayVectorizedImage(DenseVector.rand(x*y*z).toArray, ImageMetadata(x,y,z))
  }

  def time(f: => Unit) = {
    val s = System.nanoTime
    f
    System.nanoTime - s
  }

  def gflops(flop: Long, timens: Long) = 1e-3*flop/timens


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

    //Copy the filters to an image for fft filtering.
    val imgFilters = filters
      .toArray
      .grouped(filterSize*filterSize*numChannels)
      .map(x => new RowMajorArrayVectorizedImage(x, ImageMetadata(filterSize, filterSize, numChannels)))
      .toArray

    val ffttimings = images.map(i => time((convolve2dFft(i, imgFilters))))
    ffttimings.foreach(i => logInfo(s"FFT: ${i/1e6}, GFlops: ${gflops(ops, i)}")) //TODO: gotta do better than this.

    //Do standard convolution loop.
    val convolver = new Convolver(filters, imgSize, imgSize, numChannels, None)
    val vanillaTimings = images.map(i => time((convolver(i))))
    println(s"Size of vanillaTimings: ${vanillaTimings.length}")
    logInfo("Stuff")

    vanillaTimings.foreach(i => logInfo(s"Vanilla: ${i/1e6}, GFlops: ${gflops(ops, i)}")) //TODO: gotta do better than this.

    //Do FFT convolution loop.
    //Run the fft version.

    val shivTimings = images.map(i => (time(ImageUtils.conv2D(i, sepFilters(0).toArray, sepFilters(1).toArray))))
    val shivOps = 2*(filterSize*imgSize*imgSize*numChannels)
    shivTimings.foreach(i => logInfo(s"Shiv: ${i/1e6}"))
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