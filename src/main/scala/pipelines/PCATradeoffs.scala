package pipelines

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import edu.berkeley.cs.amplab.mlmatrix.{TruncatedSVD, TSQR, RowPartitionedMatrix}
import edu.berkeley.cs.amplab.mlmatrix.util.QRUtils
import nodes.images.Convolver
import nodes.learning.PCAEstimator
import org.apache.spark.{SparkContext, SparkConf}
import scopt.OptionParser
import utils.{ImageUtils, ImageMetadata, RowMajorArrayVectorizedImage, Image}

object PCATradeoffs extends Logging {
  val appName = "PCATradeoffs"
  def time[T](f: => T) = {
    val s = System.nanoTime
    val res = f
    (res, (System.nanoTime - s)/1e6)
  }

  def gflops(flop: Long, timens: Long) = flop/timens

  def shape[T](x: DenseMatrix[T]) = s"(${x.rows},${x.cols})"

  def approximatePCA(data: DenseMatrix[Double], k: Int, q: Int = 10, p: Int = 5) = {
    //This algorithm corresponds to Algorithms 4.4 and 5.1 of Halko, Martinsson, and Tropp, 2011.
    //According to sections 9.3 and  9.4 of the same, Ming Gu argues for exponentially fast convergence.

    val means = (mean(data(::, *))).toDenseVector

    val A = data(*, ::) - means
    val d = A.cols

    val l = k + p
    val omega = new DenseMatrix(d, l, randn(d*l).toArray) //cpu: d*l, mem: d*l
    val y0 = A*omega //cpu: n*d*l, mem: n*l

    var Q = QRUtils.qrQR(y0)._1 //cpu: n*l**2

    for (i <- 1 to q) {
      val YHat = Q.t * A //cpu: l*n*d, mem: l*d
      val Qh = QRUtils.qrQR(YHat.t)._1 //cpu: d*l^2, mem: d*l

      val Yj = A * Qh //cpu: n*d*l, mem: n*l
      Q = QRUtils.qrQR(Yj)._1 //cpu:  n*l^2, mem: n*l
    }

    val B = Q.t * A //cpu: l*n*d, mem: l*d
    val usvt = svd.reduced(B) //cpu: l*d^2, mem: l*d
    val pca = usvt.Vt.t
    logDebug(s"shape of pca ${shape(pca)}")

    // Mimic matlab
    // Enforce a sign convention on the coefficients -- the largest element in
    // each column will have a positive sign.

    val colMaxs = max(pca(::, *)).toArray
    val absPCA = abs(pca)
    val absColMaxs = max(absPCA(::, *)).toArray
    val signs = colMaxs.zip(absColMaxs).map { x =>
      if (x._1 == x._2) 1.0 else -1.0
    }

    pca(*, ::) :*= new DenseVector(signs)

    // Return a subset of the columns.
    pca(::, 0 until k)
  }

  def fro(x: DenseMatrix[Double]): Double = math.sqrt(x.values.map(x => x*x).sum)
  def approxError(vapprox: DenseMatrix[Double], v: DenseMatrix[Double]) = sum(abs(1.0 - (vapprox / v)))/(v.rows*v.cols)

  def run(sc: SparkContext, conf: PCATradeoffConfig) = {
    if (conf.local) {
    //First run the approximate things.
      for (
        n <- conf.ns;
        d <- conf.ds;
        k <- conf.ks;
        q <- conf.qs;
        p <- conf.ps;
        t <- 1 to conf.trials
      ) {
        //Generate the data
        val data = DenseMatrix.rand[Double](n, d)

        //Get a PCA object
        val dims = math.ceil(k * d).toInt
        val pca = new PCAEstimator(dims)
        val (p1, timing) = time(pca.computePCAd(data, dims))
        logInfo(s"svdPCA,$n,$d,$k,$q,0,$t,$dims,$timing,0.0,0.0")

        //Do the approximate PCA

        val (p2, timing2) = time(approximatePCA(data, dims, q, p))
        val absdiff = abs(p1) - abs(p2)

        logInfo(s"approxPCA,$n,$d,$k,$q,$p,$t,$dims,$timing2,${fro(absdiff) / (d * dims)},${approxError(p2, p1)}")

        logDebug(s"p1 ${shape(p1)}: $p1")
        logDebug(s"p2 ${shape(p2)}: $p2")


        logDebug(s"Max diff: ${absdiff.max}, norm: ${fro(absdiff)}")
      }

    } else {
      for (
        n <- conf.ns;
        d <- conf.ds;
        k <- conf.ks;
        q <- conf.qs;
        p <- conf.ps;
        t <- 1 to conf.trials
      ) {
        val dims = math.ceil(k * d).toInt

        val mat = RowPartitionedMatrix.createRandom(sc, n, d, conf.numParts, true)
        mat.rdd.count()

        //Get a PCA object
        val (p1, timing) = time {
          val rPart = new TSQR().qrR(mat)
          val qrDone = System.nanoTime()

          val asvd = svd(rPart)

          asvd.Vt(0 until dims, ::).t
        }
        logInfo(s"distsvdPCA,$n,$d,$k,$q,0,$t,$dims,$timing,0.0,0.0")

        //Do the approximate PCA
        val (p2, timing2) = time {
          val res = new TruncatedSVD().computePartial(mat, dims, q)

          val topres = res._2.t
          topres
        }

        val absdiff = abs(p1) - abs(p2)

        logInfo(s"distapproxPCA,$n,$d,$k,$q,$p,$t,$dims,$timing2,${fro(absdiff) / (d * dims)},${approxError(p2, p1)}")

        logDebug(s"p1 ${shape(p1)}: $p1")
        logDebug(s"p2 ${shape(p2)}: $p2")


        logDebug(s"Max diff: ${absdiff.max}, norm: ${fro(absdiff)}")

      }

    }
  }

  //Command line arguments. sizes, channels, num filters, filter size range. separable/not.
  case class PCATradeoffConfig(
      local: Boolean = true,
      ns: Array[Int] = Array(1e4, 1e5, 1e6).map(_.toInt),
      ds: Array[Int] = Array(256, 1024, 4096, 8192).map(_.toInt),
      ks: Array[Double] = Array(1e-3, 1e-2, 1e-1, 0.5, 1.0),
      qs: Array[Int] = Array(50),
      ps: Array[Int] = Array(5),
      numParts: Int = 256,
      trials: Int = 1)

  def parse(args: Array[String]): PCATradeoffConfig = new OptionParser[PCATradeoffConfig](appName) {
    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[Boolean]("local") action { (x,c) => c.copy(local=x) }
    opt[String]("ns") action { (x,c) => c.copy(ns=x.split(",").map(_.toInt).toArray) } text("size of dataset")
    opt[String]("ds") action { (x,c) => c.copy(ds=x.split(",").map(_.toInt).toArray) } text("number of dimensions")
    opt[String]("ks") action { (x,c) => c.copy(ks=x.split(",").map(_.toDouble).toArray) } text("how many components to recover")
    opt[String]("qs") action { (x,c) => c.copy(qs=x.split(",").map(_.toInt).toArray) } text("how many iterations of approximate to run")
    opt[String]("ps") action { (x,c) => c.copy(ps=x.split(",").map(_.toInt).toArray) } text("how much padding to use")
    opt[Int]("trials") action { (x,c) => c.copy(trials=x) } text("how many trials to run")
    opt[Int]("numParts") action { (x,c) => c.copy(numParts=x) }
  }.parse(args, PCATradeoffConfig()).get

  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
   * @param args
   */
  def main(args: Array[String]) = {
    val appConfig = parse(args)

    val conf = new SparkConf().setAppName(appName)
    conf.setIfMissing("spark.master", "local[2]")

    val sc = new SparkContext(conf)
    run(sc, appConfig)

    sc.stop()
  }
}