package pipelines.video.youtube8m

import java.io.File

import breeze.linalg._
import breeze.stats._
import evaluation.MeanAveragePrecisionEvaluator
import loaders._
import nodes.images.external.{FisherVector, SIFTExtractor}
import nodes.images._
import nodes.learning._
import nodes.stats.{ColumnSampler, NormalizeRows, SignedHellingerMapper}
import nodes.util.{Cacher, ClassLabelIndicatorsFromIntArrayLabels, FloatToDouble, MatrixVectorizer}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import pipelines.Logging
import scopt.OptionParser
import utils.Image
import workflow.{Transformer, Pipeline}

object Youtube8MVideoWhitenedLinear extends Serializable with Logging {
  val appName = "Youtube8MLinear"

  def getLabelMatrices(data: RDD[MultiLabeledFeatureVector]): (RDD[DenseVector[Double]], RDD[DenseVector[Double]]) = {
    val labelGrabber = ClassLabelIndicatorsFromIntArrayLabels(Youtube8MVideoLoader.NUM_CLASSES) andThen
      new Cacher

    val X = data.map(_.features.map(_.toDouble)).cache()
    val y = labelGrabber(data.map(_.labels)).get

    (X, y)
  }

  def computeOrLoadWhitener(
      trainData: RDD[DenseVector[Double]],
      location: Option[String],
      fraction: Double,
      epsilon: Double): ZCAWhitener = location match {
    case Some(fname) => {
      val whitenerMat = csvread(new File(s"$fname.whitener.csv"))
      val whitenerMeans = csvread(new File(s"$fname.means.csv")).toDenseVector

      new ZCAWhitener(whitenerMat, whitenerMeans)
    }
    case None => {
      //Collect samples.
      val samples = trainData.sample(false, fraction, 42).collect()
      val sampleMatrix = DenseMatrix(samples.map(_.toArray): _*)
      new ZCAWhitenerEstimator(epsilon).fitSingle(convert(sampleMatrix, Double))
    }
  }



  def run(sc: SparkContext, conf: LinearConfig): Pipeline[DenseVector[Double], DenseVector[Double]] =  {

    val trainData = Youtube8MVideoLoader(sc, conf.trainLocation, conf.numParts).cache()
    val (trainX, trainy) = getLabelMatrices(trainData)
    logInfo(s"Size of trainX: ${trainX.count}, trainy: ${trainy.count}")

    val whitener = computeOrLoadWhitener(trainX, conf.whitenerLocation, conf.whiteFraction, conf.whiteEpsilon)

    conf.saveWhitenerLocation match {
      case Some(fname) => {
        csvwrite(new File(s"$fname.whitener.csv"), whitener.whitener)
        csvwrite(new File(s"$fname.means.csv"), whitener.means.toDenseMatrix)
      }
      case _ => Unit
    }

    // Now featurize and apply the model to test data.
    val testData = Youtube8MVideoLoader(sc, conf.testLocation, conf.numParts).cache()
    val (testX, testy) = getLabelMatrices(testData)
    logInfo(s"Size of testX: ${testX.count}, testy: ${testy.count}")
    val testActuals = testData.map(_.labels).cache()

    def toWhiteFormat(x: DenseVector[Double]): DenseMatrix[Double] = x.toDenseMatrix
    def fromWhiteFormat(x: DenseMatrix[Double]): DenseVector[Double] = x.toDenseVector


    val predictor = Transformer(toWhiteFormat) andThen
      whitener andThen
      Transformer(fromWhiteFormat) andThen
      new Cacher andThen
      (new LinearMapEstimator(conf.lambda), trainX, trainy) andThen
      new Cacher

    val predictions = predictor(testX)

    val map = MeanAveragePrecisionEvaluator(testActuals, predictions.get, Youtube8MVideoLoader.NUM_CLASSES)
    logInfo(s"TEST APs ${conf.lambda} are: ${map.toArray.mkString(",")}")
    logInfo(s"TEST MAP ${conf.lambda} is: ${mean(map)}")

    predictor.toPipeline
  }

  case class LinearConfig(
    trainLocation: String = "",
    testLocation: String = "",
    numParts: Int = 496,
    lambda: Option[Double] = None,
    whiteFraction: Double = 0.1,
    whiteEpsilon: Double = 0.1,
    whitenerLocation: Option[String] = None,
    saveWhitenerLocation: Option[String] = None)

  def parse(args: Array[String]): LinearConfig = {
    val conf = new OptionParser[LinearConfig](appName) {
      head(appName, "0.1")
      help("help") text ("prints this usage text")
      opt[String]("trainLocation") required() action { (x, c) => c.copy(trainLocation = x) }
      opt[String]("testLocation") required() action { (x, c) => c.copy(testLocation = x) }
      opt[Int]("numParts") action { (x, c) => c.copy(numParts = x) }
      opt[Double]("lambda") action { (x, c) => c.copy(lambda = Some(x)) }
      opt[Double]("whiteFraction") action { (x, c) => c.copy(whiteFraction = x) }
      opt[Double]("whiteEpsilon") action { (x, c) => c.copy(whiteEpsilon = x) }
      opt[String]("whitenerLocation") action { (x, c) => c.copy(whitenerLocation = Some(x)) }
      opt[String]("saveWhitenerLocation") action { (x,c) => c.copy(saveWhitenerLocation = Some(x)) }
    }.parse(args, LinearConfig()).get

    conf

  }
  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
   *
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
