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
import workflow.Pipeline

object Youtube8MVideoLinear extends Serializable with Logging {
  val appName = "Youtube8MLinear"

  def getLabelMatrices(data: RDD[MultiLabeledFeatureVector]): (RDD[DenseVector[Double]], RDD[DenseVector[Double]]) = {
    val labelGrabber = ClassLabelIndicatorsFromIntArrayLabels(Youtube8MVideoLoader.NUM_CLASSES) andThen
      new Cacher

    val X = data.map(_.features.map(_.toDouble)).cache()
    val y = labelGrabber(data.map(_.labels)).get

    (X, y)
  }

  def run(sc: SparkContext, conf: LinearConfig): Pipeline[DenseVector[Double], DenseVector[Double]] =  {



    val trainData = Youtube8MVideoLoader(sc, conf.trainLocation, conf.numParts).cache()
    val (trainX, trainy) = getLabelMatrices(trainData)
    logInfo(s"Size of trainX: ${trainX.count}, testy: ${trainy.count}")


    // Now featurize and apply the model to test data.
    val testData = Youtube8MVideoLoader(sc, conf.testLocation, conf.numParts).cache()
    val (testX, testy) = getLabelMatrices(testData)
    logInfo(s"Size of testX: ${testX.count}, testy: ${testy.count}")
    val testActuals = testData.map(_.labels).cache()

    val lambdas: Seq[Option[Double]] = Seq(None, Some(1e-4), Some(1e-2), Some(1e2), Some(1e4), Some(1e-3))
    val params = for (l <- lambdas.reverse) yield {
      val predictor = new LinearMapEstimator(l).fit(trainX, trainy)

      val predictions = predictor(testX)

      val map = MeanAveragePrecisionEvaluator(testActuals, predictions, Youtube8MVideoLoader.NUM_CLASSES)
      logInfo(s"TEST APs $l are: ${map.toArray.mkString(",")}")
      logInfo(s"TEST MAP $l is: ${mean(map)}")

      (mean(map), l, predictor)

    }

    val bestPredictor = params.maxBy(_._1)._3

    bestPredictor.toPipeline
  }

  case class LinearConfig(
    trainLocation: String = "",
    testLocation: String = "",
    numParts: Int = 496,
    lambda: Double = 0.5)

  def parse(args: Array[String]): LinearConfig = new OptionParser[LinearConfig](appName) {
    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    opt[Int]("numParts") action { (x,c) => c.copy(numParts=x) }
    opt[Double]("lambda") action { (x,c) => c.copy(lambda=x) }
  }.parse(args, LinearConfig()).get

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
