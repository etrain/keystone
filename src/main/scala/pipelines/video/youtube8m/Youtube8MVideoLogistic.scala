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

object Youtube8MVideoLogistic extends Serializable with Logging {
  val appName = "Youtube8MLogistic"

  val labelGrabber = ClassLabelIndicatorsFromIntArrayLabels(Youtube8MVideoLoader.NUM_CLASSES) andThen
    new Cacher(Some("trainY"))

  def getLabelMatrices(data: RDD[MultiLabeledFeatureVector]): (RDD[DenseVector[Double]], RDD[Int]) = {
    val newData = data.flatMap(fv => {
      fv.labels.map(l => (l, fv.features))
    }).cache()

    val X = newData.map(_._2.map(_.toDouble)).cache().setName("trainX")
    val y = newData.map(_._1).cache().setName("trainY")

    (X, y)
  }

  def getDenseLabelMatrices(data: RDD[MultiLabeledFeatureVector]):   (RDD[DenseVector[Double]], RDD[DenseVector[Double]]) = {
    val X = data.map(_.features.map(_.toDouble)).cache().setName("trainX")
    val y = labelGrabber(data.map(_.labels)).get

    (X, y)
  }

  def hitAtK(actuals: RDD[Array[Int]], preds: RDD[DenseVector[Double]], k: Int = 1): Double = {
    val hitCount = preds.zip(actuals).map { case (pred, actual) => {
      val topPreds = pred.toArray.zipWithIndex.sortBy(_._1).takeRight(k).map(_._2)

      val hits = topPreds.toSet.intersect(actual.toSet)
      hits.size
    }}.filter(_ > 0).count()

    hitCount.toDouble/actuals.count()
  }

  def run(sc: SparkContext, conf: LogisticConfig): Pipeline[DenseVector[Double], DenseVector[Double]] =  {

    val trainData = Youtube8MVideoLoader(sc, conf.trainLocation, conf.numParts).cache().setName("trainData")

    // Now featurize and apply the model to test data.
    val testData = Youtube8MVideoLoader(sc, conf.testLocation, conf.numParts).cache().setName("testData")
    val testActuals = testData.map(_.labels).cache().setName("testActuals")

    val lambda = conf.lambda.getOrElse(0.0)

    val predictor = conf.method match {
      case "mllib" => {
        val (trainX, trainy) = getLabelMatrices(trainData)
        logInfo(s"Size of trainX: ${trainX.count}, testy: ${trainy.count}")

        new MultiLabelLogisticRegressionEstimator(
          Youtube8MVideoLoader.NUM_CLASSES,
          lambda,
          numIters = conf.numIters,
          numFeatures = Youtube8MVideoLoader.NUM_FEATURES,
          cache = true).fit(trainX, trainy)
      }
      case "dense" => {
        val (trainX, trainy) = getDenseLabelMatrices(trainData)
        new DenseLogisticRegressionEstimator(
          Youtube8MVideoLoader.NUM_FEATURES,
          lambda = lambda,
          numEpochs = conf.numIters).fit(trainX, trainy)
      }
    }
    logInfo("Training Model")

    val predictions = predictor(testData.map(_.features.map(_.toDouble))).cache().setName("predictions")
    logInfo("Model trained")

    logInfo(s"Size of prediction vector: ${predictions.first.length}")

    val hit1 = hitAtK(testActuals, predictions, 1)
    logInfo(s"Hit@1: $hit1")
    val hit5 = hitAtK(testActuals, predictions, 5)
    logInfo(s"Hit@5: $hit5")

    val map = MeanAveragePrecisionEvaluator(testActuals, predictions, Youtube8MVideoLoader.NUM_CLASSES)
    logInfo(s"TEST APs ${conf.lambda} are: ${map.toArray.mkString(",")}")
    logInfo(s"TEST MAP ${conf.lambda} is: ${mean(map)}")


    predictor.toPipeline
  }

  case class LogisticConfig(
    trainLocation: String = "",
    testLocation: String = "",
    numParts: Int = 496,
    numIters: Int = 20,
    method: String = "dense",
    lambda: Option[Double] = None)

  def parse(args: Array[String]): LogisticConfig = new OptionParser[LogisticConfig](appName) {
    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    opt[String]("method") action { (x,c) => c.copy(method=x) }
    opt[Int]("numParts") action { (x,c) => c.copy(numParts=x) }
    opt[Int]("numIters") action { (x,c) => c.copy(numIters=x) }
    opt[Double]("lambda") action { (x,c) => c.copy(lambda=Some(x)) }
  }.parse(args, LogisticConfig()).get

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
