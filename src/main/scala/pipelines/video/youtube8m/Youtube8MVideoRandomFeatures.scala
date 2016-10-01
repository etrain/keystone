package pipelines.video.youtube8m

import java.io.File

import breeze.linalg._
import breeze.stats._
import breeze.stats.distributions.{ThreadLocalRandomGenerator, RandBasis, CauchyDistribution}
import evaluation.MeanAveragePrecisionEvaluator
import loaders._
import nodes.images.external.{FisherVector, SIFTExtractor}
import nodes.images._
import nodes.learning._
import nodes.stats.{CosineRandomFeatures, ColumnSampler, NormalizeRows, SignedHellingerMapper}
import nodes.util._
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import pipelines.Logging
import pipelines.speech.TimitPipeline.Distributions
import scopt.OptionParser
import utils.Image
import workflow.Pipeline

object Youtube8MVideoRandomFeatures extends Serializable with Logging {
  val appName = "Youtube8MRandomFeatures"

  def getLabelMatrices(data: RDD[MultiLabeledFeatureVector]): (RDD[DenseVector[Double]], RDD[DenseVector[Double]]) = {
    val labelGrabber = ClassLabelIndicatorsFromIntArrayLabels(Youtube8MVideoLoader.NUM_CLASSES) andThen
      new Cacher

    val X = data.map(_.features.map(_.toDouble)).cache()
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

  def run(sc: SparkContext, conf: RandomFeaturesConfig): Pipeline[DenseVector[Double], DenseVector[Double]] =  {

    val trainData = Youtube8MVideoLoader(sc, conf.trainLocation, conf.numParts).cache()
    val (trainX, trainy) = getLabelMatrices(trainData)
    logInfo(s"Size of trainX: ${trainX.count}, testy: ${trainy.count}")

    // Set the constants
    val seed = 123L
    val random = new java.util.Random(seed)
    val randomSource = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(random.nextLong())))

    val numCosineFeatures = conf.numCosineFeatures
    val numCosineBatches = conf.numCosines
    val colsPerBatch = numCosineFeatures + 1


    // Now featurize and apply the model to test data.
    val testData = Youtube8MVideoLoader(sc, conf.testLocation, conf.numParts).cache()
    val (testX, testy) = getLabelMatrices(testData)
    logInfo(s"Size of testX: ${testX.count}, testy: ${testy.count}")
    val testActuals = testData.map(_.labels).cache()

    val featurizer = Pipeline.gather {
      Seq.fill(numCosineBatches) {
        if (conf.rfType == Distributions.Cauchy) {
          // TODO: Once https://github.com/scalanlp/breeze/issues/398 is released,
          // use a RandBasis for cauchy
          CosineRandomFeatures(
            Youtube8MVideoLoader.NUM_FEATURES,
            numCosineFeatures,
            conf.gamma,
            new CauchyDistribution(0, 1),
            randomSource.uniform).toPipeline
        } else {
          CosineRandomFeatures(
            Youtube8MVideoLoader.NUM_FEATURES,
            numCosineFeatures,
            conf.gamma,
            randomSource.gaussian,
            randomSource.uniform).toPipeline
        }
      }
    } andThen VectorCombiner()

    val predictor = featurizer andThen (new BlockLeastSquaresEstimator(numCosineFeatures, conf.numEpochs, conf.lambda),
      trainX, trainy)

    val predictions = predictor(testX).get.cache()

    val map = MeanAveragePrecisionEvaluator(testActuals, predictions, Youtube8MVideoLoader.NUM_CLASSES)
    val hit1 = hitAtK(testActuals, predictions, 1)
    val hit5 = hitAtK(testActuals, predictions, 5)
    logInfo(s"TEST APs ${conf.lambda} are: ${map.toArray.mkString(",")}")
    logInfo(s"TEST MAP ${conf.lambda} is: ${mean(map)}")
    logInfo(s"Hit@1: $hit1")
    logInfo(s"Hit@5: $hit5")

    predictor.toPipeline
  }

  case class RandomFeaturesConfig(
    trainLocation: String = "",
    testLocation: String = "",
    numParts: Int = 496,
    lambda: Double = 0.0,
    rfType: Distributions.Value = Distributions.Gaussian,
    numCosines: Int = 50,
    gamma: Double = 0.05555,
    numEpochs: Int = 5,
    numCosineFeatures: Int = 4096,
    checkpointDir: Option[String] = None)

  def parse(args: Array[String]): RandomFeaturesConfig = new OptionParser[RandomFeaturesConfig](appName) {
    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    opt[Int]("numParts") action { (x,c) => c.copy(numParts=x) }
    opt[Double]("lambda") action { (x,c) => c.copy(lambda=x) }
    opt("rfType")(scopt.Read.reads(Distributions withName _)) action { (x,c) => c.copy(rfType = x)}
    opt[Int]("numCosines") action { (x,c) => c.copy(numCosines=x) }
    opt[Int]("numEpochs") action { (x,c) => c.copy(numEpochs=x) }
    opt[Int]("numCosineFeatures") action { (x,c) => c.copy(numCosineFeatures=x) }
    opt[Double]("gamma") action { (x,c) => c.copy(gamma=x) }
    opt[String]("checkpointDir") action { (x,c) => c.copy(checkpointDir=Some(x)) }
  }.parse(args, RandomFeaturesConfig()).get

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
