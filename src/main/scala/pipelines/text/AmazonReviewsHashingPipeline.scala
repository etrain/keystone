package pipelines.text

import evaluation.BinaryClassifierEvaluator
import loaders.{AmazonReviewsDataLoader, LabeledData}
import nodes.learning.LogisticRegressionLBFGSEstimatorNoScaling
import nodes.nlp._
import org.apache.spark.{SparkConf, SparkContext}
import pipelines.Logging
import scopt.OptionParser
import workflow.Optimizer

object AmazonReviewsHashingPipeline extends Logging {
  val appName = "AmazonReviewsPipeline"

  def run(sc: SparkContext, conf: AmazonReviewsConfig) {

    logInfo("PIPELINE TIMING: Started training the classifier")
    val trainData = LabeledData(AmazonReviewsDataLoader(sc, conf.trainLocation, conf.threshold).labeledData.repartition(conf.numParts))

    val training = trainData.data
    val labels = trainData.labels

    val numFeatures = 1048576

    // Build the classifier estimator
    logInfo("Training classifier")
    val predictorPipeline = Trim andThen LowerCase() andThen
        Tokenizer() andThen
        NGramsHashingTF(1 to conf.nGrams, numFeatures) andThen
        //NGramsFeaturizer(1 to conf.nGrams) andThen
        //HashingTFNode(numFeatures) andThen
        (LogisticRegressionLBFGSEstimatorNoScaling(numIters = 20, numFeatures = numFeatures), training, labels)


    val predictor = Optimizer.execute(predictorPipeline)
    logInfo("\n" + predictor.toDOTString)

    predictor.apply("Test review")
    logInfo("PIPELINE TIMING: Finished training the classifier")

    // Evaluate the classifier
    logInfo("PIPELINE TIMING: Evaluating the classifier")

    val testData = LabeledData(AmazonReviewsDataLoader(sc, conf.testLocation, conf.threshold).labeledData.repartition(conf.numParts).cache())
    val testLabels = testData.labels
    val testResults = predictor(testData.data)
    val eval = BinaryClassifierEvaluator(testResults.map(_ > 0), testLabels.map(_ > 0))

    logInfo("\n" + eval.summary())
    logInfo("PIPELINE TIMING: Finished evaluating the classifier")
  }

  case class AmazonReviewsConfig(
    trainLocation: String = "",
    testLocation: String = "",
    threshold: Double = 3.5,
    nGrams: Int = 2,
    commonFeatures: Int = 100000,
    numParts: Int = 512)

  def parse(args: Array[String]): AmazonReviewsConfig = new OptionParser[AmazonReviewsConfig](appName) {
    head(appName, "0.1")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    opt[Double]("threshold") action { (x,c) => c.copy(threshold=x)}
    opt[Int]("nGrams") action { (x,c) => c.copy(nGrams=x) }
    opt[Int]("commonFeatures") action { (x,c) => c.copy(commonFeatures=x) }
    opt[Int]("numParts") action { (x,c) => c.copy(numParts=x) }
  }.parse(args, AmazonReviewsConfig()).get

  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
   * @param args
   */
  def main(args: Array[String]) = {
    val conf = new SparkConf().setAppName(appName)
    conf.setIfMissing("spark.master", "local[8]") // This is a fallback if things aren't set via spark submit.

    val sc = new SparkContext(conf)

    val appConfig = parse(args)
    run(sc, appConfig)

    sc.stop()
  }

}