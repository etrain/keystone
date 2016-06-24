package pipelines.text

import breeze.linalg.SparseVector
import evaluation.MulticlassClassifierEvaluator
import loaders.NewsgroupsDataLoader
import nodes.learning.NaiveBayesEstimator
import nodes.nlp._
import nodes.stats.TermFrequency
import nodes.util.{CommonSparseFeatures, MaxClassifier}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import pipelines.Logging
import scopt.OptionParser
import workflow.Pipeline
import workflow.tuning._

object NewsgroupsTuningPipeline extends Logging {
  val appName = "NewsgroupsTuningPipeline"

  def run(sc: SparkContext, conf: NewsgroupsConfig): Pipeline[String, Int] = {

    val trainData = NewsgroupsDataLoader(sc, conf.trainLocation)
    val numClasses = NewsgroupsDataLoader.classes.length

    // Build the classifier estimator
    logInfo("Training classifier")

    val pipeSpace = TransformerP("Trim", {x: Int => Trim}, EmptyParameter()) andThen
      TransformerP("LowerCase", {x: Int => LowerCase()}, EmptyParameter()) andThen
      TransformerP("Tokenizer", {x: Int => Tokenizer()}, EmptyParameter()) andThen
      TransformerP("NGramsFeaturizer",
        {x:Int => NGramsFeaturizer(1 to x)},
        IntParameter("maxGrams", conf.nGramsMin, conf.nGramsMax)) andThen
      TransformerP("TermFrequency", {x:Int => TermFrequency(y => 1)}, EmptyParameter()) andThen
      EstimatorP("CommonSparseFeatures",
        {x:Int => CommonSparseFeatures[Seq[String]](x)},
        IntParameter("commonSparseFeatures", conf.commonFeaturesMin, conf.commonFeaturesMax),
        trainData.data) andThen
      LabelEstimatorP("NaiveBayesEstimator",
        {x:Int => NaiveBayesEstimator[SparseVector[Double]](numClasses)},
        EmptyParameter(),
        trainData.data,
        trainData.labels) andThen
      TransformerP("MaxClassifier", {x:Int => MaxClassifier}, EmptyParameter())

    val pipes = (0 until conf.numConfigs).map(x => pipeSpace.sample[String, Int]())

    val evaluator: (RDD[Int], RDD[Int]) => Double = MulticlassClassifierEvaluator(_, _, numClasses).macroFScore()

    val predictor = conf.execStyle match {
      case "tune" => PipelineTuning.tune(pipes, trainData.data, trainData.labels, evaluator)
      case "sequential" => PipelineTuning.sequential(pipes, trainData.data, trainData.labels, evaluator)
    }

    // Evaluate the classifier
    logInfo("Evaluating classifier")

    val testData = NewsgroupsDataLoader(sc, conf.testLocation)
    val testLabels = testData.labels
    val testResults = predictor(testData.data)
    val eval = MulticlassClassifierEvaluator(testResults.get, testLabels, numClasses)

    logInfo("\n" + eval.summary(NewsgroupsDataLoader.classes))

    predictor
  }

  case class NewsgroupsConfig(
    trainLocation: String = "",
    testLocation: String = "",
    nGramsMin: Int = 2,
    nGramsMax: Int = 4,
    commonFeaturesMin: Int = 1000,
    commonFeaturesMax: Int = 100000,
    numConfigs: Int = 10,
    execStyle: String = "tune")

  def parse(args: Array[String]): NewsgroupsConfig = new OptionParser[NewsgroupsConfig](appName) {
    head(appName, "0.1")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    opt[Int]("nGramsMin") action { (x,c) => c.copy(nGramsMin=x) }
    opt[Int]("nGramsMax") action { (x,c) => c.copy(nGramsMax=x) }
    opt[Int]("commonFeaturesMin") action { (x,c) => c.copy(commonFeaturesMin=x) }
    opt[Int]("commonFeaturesMax") action { (x,c) => c.copy(commonFeaturesMax=x) }
    opt[Int]("numConfigs") action { (x,c) => c.copy(numConfigs=x) }
    opt[String]("execStyle") action { (x,c) => c.copy(execStyle=x) }
  }.parse(args, NewsgroupsConfig()).get

  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
   *
   * @param args
   */
  def main(args: Array[String]) = {
    val conf = new SparkConf().setAppName(appName)
    conf.setIfMissing("spark.master", "local[2]") // This is a fallback if things aren't set via spark submit.

    val sc = new SparkContext(conf)

    val appConfig = parse(args)
    run(sc, appConfig)

    sc.stop()
  }

}
