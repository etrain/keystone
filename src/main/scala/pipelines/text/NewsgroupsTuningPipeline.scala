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
import workflow._
import workflow.tuning._

object NewsgroupsTuningPipeline extends Logging {
  val appName = "NewsgroupsTuningPipeline"

  def run(sc: SparkContext, conf: NewsgroupsConfig): Unit = {

    val trainData = NewsgroupsDataLoader(sc, conf.trainLocation)
    val numClasses = NewsgroupsDataLoader.classes.length

    // Build the classifier estimator
    logInfo("Training classifier")
    def one(y: Double) = 1.0
    val tf = TermFrequency(one)

    val pipeSpace = TransformerP("Trim", {x: Unit => Trim}, EmptyParameter()) andThen
      TransformerP("LowerCase", {x: Unit => LowerCase()}, EmptyParameter()) andThen
      TransformerP("Tokenizer", {x: Unit => Tokenizer()}, EmptyParameter()) andThen
      TransformerP("NGramsFeaturizer",
        {x:Int => NGramsFeaturizer(1 to x)},
        IntParameter("maxGrams", conf.nGramsMin, conf.nGramsMax)) andThen
      TransformerP("TermFrequency", {x:Unit => tf}, EmptyParameter()) andThen
      EstimatorP("CommonSparseFeatures",
        {x:Int => CommonSparseFeatures[Seq[String]](math.pow(10,x).toInt)},
        IntParameter("commonSparseFeatures", conf.commonFeaturesMin, conf.commonFeaturesMax),
        trainData.data) andThen
      LabelEstimatorP("NaiveBayesEstimator",
        {x:Double => NaiveBayesEstimator[SparseVector[Double]](numClasses,x)},
        ContinuousParameter("lambda", conf.lambdaMin, conf.lambdaMax, scale=Scale.Log),
        trainData.data,
        trainData.labels) andThen
      TransformerP("MaxClassifier", {x:Unit => MaxClassifier}, EmptyParameter())

    val pipes = (0 until conf.numConfigs).map(x => pipeSpace.sample[String, Int]())
    logInfo(s"Length of pipes: ${pipes.length}")

    val evaluator: (RDD[Int], RDD[Int]) => Double = MulticlassClassifierEvaluator(_, _, numClasses).macroFScore()

    val predictor = conf.execStyle match {
      case "gather" => Pipeline.gather(pipes)
      case "tunefull" => PipelineTuning.tune(pipes, trainData.data, trainData.labels, evaluator)
      case "sequential" => PipelineTuning.sequential(pipes, trainData.data, trainData.labels, evaluator)
    }

    logInfo("Beginning profiling")

    new AutoCacheRule(null).profileExtrapolateAndWrite(predictor.apply(trainData.data), s"./profile.newsgroups20.${conf.numConfigs}.json", conf.profile)

    // Evaluate the classifier
    logInfo("Evaluating classifier")

    /*val testData = NewsgroupsDataLoader(sc, conf.testLocation)
    val testLabels = testData.labels
    val testResults = predictor(testData.data)
    val eval = MulticlassClassifierEvaluator(testResults.get, testLabels, numClasses)

    logInfo("\n" + eval.summary(NewsgroupsDataLoader.classes))*/

    Unit
  }

  case class NewsgroupsConfig(
    trainLocation: String = "",
    testLocation: String = "",
    nGramsMin: Int = 2,
    nGramsMax: Int = 4,
    commonFeaturesMin: Int = 3,
    commonFeaturesMax: Int = 5,
    lambdaMin: Double = 0.0,
    lambdaMax: Double = 1e4,
    numConfigs: Int = 10,
    execStyle: String = "gather",
    profile: Boolean = false)

  def parse(args: Array[String]): NewsgroupsConfig = new OptionParser[NewsgroupsConfig](appName) {
    head(appName, "0.1")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    opt[Int]("nGramsMin") action { (x,c) => c.copy(nGramsMin=x) }
    opt[Int]("nGramsMax") action { (x,c) => c.copy(nGramsMax=x) }
    opt[Int]("commonFeaturesMin") action { (x,c) => c.copy(commonFeaturesMin=x) }
    opt[Int]("commonFeaturesMax") action { (x,c) => c.copy(commonFeaturesMax=x) }
    opt[Double]("lambdaMin") action { (x,c) => c.copy(lambdaMin=x) }
    opt[Double]("lambdaMax") action { (x,c) => c.copy(lambdaMax=x) }
    opt[Int]("numConfigs") action { (x,c) => c.copy(numConfigs=x) }
    opt[String]("execStyle") action { (x,c) => c.copy(execStyle=x) }
    opt[Boolean]("profile") action { (x,c) => c.copy(profile=x)}
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
