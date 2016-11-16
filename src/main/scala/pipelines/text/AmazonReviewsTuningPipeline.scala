package pipelines.text

import breeze.linalg.SparseVector
import evaluation.{MulticlassClassifierEvaluator, BinaryClassifierEvaluator}
import loaders.{AmazonReviewsDataLoader, LabeledData}
import nodes.learning.{NaiveBayesEstimator, LogisticRegressionEstimator}
import nodes.nlp._
import nodes.stats.TermFrequency
import nodes.util.{MaxClassifier, CommonSparseFeatures}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import pipelines.Logging
import pipelines.text.NewsgroupsTuningPipeline._
import scopt.OptionParser
import workflow.{Transformer, AutoCacheRule, Pipeline}
import workflow.tuning._

object AmazonReviewsTuningPipeline extends Logging {
  val appName = "AmazonReviewsTuningPipeline"

  def run(sc: SparkContext, conf: AmazonReviewsTuningConfig): Unit = {
    val tf = TermFrequency()
    val ac = Transformer[Int,Boolean]((x: Int) => x > 0)

    val amazonTrainData = AmazonReviewsDataLoader(sc, conf.trainLocation, conf.threshold).labeledData
    val trainData = LabeledData(amazonTrainData.repartition(conf.numParts).cache())

    val training = trainData.data
    val labels = ac(trainData.labels)


    // Build the classifier estimator
    val pipeSpace = TransformerP("Trim", {x: Unit => Trim}, EmptyParameter()) andThen
      TransformerP("LowerCase", {x: Unit => LowerCase()}, EmptyParameter()) andThen
      TransformerP("Tokenizer", {x: Unit => Tokenizer()}, EmptyParameter()) andThen
      TransformerP("NGramsFeaturizer",
        {x:Int => NGramsFeaturizer(1 to x)},
        IntParameter("maxGrams", conf.nGramsMin, conf.nGramsMax)) andThen
      TransformerP("TermFrequency", {x:Unit => tf}, EmptyParameter()) andThen
      EstimatorP("CommonSparseFeatures",
        {x:Int => CommonSparseFeatures[Seq[String]](math.pow(10,x).toInt)},
        IntParameter("commonSparseFeatures", conf.commonFeaturesPowMin, conf.commonFeaturesPowMax),
        trainData.data) andThen
      LabelEstimatorP("NaiveBayesEstimator",
        {x:Int => LogisticRegressionEstimator[SparseVector[Double]](
          numClasses = 2,
          regParam = math.pow(10,x),
          numIters = conf.numIters)},
        IntParameter("lambda", conf.lambdaPowMin, conf.lambdaPowMax),
        trainData.data,
        trainData.labels) andThen
      TransformerP("AssignClass", {x: Unit => ac}, EmptyParameter())

    // Evaluate the classifier
    val pipes = (0 until conf.numConfigs).map(x => pipeSpace.sample[String, Boolean]())
    logInfo(s"Length of pipes: ${pipes.length}")



    val evaluator: (RDD[Boolean], RDD[Boolean]) => Double = BinaryClassifierEvaluator(_, _).accuracy

    val predictor = conf.execStyle match {
      case "gather" => Pipeline.gather(pipes)
      case "tunefull" => PipelineTuning.tune(pipes, training, labels, evaluator)
      case "sequential" => PipelineTuning.sequential(pipes, training, labels, evaluator)
    }

    logInfo("Beginning profiling")

    new AutoCacheRule(null).profileExtrapolateAndWrite(predictor.apply(training), s"./profile.amazon.${conf.numConfigs}.json", conf.profile)

    // Evaluate the classifier
    logInfo("Evaluating classifier")
  }

  case class AmazonReviewsTuningConfig(
    trainLocation: String = "",
    testLocation: String = "",
    threshold: Double = 3.5,
    nGramsMin: Int = 2,
    nGramsMax: Int = 4,
    commonFeaturesPowMin: Int = 4,
    commonFeaturesPowMax: Int = 6,
    lambdaPowMin: Int = -5,
    lambdaPowMax: Int = 5,
    numIters: Int = 20,
    numParts: Int = 512,
    numConfigs: Int = 1,
    execStyle: String = "gather",
    profile: Boolean = false)

  def parse(args: Array[String]): AmazonReviewsTuningConfig = new OptionParser[AmazonReviewsTuningConfig](appName) {
    head(appName, "0.1")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    opt[Double]("threshold") action { (x,c) => c.copy(threshold=x)}
    opt[Int]("nGramsMin") action { (x,c) => c.copy(nGramsMin=x) }
    opt[Int]("nGramsMax") action { (x,c) => c.copy(nGramsMax=x) }
    opt[Int]("commonFeaturesMin") action { (x,c) => c.copy(commonFeaturesPowMin=x) }
    opt[Int]("commonFeaturesMax") action { (x,c) => c.copy(commonFeaturesPowMax=x) }
    opt[Int]("lambdaPowMin") action { (x,c) => c.copy(lambdaPowMin=x) }
    opt[Int]("lambdaPowMax") action { (x,c) => c.copy(lambdaPowMax=x) }
    opt[Int]("numIters") action { (x,c) => c.copy(numIters=x) }
    opt[Int]("numParts") action { (x,c) => c.copy(numParts=x) }
    opt[Int]("numConfigs") action { (x,c) => c.copy(numConfigs=x) }
    opt[String]("execStyle") action { (x,c) => c.copy(execStyle=x) }
    opt[Boolean]("profile") action { (x,c) => c.copy(profile=x) }
  }.parse(args, AmazonReviewsTuningConfig()).get

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
