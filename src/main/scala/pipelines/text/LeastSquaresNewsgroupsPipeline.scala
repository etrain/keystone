package pipelines.text

import evaluation.MulticlassClassifierEvaluator
import loaders.NewsgroupsDataLoader
import nodes.learning.{NaiveBayesEstimator, SparseLBFGSwithL2, LeastSquaresSparseGradient, LogisticRegressionLBFGSEstimator}
import nodes.nlp._
import nodes.stats.TermFrequency
import nodes.util.{CommonSparseFeatures, MaxClassifier, Cacher}
import org.apache.spark.{SparkConf, SparkContext}
import pipelines.Logging
import scopt.OptionParser
import workflow.Optimizer
import workflow.Transformer

object LeastSquaresNewsgroupsPipeline extends Logging {
  val appName = "LeastSquaresNewsgroupsPipeline"

  def run(sc: SparkContext, conf: NewsgroupsConfig) {

    val trainData = NewsgroupsDataLoader(sc, conf.trainLocation)
    val numClasses = NewsgroupsDataLoader.classes.length

    // Build the classifier estimator
    logInfo("Training classifier")
    val predictorPipeline = Trim andThen LowerCase() andThen
        Tokenizer() andThen
        NGramsFeaturizer(1 to conf.nGrams) andThen
        TermFrequency(x => 1) andThen
        (CommonSparseFeatures(conf.commonFeatures), trainData.data) andThen
        new Cacher andThen
        (new SparseLBFGSwithL2(new LeastSquaresSparseGradient(numClasses), true, numClasses, numIterations=20), trainData.data, trainData.labels) andThen
        MaxClassifier
//       (LogisticRegressionLBFGSEstimator(numClasses, numIters = 20), trainData.data, trainData.labels) andThen
//       Transformer(_.toInt)

    val predictor = Optimizer.execute(predictorPipeline)
    logInfo("\n" + predictor.toDOTString)

    // Evaluate the classifier
    logInfo("Evaluating classifier")

    val testData = NewsgroupsDataLoader(sc, conf.testLocation)
    val testLabels = testData.labels
    val testResults = predictor(testData.data)
    val eval = MulticlassClassifierEvaluator(testResults, testLabels, numClasses)

    logInfo("\n" + eval.summary(NewsgroupsDataLoader.classes))
  }

  case class NewsgroupsConfig(
    trainLocation: String = "",
    testLocation: String = "",
    nGrams: Int = 2,
    commonFeatures: Int = 100000)

  def parse(args: Array[String]): NewsgroupsConfig = new OptionParser[NewsgroupsConfig](appName) {
    head(appName, "0.1")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    opt[Int]("nGrams") action { (x,c) => c.copy(nGrams=x) }
    opt[Int]("commonFeatures") action { (x,c) => c.copy(commonFeatures=x) }
  }.parse(args, NewsgroupsConfig()).get

  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
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
