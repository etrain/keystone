package pipelines.text

import breeze.linalg.{DenseVector, SparseVector}
import breeze.stats.distributions.{ThreadLocalRandomGenerator, RandBasis, CauchyDistribution}
import evaluation.MulticlassClassifierEvaluator
import loaders.{TimitFeaturesDataLoader, NewsgroupsDataLoader}
import nodes.learning.{LeastSquaresDenseGradient, DenseLBFGSwithL2, LeastSquaresEstimator, NaiveBayesEstimator}
import nodes.nlp._
import nodes.stats.{CosineRandomFeatures, TermFrequency}
import nodes.util.{ClassLabelIndicatorsFromIntLabels, CommonSparseFeatures, MaxClassifier}
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import pipelines.Logging
import scopt.OptionParser
import workflow._
import workflow.tuning._

object TimitTuningPipeline extends Logging {
  val appName = "TimitTuningPipeline"

  def run(sc: SparkContext, conf: TimitTuningConfig): Unit = {
    val seed = 123L
    val random = new java.util.Random(seed)
    val randomSource = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(random.nextLong())))

    // Load the data
    val timitFeaturesData = TimitFeaturesDataLoader(
      sc,
      conf.trainDataLocation,
      conf.trainLabelsLocation,
      conf.testDataLocation,
      conf.testLabelsLocation,
      conf.numParts)

    // Build the pipeline
    val trainData = timitFeaturesData.train.data.cache().setName("trainRaw")
    trainData.count()

    val labels = timitFeaturesData.train.labels.cache().setName("labels")

    // Build the classifier estimator
    logInfo("Training classifier")
    def one(y: Double) = 1.0
    val cauchy = new CauchyDistribution(0, 1)


    val numCosineFeatures = 16384
    val numFeatures = TimitFeaturesDataLoader.timitDimension
    val numClasses = TimitFeaturesDataLoader.numClasses

    val pipeSpace = TransformerP("RandomFeatures",
        {x:Seq[(String,Any)] => CosineRandomFeatures(
          numFeatures,
          numCosineFeatures,
          math.pow(5.5, x(1)._2.asInstanceOf[Int]),
          if(x(0)._2.asInstanceOf[Int] == 1) cauchy else randomSource.gaussian,
          randomSource.uniform)},
      SeqParameter("typeAndGamma", Seq(IntParameter("type", 0, 1), IntParameter("gammaPow",conf.gammaMinPow, conf.gammaMaxPow)))) andThen
      LabelEstimatorP("LinearSolver",
        {x:Double => new DenseLBFGSwithL2[DenseVector[Double]](new LeastSquaresDenseGradient, regParam = x, numIterations = 20)},
        ContinuousParameter("lambda", conf.lambdaMin, conf.lambdaMax, scale=Scale.Log),
        trainData,
        ClassLabelIndicatorsFromIntLabels(numClasses).apply(labels))
      TransformerP("MaxClassifier", {x:Unit => MaxClassifier}, EmptyParameter())

    val pipes = (0 until conf.numConfigs).map(x => pipeSpace.sample[DenseVector[Double], Int]())
    logInfo(s"Length of pipes: ${pipes.length}")

    val evaluator: (RDD[Int], RDD[Int]) => Double = MulticlassClassifierEvaluator(_, _, numClasses).macroFScore()

    val predictor = conf.execStyle match {
      case "gather" => Pipeline.gather(pipes)
      case "tunefull" => PipelineTuning.tune(pipes, trainData, labels, evaluator)
      case "sequential" => PipelineTuning.sequential(pipes, trainData, labels, evaluator)
    }

    logInfo("Beginning profiling")

    new AutoCacheRule(null).profileExtrapolateAndWrite(predictor.apply(trainData), s"./profile.timit.${conf.numConfigs}.json", conf.profile)

    // Evaluate the classifier
    logInfo("Evaluating classifier")

    /*val testData = NewsgroupsDataLoader(sc, conf.testLocation)
    val testLabels = testData.labels
    val testResults = predictor(testData.data)
    val eval = MulticlassClassifierEvaluator(testResults.get, testLabels, numClasses)

    logInfo("\n" + eval.summary(NewsgroupsDataLoader.classes))*/

    Unit
  }

  case class TimitTuningConfig(
    trainDataLocation: String = "",
    trainLabelsLocation: String = "",
    testDataLocation: String = "",
    testLabelsLocation: String = "",
    numParts: Int = 512,
    numCosines: Int = 50,
    gammaMinPow: Int = -4,
    gammaMaxPow: Int = 4,
    lambdaMin: Double = 0.0,
    lambdaMax: Double = 1e5,
    numEpochs: Int = 5,
    checkpointDir: Option[String] = None,
    numConfigs: Int = 10,
    execStyle: String = "gather",
    profile: Boolean = false)

  object Distributions extends Enumeration {
    type Distributions = Value
    val Gaussian, Cauchy = Value
  }

  def parse(args: Array[String]): TimitTuningConfig = new OptionParser[TimitTuningConfig](appName) {
    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[String]("trainDataLocation") required() action { (x,c) => c.copy(trainDataLocation=x) }
    opt[String]("trainLabelsLocation") required() action { (x,c) => c.copy(trainLabelsLocation=x) }
    opt[String]("testDataLocation") required() action { (x,c) => c.copy(testDataLocation=x) }
    opt[String]("testLabelsLocation") required() action { (x,c) => c.copy(testLabelsLocation=x) }
    opt[String]("checkpointDir") action { (x,c) => c.copy(checkpointDir=Some(x)) }
    opt[Int]("numParts") action { (x,c) => c.copy(numParts=x) }
    opt[Int]("numCosines") action { (x,c) => c.copy(numCosines=x) }
    opt[Int]("numEpochs") action { (x,c) => c.copy(numEpochs=x) }
    opt[Int]("gammaMinPow") action { (x,c) => c.copy(gammaMinPow=x) }
    opt[Int]("gammaMaxPow") action { (x,c) => c.copy(gammaMaxPow=x) }
    opt[Double]("lambdaMin") action { (x,c) => c.copy(lambdaMin=x) }
    opt[Double]("lambdaMax") action { (x,c) => c.copy(lambdaMax=x) }
    opt[Int]("numConfigs") action { (x,c) => c.copy(numConfigs=x) }
    opt[String]("execStyle") action { (x,c) => c.copy(execStyle=x) }
    opt[Boolean]("profile") action { (x,c) => c.copy(profile=x)}
  }.parse(args, TimitTuningConfig()).get


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
