package pipelines.images.lbl

import nodes.images.{GrayScaler, ImageVectorizer}
import nodes.learning.{ZCAWhitenerEstimator, ZCAWhitener}
import nodes.loaders.BlockedImageLoader
import nodes.util.MaxClassifier
import org.apache.spark.{SparkConf, SparkContext}
import pipelines.Logging
import scopt.OptionParser


object MaterialsClustering extends Logging {
  val appName = "MaterialsClustering"
  case class MaterialsClusteringConfig(
    dataLocation: String = "",
    nCenters: Int = 0,
    nSamples: Int = 1e6.toInt,
    maxIters: Int = 20)

  def run(sc: SparkContext, config: MaterialsClusteringConfig) = {
    val trainData = BlockedImageLoader(sc, config.dataLocation, (100, 100, 2)).cache()

    logInfo(s"Got some blocks: ${trainData.map(_._1).collect.mkString(",")}")

    // Our training features are the featurizer applied to our training data.
    /*val assignmentPipeline = {Convolver3D() then
      ImageVectorizer thenEstimator
      new ZCAWhitenerEstimator() thenEstimator
      KMeansEstimator(config.nCenters, config.maxIters) then
      MaxClassifier}.fit(trainData)

    logInfo(s"Clusters assigned!")

    assignmentPipeline*/
  }

  def parse(args: Array[String]): MaterialsClusteringConfig = new OptionParser[MaterialsClusteringConfig](appName) {
    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(dataLocation=x) }
    opt[Int]("nCenters") required() action { (x,c) => c.copy(nCenters=x) }
    opt[Int]("nSamples") action { (x,c) => c.copy(nSamples=x) }
    opt[Int]("maxIters") action { (x,c) => c.copy(maxIters=x) }
  }.parse(args, MaterialsClusteringConfig()).get

  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
   * @param args
   */
  def main(args: Array[String]) = {
    val appConfig = parse(args)

    val conf = new SparkConf().setAppName(appName)
    conf.setIfMissing("spark.master", "local[2]") // This is a fallback if things aren't set via spark submit.
    val sc = new SparkContext(conf)
    run(sc, appConfig)

    sc.stop()
  }

}
