package nodes.learning

import breeze.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.classification.{LogisticRegressionModel => MLlibLRM}
import org.apache.spark.mllib.linalg.{Vector => MLlibVector}
import org.apache.spark.mllib.optimization.{SquaredL2Updater, LogisticGradient, LBFGS}
import org.apache.spark.mllib.regression.{GeneralizedLinearAlgorithm, LabeledPoint}
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.rdd.RDD
import utils.MLlibUtils.breezeVectorToMLlib
import workflow.{LabelEstimator, Transformer}

import scala.reflect.ClassTag

/**
 * A Multilabel Logistic Regression model that transforms feature vectors to vectors containing
 * the logistic regression output of the different classes
 */

case class MultiLabelLogisticRegressionModel[T <: Vector[Double]](model: MLlibLRM)
    extends Transformer[T, DenseVector[Double]] {

  /**
    * Yields probabilities over the classes using MLlib's predictPoint as a starting point.
    */
  override def apply(in: T): DenseVector[Double] = {
    predictPoint(breezeVectorToMLlib(in), model.weights, model.intercept)
  }

  /**
    * Assumes that the model is on greater than 2 classes.
    * @param dataMatrix
    * @param weightMatrix
    * @param intercept
    * @return
    */
  def predictPoint(
      dataMatrix: MLlibVector,
      weightMatrix: MLlibVector,
      intercept: Double): DenseVector[Double] = {
    require(dataMatrix.size == model.numFeatures)

    /**
      * Compute and find the one with maximum margins. If the maxMargin is negative, then the
      * prediction result will be the first class.
      *
      * PS, if you want to compute the probabilities for each outcome instead of the outcome
      * with maximum probability, remember to subtract the maxMargin from margins if maxMargin
      * is positive to prevent overflow.
      */

    val dataWithBiasSize = model.weights.size / (model.numClasses - 1)
    val weightsArray = model.weights.toArray
    val withBias = dataMatrix.size + 1 == dataWithBiasSize
    val margins = 0.0 +: (0 until model.numClasses - 1).map { i =>
      var margin = 0.0
      dataMatrix.foreachActive { (index, value) =>
        if (value != 0.0) margin += value * weightsArray((i * dataWithBiasSize) + index)
      }
      // Intercept is required to be added into margin.
      if (withBias) {
        margin += weightsArray((i * dataWithBiasSize) + dataMatrix.size)
      }
      margin
    }.toArray
    def logit(x: Double) = 1.0 / (1.0 + math.exp(-x))

    new DenseVector(margins.map(logit))
  }
}

/**
 * A LabelEstimator which learns a Logistic Regression model from training data.
 * Currently does so using LBFG-S
 *
 * @param numClasses The number of classes
 * @param numIters The max number of iterations to use. Default 100
 * @param convergenceTol Set the convergence tolerance of iterations for the optimizer. Default 1E-4.
 */
case class MultiLabelLogisticRegressionEstimator[T <: Vector[Double] : ClassTag](
    numClasses: Int,
    regParam: Double = 0,
    numIters: Int = 100,
    convergenceTol: Double = 1E-4,
    numFeatures: Int = -1,
    cache: Boolean = false
  ) extends LabelEstimator[T, DenseVector[Double], Int] {

  /**
   * Train a classification model for Multinomial/Binary Logistic Regression using
   * Limited-memory BFGS. Standard feature scaling and L2 regularization are used by default.
   * NOTE: Labels used in Logistic Regression should be {0, 1, ..., k - 1}
   * for k classes multi-label classification problem.
   */
  private[this] class LogisticRegressionWithLBFGS(numClasses: Int, numFeaturesValue: Int)
      extends GeneralizedLinearAlgorithm[MLlibLRM] with Serializable {

    this.numFeatures = numFeaturesValue
    override val optimizer = new LBFGS(new LogisticGradient, new SquaredL2Updater)

    override protected val validators = List(multiLabelValidator)

    require(numClasses > 1)
    numOfLinearPredictor = numClasses - 1
    if (numClasses > 2) {
      optimizer.setGradient(new LogisticGradient(numClasses))
    }

    private def multiLabelValidator: RDD[LabeledPoint] => Boolean = { data =>
      if (numOfLinearPredictor > 1) {
        DataValidators.multiLabelValidator(numOfLinearPredictor + 1)(data)
      } else {
        DataValidators.binaryLabelValidator(data)
      }
    }

    override protected def createModel(weights: MLlibVector, intercept: Double) = {
      if (numOfLinearPredictor == 1) {
        new MLlibLRM(weights, intercept)
      } else {
        new MLlibLRM(weights, intercept, numFeatures, numOfLinearPredictor + 1)
      }
    }
  }

  override def fit(in: RDD[T], labels: RDD[Int]): MultiLabelLogisticRegressionModel[T] = {
    val labeledPoints = if (cache) {
      labels.zip(in).map(x => LabeledPoint(x._1, breezeVectorToMLlib(x._2))).cache()
    } else {
      labels.zip(in).map(x => LabeledPoint(x._1, breezeVectorToMLlib(x._2)))
    }

    val trainer = new LogisticRegressionWithLBFGS(numClasses, numFeatures)
    trainer.setValidateData(false).optimizer.setNumIterations(numIters).setRegParam(regParam)
    val model = trainer.run(labeledPoints)

    new MultiLabelLogisticRegressionModel(model)
  }
}