package nodes.learning

import pipelines.Logging
import nodes.stats.{StandardScalerModel, StandardScaler}
import utils.MatrixUtils
import workflow.{LabelEstimator, Estimator}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import breeze.linalg._
import breeze.numerics._
import breeze.math._
import breeze.stats._
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS => BreezeLBFGS}
import breeze.optimize.{LBFGS => BreezeLBFGS}
import breeze.optimize.FirstOrderMinimizer

import edu.berkeley.cs.amplab.mlmatrix.util.{Utils => MLMatrixUtils}

import org.apache.spark.rdd.RDD

class SoftMaxGradient extends Serializable {

  def compute(
      data: DenseMatrix[Double],
      labels: DenseMatrix[Double],
      model: DenseMatrix[Double])
    : (DenseMatrix[Double], Double) = {

    val axb = data * model

    // find row-wise maxes
    val maxs = max(axb(*, ::))

    // subtract max from each column for numerical stability
    val axbMinusMaxs = axb(::, *) - maxs

    // Now compute exp(row) / sum(exp(row)) for each row
    exp.inPlace(axbMinusMaxs)
    val expRowSums = sum(axbMinusMaxs(*, ::))

    val softmax = axbMinusMaxs(::, *) / expRowSums
    val diff = softmax - labels
    val grad = data.t * (diff)

    // To compute loss we only need to get the softmax values
    // for where the labels is 1. 
    var i = 0
    var lossSum = 0.0
    while (i < labels.rows) {
      val softMaxForLabel = softmax(i, argmax(labels(i, ::)))
      lossSum += math.log(softMaxForLabel)
      i = i + 1
    }
    // val prod = labels :* softmax
    // val prodRowSum = sum(prod(*, ::)).toDenseVector
    // val loss = -1.0 * sum(log(prodRowSum))
    val loss = -1.0 * lossSum
    (grad, loss)
  }

}

object SoftMaxLBFGS {
  /**
  * Run Limited-memory BFGS (L-BFGS) in parallel.
  * Averaging the subgradients over different partitions is performed using one standard
  * spark map-reduce in each iteration.
  */
  def runLBFGS(
    trainingFeatures: RDD[DenseVector[Double]],
    trainingLabels: RDD[DenseVector[Double]],
    numCorrections: Int,
    convergenceTol: Double,
    numEpochs: Int,
    lambda: Double): DenseMatrix[Double] = {

    val sc = trainingLabels.context

    val lossHistory = mutable.ArrayBuilder.make[Double]

    val nTrain = trainingFeatures.count
    val numFeatures = trainingFeatures.first.length
    val numClasses = trainingLabels.first.length

    // TODO(shivaram): Should we make this caching optional ?
    val featuresMat = trainingFeatures.mapPartitions { part =>
      Iterator.single(MatrixUtils.rowsToMatrix(part))
    }.cache()

    val labelsMat = trainingLabels.mapPartitions { part =>
      Iterator.single(MatrixUtils.rowsToMatrix(part))
    }.cache()

    // matrix collapsed into a vector as breeze can't do
    // InnerProducts with matrices ?
    var model = DenseVector.zeros[Double](numFeatures * numClasses)

    val gradient = new SoftMaxGradient()

    val lbfgs = new BreezeLBFGS[DenseVector[Double]](numEpochs, numCorrections, convergenceTol)
    val costFun = new CostFun(featuresMat, labelsMat, gradient, lambda, nTrain, numFeatures, numClasses)
    val states = lbfgs.iterations(new CachedDiffFunction(costFun), model)

    // Run it once to initialize etc.
    var epoch = 0
    while (states.hasNext) {
      val epochBegin = System.nanoTime

      val state = states.next()

      val primalObjective = state.value
      lossHistory += primalObjective

      // extract model from state for this class
      model = state.x
      val modelMat = model.asDenseMatrix.reshape(numFeatures, numClasses)
      println("For epoch " + epoch + " value is " + state.value)
      println("For epoch " + epoch + " iter is " + state.iter)
      println("For epoch " + epoch + " grad norm is " + norm(state.grad))
      println("For epoch " + epoch + " searchFailed ? " + state.searchFailed)

      println("model is " + modelMat.rows + " x " + modelMat.cols)
      println("model 2-norm is " + norm(model))

      val epochTime = System.nanoTime - epochBegin
      println("EPOCH_" + epoch + "_time: " + epochTime)
      println("EPOCH_" + epoch + "_primalObjective: " + primalObjective)
      epoch = epoch + 1
    }
    val modelMat = model.asDenseMatrix.reshape(numFeatures, numClasses)
    modelMat
  } // End of runLBFGS

  /**
   * CostFun implements Breeze's DiffFunction[T], which returns the loss and gradient
   * at a particular point. It's used in Breeze's convex optimization routines.
   */
  private class CostFun(
    dataMat: RDD[DenseMatrix[Double]],
    labelsMat: RDD[DenseMatrix[Double]],
    gradient: SoftMaxGradient,
    lambda: Double,
    numExamples: Long,
    numFeatures: Int,
    numClasses: Int) extends DiffFunction[DenseVector[Double]] {

    override def calculate(model: DenseVector[Double]): (Double, DenseVector[Double]) = {
      val modelMat = model.asDenseMatrix.reshape(numFeatures, numClasses)
      // Have a local copy to avoid the serialization of CostFun object which is not serializable.
      val bcW = dataMat.context.broadcast(modelMat)
      val localGradient = gradient

      val (gradientSum, lossSum) = if (dataMat.context.isLocal) {
        dataMat.zip(labelsMat).map { x =>
            localGradient.compute(x._1, x._2, bcW.value)
          }.reduce( (a: (DenseMatrix[Double], Double), b: (DenseMatrix[Double], Double)) => {
            a._1 += b._1
            (a._1, a._2 + b._2)
          })
        } else {
          MLMatrixUtils.treeReduce(dataMat.zip(labelsMat).map { x =>
              localGradient.compute(x._1, x._2, bcW.value)
          }, (a: (DenseMatrix[Double], Double), b: (DenseMatrix[Double], Double)) => {
            a._1 += b._1
            (a._1, a._2 + b._2)
          })
        }

      val regularizedGrad = gradientSum / numExamples.toDouble + modelMat * lambda
      val regularizedLoss = lossSum / numExamples.toDouble + (lambda/2.0) * math.pow(norm(model), 2)
      println("Gradient NORM is " + norm(regularizedGrad.toDenseVector))
      println("loss SUM is " + regularizedLoss)
      bcW.destroy()
      (regularizedLoss, regularizedGrad.toDenseVector)
    }
  }
}

class DenseLogisticRegressionEstimator(numFeatures: Int,
    numCorrections: Int = 10,
    convergenceTol: Double = 1e-8,
    numEpochs: Int = 5,
    lambda: Double = 0.0) extends LabelEstimator[DenseVector[Double], DenseVector[Double], DenseVector[Double]] with Logging{

  def fit(in: RDD[DenseVector[Double]], labels: RDD[DenseVector[Double]]): LinearMapper[DenseVector[Double]] = {

    val labelScaler = new StandardScaler(normalizeStdDev = false).fit(labels)
    val featureScaler = new StandardScaler(normalizeStdDev = false).fit(in)

    val inScaled = featureScaler.apply(in)
    val labelsScaled = labelScaler.apply(labels)

    val model = SoftMaxLBFGS.runLBFGS(inScaled, labelsScaled, numCorrections, convergenceTol, numEpochs, lambda)

    new LinearMapper(model, Some(labelScaler.mean), Some(featureScaler))
  }

}
