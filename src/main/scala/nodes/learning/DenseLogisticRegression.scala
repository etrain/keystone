package nodes.learning

import pipelines.Logging
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

class SoftMaxLBFGS(
    val blockSize: Int,
    val numCorrections: Int  = 10,
    val convergenceTol: Double = 1e-8,
    val numEpochs: Int = 5,
    val lambda: Double) extends Serializable {

  def fit(
    trainingFeatures: RDD[DenseVector[Double]],
    trainingLabels: RDD[DenseVector[Double]],
    intermediateCallback: Option[(Seq[DenseMatrix[Double]], DenseVector[Double], Int) => (Double, Double)])
    : (Seq[DenseMatrix[Double]], DenseVector[Double]) = {

    SoftMaxLBFGS.runLBFGS(
      blockSize,
      trainingFeatures,
      trainingLabels,
      numCorrections,
      convergenceTol,
      numEpochs,
      lambda,
      intermediateCallback)
  }

}

object SoftMaxLBFGS {
  /**
  * Run Limited-memory BFGS (L-BFGS) in parallel.
  * Averaging the subgradients over different partitions is performed using one standard
  * spark map-reduce in each iteration.
  */
  def runLBFGS(
    blockSize: Int,
    trainingFeatures: RDD[DenseVector[Double]],
    trainingLabels: RDD[DenseVector[Double]],
    numCorrections: Int,
    convergenceTol: Double,
    numEpochs: Int,
    lambda: Double,
    intermediateCallback: Option[(Seq[DenseMatrix[Double]], DenseVector[Double], Int) => (Double, Double)] = None)
    : (Seq[DenseMatrix[Double]], DenseVector[Double]) = {

    val sc = trainingLabels.context

    val lossHistory = mutable.ArrayBuilder.make[Double]

    val nTrain = trainingFeatures.count
    val numFeatures = trainingFeatures.first.length
    val numClasses = trainingLabels.first.length

    val featureMean = computePopFeatureMean(trainingFeatures, nTrain, numFeatures)

    // Pay some extra cost and compute featuresMat from DISK
    // This is useful because we can avoid doing rowsToMatrix every iteration
    trainingFeatures.unpersist()
    val featuresMat = zeroMeanFeatures(trainingLabels.zip(trainingFeatures).mapPartitions { part =>
      Iterator.single(MatrixUtils.rowsToMatrix(part.map(_._2)))
    }, featureMean).cache()
    featuresMat.count()

    val labelsMat = trainingLabels.mapPartitions { part =>
      Iterator.single(MatrixUtils.rowsToMatrix(part))
    }
    // val labelMean = labelsMat.map { mat =>
    //   sum(mat(::, *)).toDenseVector
    // }.reduce( (a: DenseVector[Double], b: DenseVector[Double]) => a += b ) /= nTrain.toDouble
    // val labelsZM = zeroMeanLabels(labelsMat, labelMean)
    // labelsZM.cache()
    // labelsZM.count

    // import lbfgs._
    val numBlocks = math.ceil(numFeatures.toDouble / blockSize).toInt
    var modelSplit= (0 until numBlocks).map { block =>
      // NOTE: This assumes uniform block sizes here.
      // W check the number of columns in the first pass and fix it if reqd.
      if (block == numBlocks - 1) {
        DenseMatrix.zeros[Double](numFeatures - (blockSize * (numBlocks - 1)), numClasses)
      } else {
        DenseMatrix.zeros[Double](blockSize, numClasses)
      }
    }.toArray
    var intercept = DenseVector.zeros[Double](numClasses)
    
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

      var epochTime = 0L

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

      val intercept = -1.0 * modelMat.t * featureMean

      //need to split model here because intermediatecall back requires Seq
      (0 until numBlocks).map { blockNum =>
        val end = math.min(numFeatures, (blockNum + 1) * blockSize)
        modelSplit(blockNum) := modelMat(blockNum*blockSize until end, ::)
      }
      
      epochTime = System.nanoTime - epochBegin
      println("EPOCH_" + epoch + "_time: " + epochTime)
      println("EPOCH_" + epoch + "_primalObjective: " + primalObjective)
      intermediateCallback.foreach { fn =>
        fn(modelSplit, intercept, epoch)
      }
      epoch = epoch + 1
    }
      
    (modelSplit, intercept)
  } //End of runLBFGS

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

  /** 
   * Computes the mean of each colum of the feature matrix
   * Returns a d-dimensional vector representing the feature mean
   */
  def computePopFeatureMean(
    features: RDD[DenseVector[Double]],
    nTrain: Long,
    nFeatures: Int): DenseVector[Double] = {

    // To compute the column means, compute the colSum in each partition, add it
    // up and then divide by number of rows.
    features.fold(DenseVector.zeros[Double](nFeatures))((a: DenseVector[Double], b: DenseVector[Double]) =>
      a += b
    ) /= nTrain.toDouble
  }

  def zeroMeanFeatures(
      featuresMat: RDD[DenseMatrix[Double]],
      featureMean: DenseVector[Double]): RDD[DenseMatrix[Double]] = {
    featuresMat.map(x => x(*, ::) - featureMean)
  }
  
  def zeroMeanLabels(
      labelsMat: RDD[DenseMatrix[Double]],
      labelMean: DenseVector[Double]): RDD[DenseMatrix[Double]] = {
    labelsMat.map(x => x(*,::)-labelMean)
  }
}

class DenseLogisticRegressionEstimator(numFeatures: Int,
    numCorrections: Int = 10,
    convergenceTol: Double = 1e-8,
    numEpochs: Int = 5,
    lambda: Double = 0.0) extends LabelEstimator[DenseVector[Double], DenseVector[Double], DenseVector[Double]] with Logging{

  def fit(in: RDD[DenseVector[Double]], labels: RDD[DenseVector[Double]]): LinearMapper[DenseVector[Double]] = {

    val (x, b) = SoftMaxLBFGS.runLBFGS(numFeatures, in, labels, numCorrections, convergenceTol, numEpochs, lambda, None)

    logWarning("We are assuming that the input features are one block. Fix this code before running on more features")
    new LinearMapper(x.head, Some(b))
  }

}
