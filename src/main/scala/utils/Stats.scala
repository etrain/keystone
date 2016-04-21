package utils

import java.util.{Random => JRandom}

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import breeze.stats.distributions._
import nodes.util.TopKClassifier
import org.apache.spark.rdd.RDD

object Stats extends Serializable {
  /**
    * Margin to use for comparing numerical values.
    */
  val thresh = 1e-8

  /**
    * Compares two numbers for approximate equality
    * @param a A number.
    * @param b A second number.
    * @param thresh equality threshold
    * @return True if the two numbers are within `thresh` of each other.
    */
  def aboutEq(a: Double, b: Double, thresh: Double): Boolean = {
    math.abs(a-b) < thresh
  }
  
  def aboutEq(a: Double, b: Double): Boolean = {
    Stats.aboutEq(a,b,thresh)
  }

  /**
    * Compares two arrays for approximate equality. Modify margin by setting Stats.thresh.
    *
    * @param as A array of numbers.
    * @param bs A second array of numbers.
    * @param thresh equality threshold
    * @return True if the two numbers are within `thresh` of each other.
    */
  def aboutEq(as: DenseVector[Double], bs: DenseVector[Double], thresh: Double): Boolean = {
    require(as.length == bs.length, "Vectors must be the same size")
    abs(as-bs).toArray.forall(_ < thresh)
  }
  
  def aboutEq(as: DenseVector[Double], bs: DenseVector[Double]): Boolean = {
    Stats.aboutEq(as,bs,thresh)
  }

  /**
    * Compares two matrices for approximate equality.
    *
    * @param a A matrix.
    * @param b A second matrix.
    * @param thresh equality threshold
    * @return True iff all corresponding elements of the two matrices are within `thresh` of each other.
    */
  def aboutEq(a: DenseMatrix[Double], b: DenseMatrix[Double], thresh: Double): Boolean = {
    require(a.rows == b.rows && a.cols == b.cols, "Matrices must be the same size.")

    abs(a-b).toArray.forall(_ < thresh)
  }

  def aboutEq(a: DenseMatrix[Double], b: DenseMatrix[Double]): Boolean = {
    Stats.aboutEq(a,b,thresh)
  }

  /**
   * Computes top-k classification error based on a distribution of predictions.
   *
   * @param predictions Distribution of predictions.
   * @param actuals Distribution of actuals.
   * @param k How many classes to include.
   * @return The error percent.
   */
  def classificationError(predictions: RDD[DenseVector[Double]], actuals: RDD[DenseVector[Double]], k: Int = 1) = {
    val counts = predictions.count
    Stats.getErrPercent((TopKClassifier(k))(predictions), (TopKClassifier(k))(actuals), counts)
  }

  /**
   * Computes error percent based on array of predicted labels and array of actual labels.
   *
   * @param predicted The predicted labels.
   * @param actual The actual labels.
   * @param num Total number of examples.
   * @return Error percent.
   */
  def getErrPercent(predicted: RDD[Array[Int]], actual: RDD[Array[Int]], num: Long): Double = {
    val totalErr = predicted.zip(actual).map({ case (topKLabels, actualLabel) =>
      // TODO: this only works for examples that have a single label.
      // Things that have multiple labels require an intersection?
      if (topKLabels.contains(actualLabel(0))) {
        0.0
      } else {
        1.0
      }
    }).reduce(_ + _)

    val errPercent = totalErr / num.toDouble * 100.0
    errPercent
  }

  /**
   * Given a local matrix, compute the mean and variance of each row.
   * Subtract the row mean from each row and divide by the sqrt(variance + alpha).
   *
   * @param mat Input matrix.
   * @param alpha Alpha for the denominator.
   * @return Normalized Matrix.
   */
  def normalizeRows(mat: DenseMatrix[Double], alpha: Double = 1.0): DenseMatrix[Double] = {
    // FIXME: This currently must convert the matrices to double due to breeze implicits
    // TODO: Could optimize, use way fewer copies
    //val rowMeans: DenseVector[Double] = mean(mat(*, ::)).map(x => if (x.isNaN) 0 else x)
    //val variances: DenseVector[Double] = sum((mat(::, *) - rowMeans) :^= 2.0, Axis._1) :/= (mat.cols.toDouble - 1.0)
    //val sds: DenseVector[Double] = sqrt(variances + alpha.toDouble).map(x => if (x.isNaN) math.sqrt(alpha) else x)


    val counts = mat.cols.toDouble
    val sums = DenseVector.zeros[Double](mat.rows)
    val sumssq = DenseVector.zeros[Double](mat.rows)
    val out = DenseMatrix.zeros[Double](mat.rows, mat.cols)
    //val means = DenseVector.zeros[Double](mat.rows)
    //val sds = DenseVector.zeros[Double](mat.rows)

    var r,c = 0
    while(c < mat.cols) {
      r=0
      while(r < mat.rows) {
        sums(r)+=mat(r,c)
        sumssq(r)+=mat(r,c)*mat(r,c)
        r+=1
      }
      c+=1
    }

    r = 0
    while(r < mat.rows) {
      sumssq(r) = math.sqrt(alpha + (sumssq(r) - sums(r)*sums(r)/counts)/(counts-1))
      sums(r) = sums(r)/counts
      r+=1
    }

    c = 0
    while(c < mat.cols) {
      r=0
      while(r < mat.rows) {
        out(r,c) = (mat(r,c)-sums(r))/sumssq(r)
        r+=1
      }
      c+=1
    }

    //val out = mat(::, *) - rowMeans
    //out(::, *) /= sds

    out
  }
}
