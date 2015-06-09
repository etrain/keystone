package nodes.learning

import breeze.linalg._
import breeze.numerics._
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import pipelines.{Estimator, Transformer}
import utils.MatrixUtils

/**
 * Assigns a point to centers estimated by a KMeans model.
 * @param centers
 */
class KMeansModel(centers: DenseMatrix[Double]) extends Transformer[DenseVector[Double], DenseVector[Double]] {

  /**
   * Computes the (negative) distance between the input point and each centroid.
   * The reason this returns the negative distance is that it can be easily passed into
   * the MaxClassifier for cluster assignment.
   *
   * @param in  The input item to pass into this transformer
   * @return  The output value
   */
  def apply(in: DenseVector[Double]): DenseVector[Double] = {
    val sqDiff = pow(centers(::,*) - in, 2.0)
    val dist = sqrt(sum(sqDiff(::,*)))
    dist.toDenseVector
  }
}

class KMeansEstimator(nCenters: Int, maxIters: Int) extends Estimator[DenseVector[Double], DenseVector[Double]] {
  def fit(trainingFeatures: RDD[DenseVector[Double]]): KMeansModel = {

    val model = KMeans.train(trainingFeatures.map(v => Vectors.dense(v.toArray)), nCenters, maxIters)
    val modelMat = MatrixUtils.rowsToMatrix(model.clusterCenters.map(v => DenseVector(v.toArray))).t
    new KMeansModel(modelMat)
  }
}