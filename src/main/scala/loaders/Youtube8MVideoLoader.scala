package loaders

import breeze.linalg.{DenseVector, DenseMatrix}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import org.tensorflow.example.feature._

case class MultiLabeledFeatureVector(labels: Array[Int], features: DenseVector[Float])

object Youtube8MVideoLoader {
  val NUM_CLASSES = 4800

  def makeMultiLabeledFeatureVector(in: Example): MultiLabeledFeatureVector = {
    val labels = in.getFeatures.feature("labels").getInt64List.value.map(_.toInt).toArray
    val incFeature: Feature = in.getFeatures.feature("mean_inc3")
    val features: DenseVector[Float] = DenseVector(incFeature.getFloatList.value.map(_.toFloat).toArray)

    MultiLabeledFeatureVector(labels, features)
  }

  def apply(sc: SparkContext, filename: String, numParts: Int): RDD[MultiLabeledFeatureVector] = {
    TFRecordLoader(sc, filename, numParts, Example.parseFrom).map(makeMultiLabeledFeatureVector).repartition(numParts)
  }
}