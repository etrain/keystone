package loaders

import breeze.linalg.{DenseVector, DenseMatrix}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import org.tensorflow.example.feature._

case class MultiLabeledFeatures(labels: Array[Int], features: DenseMatrix[Byte])

object Youtube8MFramesLoader {
  def makeVector(in: Feature): Array[Byte] = {
    val byteArray = in.getBytesList.value(0).toByteArray
    byteArray
  }

  def makeMultiLabeledFeatures(in: SequenceExample): MultiLabeledFeatures = {
    val labels = in.getContext.feature("labels").getInt64List.value.map(_.toInt).toArray
    val featureList: Seq[Feature] = in.getFeatureLists.featureList("inc3").feature
    val features: DenseMatrix[Byte] = DenseMatrix(featureList.map(makeVector).toSeq:_*)

    MultiLabeledFeatures(labels, features)
  }

  def apply(sc: SparkContext, filename: String, numParts: Int): RDD[MultiLabeledFeatures] = {
    TFRecordLoader(sc, filename, numParts, SequenceExample.parseFrom).map(makeMultiLabeledFeatures)
  }
}