package pipelines.tabular

import breeze.linalg.{SparseVector, DenseVector}
import evaluation.BinaryClassifierEvaluator
import nodes.learning.LeastSquaresEstimator
import nodes.stats.TermFrequency
import org.apache.spark.rdd.RDD
import nodes.util.{VectorCombiner, MaxClassifier, CommonSparseFeatures, ClassLabelIndicatorsFromIntLabels}
import nodes.nlp.{NGramsHashingTF, NGramsFeaturizer, Tokenizer, LowerCase}
import workflow.{Pipeline, Transformer}

object TitanicSurvivors {

  case class Survivor(
      survival: Boolean,
      pclass: Int,
      name: String,
      sex: Boolean,
      age: Double,
      sibsp: String,
      parch: String,
      ticket: Int,
      fare: Int,
      cabin: Int,
      embarked: String)

  //Assume we've got a function to load the base data.
  val trainData: RDD[Survivor] = ???

  //Extract the labels.
  val trainLabels = trainData.map(_.survival)
  val labelExtractor = Transformer((x: Boolean) => if (x) 1 else 0) andThen
    ClassLabelIndicatorsFromIntLabels(2)

  //For simplicity let's assume we use only a few columns.
  //Each "Extractor" is a Pipeline[Survivor,DenseVector[Double]]

  //One-hot encode the pclass.
  val pclassExtractor = Transformer((x: Survivor) => x.pclass) andThen
    ClassLabelIndicatorsFromIntLabels(3)

  //Compute 50 most frequent unigrams in names.
  val nameExtractor = Transformer((x: Survivor) => x.name) andThen
    LowerCase() andThen
    LowerCase() andThen
    Tokenizer() andThen
    NGramsFeaturizer(Seq(1)) andThen
    TermFrequency(x => 1) andThen
    (CommonSparseFeatures[Seq[String]](50), trainData) andThen
    Transformer((x: SparseVector[Double]) => x.toDenseVector)

  //Map the age to a double.
  val ageExtractor = Transformer((x: Survivor) => x.age) andThen
    Transformer((x: Double) => DenseVector[Double](x))

  //Concatenate the results.
  val featurePipeline = Pipeline.gather(Seq(pclassExtractor, nameExtractor, ageExtractor)) andThen
    VectorCombiner[Double]

  //The prediction pipeline is features fed into a learning model.
  val predictor = featurePipeline andThen
    (new LeastSquaresEstimator[DenseVector[Double]](), trainData, labelExtractor(trainLabels)) andThen
    MaxClassifier andThen
    Transformer((x: Int) => x > 0)


  val trainEval = BinaryClassifierEvaluator(predictor(trainData).get, trainLabels)

}

