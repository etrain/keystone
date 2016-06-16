package workflow.tuning

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.scalatest.FunSuite
import pipelines.Logging
import workflow.{PipelineContext, Estimator, Transformer}

//Specify some basic pipeline components.
case class AddN(n: Int) extends Transformer[Double,Double] { def apply(in: Double) = in.toDouble + n}

case class SubOver(n: Double, m: Double) extends Transformer[Double,Double] { def apply(in: Double) = (in-n) / m}

case class Normalize(eps: Double) extends Estimator[Double,Double] {
  def fit(in: RDD[Double]): SubOver = {
    val sum = in.sum
    val sumsq = in.map(x => x*x).sum
    val n = in.count

    val mean = sum.toDouble/n
    val sd = math.sqrt(sumsq.toDouble/n - mean)

    SubOver(mean, sd+eps)
  }
}


class PipelineParameterSuite extends FunSuite with Logging with PipelineContext {
  test("Define some pipe parameters.") {
    sc = new SparkContext("local", "test")
    val dataset = sc.parallelize(Array[Double](100.0, 200.0))

    val parameterizedPipeline =
      TransformerP("adder",
        { x: Int => AddN(x) },
        IntParameter("n", 10, 1000)) andThen
      TransformerP("divider",
        { x: Seq[(String,Any)] => {
          val p = x.toMap
          SubOver(p("mu").asInstanceOf[Double], p("sigma").asInstanceOf[Double])
        } },
        SeqParameter("muSigma",
          Seq(
            ContinuousParameter("mu", 100.0, 1000.0),
            ContinuousParameter("sigma", 1.0, 100.0)))) andThen
        EstimatorP("subMean",
          { x: Double => Normalize(x) },
          ContinuousParameter("eps", 1e-6, 1e6, Scale.Log), dataset)

    val pipeSample = parameterizedPipeline.sample[Double,Double]()
    logInfo(pipeSample.executor.graph.toDOTString)

    logInfo(s"Pipe(RDD(100.0,200.0): ${pipeSample(dataset).get.collect.mkString(",")}")
    logInfo(s"Pepe(5.0): ${pipeSample(5.0).get}")
  }
}