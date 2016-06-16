package workflow.tuning

import org.apache.spark.rdd.RDD
import workflow._

trait PipelineParameter extends PPChainable {

  def toParameterPipeline: ParameterPipeline = {
    new ParameterPipeline(Seq(this))
  }
}

case class TransformerP[PT,I,O](name: String, x: PT => Transformer[I,O], p: Parameter[PT]) extends PipelineParameter
case class EstimatorP[PT,I,O](name: String, x: PT => Estimator[I,O], p: Parameter[PT], data: RDD[I]) extends PipelineParameter
case class LabelEstimatorP[PT,I,O,L](name: String, x: PT => LabelEstimator[I,O,L], p: Parameter[PT], data: RDD[I], labels: RDD[L]) extends PipelineParameter

trait ConcreteChainable
case class TransformerChainable[A,B](t: Transformer[A,B]) extends ConcreteChainable
case class EstimatorChainable[A,B](e: Estimator[A,B], data: RDD[A]) extends ConcreteChainable
case class LabelEstimatorChainable[A,B,L](e: LabelEstimator[A,B,L], data: RDD[A], labels: RDD[L]) extends ConcreteChainable

class ParameterPipeline(val nodes: Seq[PipelineParameter]) extends PPChainable { //note: for now we assume linear pipelines made of transformers, estimators, labelestimators.
  def toParameterPipeline = this

  private def samplePipelineParameter(p: PipelineParameter): ConcreteChainable = {
    p match {
      case t: TransformerP[_,_,_] => TransformerChainable(t.x(t.p.sample._2))
      case e: EstimatorP[_,_,_] => EstimatorChainable(e.x(e.p.sample._2), e.data)
      case le: LabelEstimatorP[_,_,_,_] => LabelEstimatorChainable(le.x(le.p.sample._2), le.data, le.labels)
    }
  }

  private def samplePipelineParameter[PT,I,O](p: TransformerP[PT,I,O]): TransformerChainable[I,O] = {
    TransformerChainable(p.x(p.p.sample._2))
  }

  private def samplePipelineParameter[PT,I,O](p: EstimatorP[PT,I,O]): EstimatorChainable[I,O] = {
    EstimatorChainable(p.x(p.p.sample._2), p.data)
  }

  private def samplePipelineParameter[PT,I,O,L](p: LabelEstimatorP[PT,I,O,L]): LabelEstimatorChainable[I,O,L] = {
    LabelEstimatorChainable(p.x(p.p.sample._2), p.data, p.labels)
  }

  def sample[A,B](): Pipeline[A,B] = {
    nodes.map(samplePipelineParameter).foldLeft(Identity().toPipeline.asInstanceOf[Pipeline[Any,Any]]) {
      case (a,b) =>
        b match {
          case i: TransformerChainable[Any,Any] => a andThen i.t
          case e: EstimatorChainable[Any,Any] => a andThen (e.e, e.data)
          case le: LabelEstimatorChainable[Any,Any,Any] => a andThen (le.e, le.data, le.labels)
        }
    }.asInstanceOf[Pipeline[A,B]]

  }
  def grid(n: Int): Iterator[Pipeline[_,_]] = ???
}


trait PPChainable {
  def toParameterPipeline: ParameterPipeline

  def andThen(p: PipelineParameter): ParameterPipeline = {
    new ParameterPipeline(this.toParameterPipeline.nodes :+ p)
  }
}
