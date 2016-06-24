package workflow.tuning

object Scale extends Enumeration {
  type Scale = Value
  val Linear = Value("Linear")
  val Log = Value("Log")
}

import Scale._
import scala.util.Random


abstract class Parameter[+T](val name: String) {
  def sample: (String, T)
  def range(n: Int): Seq[(String, T)]
}

case class ContinuousParameter(
    override val name: String,
    min: Double,
    max: Double,
    scale: Scale = Scale.Linear)
  extends Parameter[Double](name) {

  def sample() = {
    scale match {
      case Scale.Log => (name, SearcherUtils.randRangeLog(min, max))
      case Scale.Linear => (name, SearcherUtils.randRange(min, max))
    }
  }

  def range(n: Int) = scale match {
    case Scale.Log => SearcherUtils.logSpace(min, max, n).map(x => (name, x))
    case Scale.Linear => SearcherUtils.linSpace(min, max, n).map(x => (name, x))
  }
}

case class IntParameter(override val name: String, min: Int, max: Int) extends Parameter[Int](name) {
  def sample() = (name, (min + SearcherUtils.rand.nextInt(max-min)))
  def range(n: Int) = (min to max by math.max(1, ((max-min)/(n.toDouble-1)).round.toInt)).map(x => (name, x))
}

case class DiscreteParameter(override val name: String, params: Seq[String]) extends Parameter[String](name) {
  def sample() = (name, params(SearcherUtils.rand.nextInt(params.length)))
  def range(n: Int) = SearcherUtils.rand.shuffle(params).take(n).map(x => (name, x))
}

case class ChoiceParameter(override val name: String, params: Seq[Parameter[Any]]) extends Parameter[(String, Any)](name) {
  def sample() = (name, SearcherUtils.rand.shuffle(params).head.sample)
  def range(n: Int) = {
    val s = params.length
    params.flatMap(_.range(n/s)).map(x => (name, x))
  }
}

case class SeqParameter(override val name: String, params: Seq[Parameter[Any]]) extends Parameter[Seq[(String,Any)]](name) {
  def sample(): (String, Seq[(String, Any)]) = (name, params.map(_.sample))
  def range(n: Int): Seq[(String,Seq[(String,Any)])] = {
    val ns = math.pow(n, 1.0/params.length).toInt
    SearcherUtils.cartesian(params.map(_.range(ns).toList).toList).map(p => (name, p))
  }
}

object EmptyParameter {
  def apply(): IntParameter = IntParameter("empty", 0, 1)
}

object SearcherUtils {
  val rand = new Random(42)

  def randRange(min: Double, max: Double) = min + (max - min)*rand.nextDouble()

  def randRangeLog(min: Double, max: Double) = math.exp(randRange(math.log(min), math.log(max)))
  def randRangeInt(min: Int, max: Int): Int = min + rand.nextInt(max-min)

  def linSpace(from: Double, until: Double, points: Int) = from to until by (until-from)/(points.toDouble-1)
  def logSpace(from: Double, until: Double, points: Int) = {
    linSpace(math.log10(from),math.log10(until),points).map(math.pow(10.0, _))
  }

  def cartesian[T](xss: List[List[T]]): List[List[T]] = xss match {
    case Nil => List(Nil)
    case h :: t => for(xh <- h; xt <- cartesian(t)) yield xh :: xt
  }
}

object Parameters {
  /**
    * Helper to serialize to JSON for loading in python. Format of output object may change.
    * @param p Parameter to serialize - generally the full search space.
    * @tparam T Inferred.
    * @return JSON Serialized version of parameter.
    */
  def serialize[T](p: Parameter[T]): String = {
    p match {
      case c: ChoiceParameter => "{" + c.params.map(a => "\""+a.name+"\""+":"+serialize(a)).mkString(",") + "}"
      case s: SeqParameter => "[" + s.params.map(a => serialize(a)).mkString(",") + "]"
      case i: IntParameter => "[" + Seq("\"int\"", "\""+i.name+"\"", i.min, i.max).mkString(",") + "]"
      case c: ContinuousParameter => "[" + Seq("\"continuous\"","\""+c.name+"\"", c.min, c.max, "\""+c.scale.toString+"\"").mkString(",") + "]"
      case d: DiscreteParameter => "[" + d.params.map("\""+_+"\"").mkString(",") + "]"
    }
  }
}
