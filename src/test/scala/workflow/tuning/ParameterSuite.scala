package workflow.tuning

import org.scalatest.FunSuite
import pipelines.Logging


class ParameterSuite extends FunSuite with Logging {

  val searchSpace = ChoiceParameter("family",
    Seq(
      SeqParameter("RandomForest",
        Seq(
          DiscreteParameter("loss", Seq("gini","entropy")),
          ContinuousParameter("numFeatures", 0.0, 1.0),
          IntParameter("minSplit", 1, 20),
          IntParameter("nodeSize", 2, 50)
        )
      ),
      SeqParameter("LinSVM",
        Seq(
          ContinuousParameter("Reg", 1e-6, 1e6),
          ContinuousParameter("Step", 1e-6, 1e6))
      ),
      SeqParameter("RndSVM",
        Seq(
          ContinuousParameter("Reg", 1e-6, 1e6),
          ContinuousParameter("Step", 1e-6, 1e6),
          ContinuousParameter("gamma", 1e-4, 1e4),
          IntParameter("D", 1, 10)
        )
      )
    )
  )

  test("Continuous Parameter tests") {
    val param = ContinuousParameter("stuff", 1e-6, 1e6)
    for (i <- 0 until 100) {
      val p = param.sample()._2
      assert(p >= 1e-6 && p <= 1e6, "Continuous parameter must be in range.")
    }
  }

  test("Int Parameter tests") {
    val param = IntParameter("stuff", 1, 100)
    for (i <- 0 until 100) {
      val p = param.sample()._2
      assert(p >= 1 && p <= 100, "Int parameter must be in range.")
    }
  }

  test("Discrete Parameter tests") {
    val param = DiscreteParameter("stuff", Seq("A","B","C"))
    for (i <- 0 until 100) {
      val p = param.sample()._2
      assert(Set("A","B","C").contains(p), "Discrete parameter must be in range.")
    }

  }

  test("Deep parameter tests") {
    assert(Set("RandomForest", "LinSVM", "RndSVM").contains(searchSpace.sample()._2._1),
      "ChoiceParameter sample should be a valid string name")

    for (i <- 0 until 100) {
      val sample = searchSpace.sample()

      sample match {
        case ("family", ("RandomForest", x)) => {
          val params = x.asInstanceOf[Seq[(String, Any)]].toMap
          val loss = params("loss").asInstanceOf[String]
          val numFeatures = params("numFeatures").asInstanceOf[Double]
          val minSplit = params("minSplit").asInstanceOf[Int]
          val nodeSize = params("nodeSize").asInstanceOf[Int]

          assert(Set("gini","entropy").contains(loss), "Loss may be one of two values for RF")
          assert(numFeatures >= 0.0 && numFeatures <= 1.0, "numFeatures must be >=0 and <=1")
          assert(minSplit >= 1 && minSplit <=20, "minSplit must be >=1 and <=20")
          assert(nodeSize >=2 && nodeSize <=50, "nodeSize must be >=2 and <=20")
        }
        case ("family", ("LinSVM", x)) => {
          val params = x.asInstanceOf[Seq[(String, Any)]].toMap
          val reg = params("Reg").asInstanceOf[Double]
          val step = params("Step").asInstanceOf[Double]
          assert(reg >= 1e-6 && reg <= 1e6, "Reg must be >=1e-6 and <=1e6")
          assert(step >= 1e-6 && step <= 1e6, "Step must be >=1e-6 and <1e6")
        }
        case ("family", ("RndSVM", x)) => {
          val params = x.asInstanceOf[Seq[(String, Any)]].toMap
          val reg = params("Reg").asInstanceOf[Double]
          val step = params("Step").asInstanceOf[Double]
          val gamma = params("gamma").asInstanceOf[Double]
          val d = params("D").asInstanceOf[Int]

          assert(reg >= 1e-6 && reg <= 1e6, "Reg must be >=1e-6 and <=1e6")
          assert(step >= 1e-6 && step <= 1e6, "Step must be >=1e-6 and <1e6")
          assert(gamma >= 1e-4 && gamma <= 1e4, "Gamma must be >=1e-6 and <1e6")
          assert(d >= 1 && d <= 10, "Step must be >=1 and <10")
        }
      }
    }
  }


}