package nodes.stats

import breeze.linalg._
import org.scalatest.{Matchers, FunSuite}
import pipelines.Logging

class RandomSignNodeSuite extends FunSuite with Logging with Matchers {

  test("RandomSignNode") {
    val signs = DenseVector(1.0, -1.0, 1.0)
    val node = RandomSignNode(signs)
    val data: DenseVector[Double] = DenseVector(1.0, 2.0, 3.0)
    val result = node(data)
    Seq(result) should equal (Seq(DenseVector(1.0, -2.0, 3.0)))
  }

  test("RandomSignNode.create") {
    val node = RandomSignNode(1000)
    
    node.signs.foreach(elt => assert(elt == -1.0 || elt == 1.0))
  }
}
