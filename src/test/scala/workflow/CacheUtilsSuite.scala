package workflow

import org.scalatest.FunSuite
import pipelines.Logging

case class CachePolicyResult(
  builder: (Int, Map[GraphId,Long]) => CacheCalculator,
  cacheSize: Int,
  reads: Seq[GraphId],
  sizes: Map[GraphId,Long],
  hitRateMb: Int)

class CacheUtilsSuite extends FunSuite with Logging {

  def makeId(x: Int): GraphId = NodeId(x.toLong)
  def makeMaps(x: Seq[(Int,Int)]) = x.map(i => (makeId(i._1), i._2.toLong)).toMap

  val read1 = Seq(1,2,3,4,1,2,3,4).map(makeId)

  val sizes1 = makeMaps(Seq((1,1), (2,2), (3,3), (4,4)))
  val sizes2 = makeMaps(Seq((1,1), (2,1), (3,1), (4,1)))

  test("LRU basic tests") {
    val tests = Seq(
      CachePolicyResult(LRUCache.apply, 3, read1, sizes1, 0),
      CachePolicyResult(LRUCache.apply, 6, read1, sizes1, 0),
      CachePolicyResult(LRUCache.apply, 8, read1, sizes1, 0),
      CachePolicyResult(LRUCache.apply, 10, read1, sizes1, 10),
      CachePolicyResult(LRUCache.apply, 3, read1, sizes2, 0),
      CachePolicyResult(LRUCache.apply, 4, read1, sizes2, 4)
    )

    for (t <- tests) {
      logDebug(s"Running ${t}")
      val res = t.builder(t.cacheSize, t.sizes).hitRateMb(t.reads)
      logDebug(s"Got $res, expected ${t.hitRateMb}")
      assert(res == t.hitRateMb)
    }
  }

  test("OPT basic tests") {
    val tests = Seq(
      CachePolicyResult(OPTCache.apply, 3, read1, sizes1, 3),
      CachePolicyResult(OPTCache.apply, 6, read1, sizes1, 6),
      CachePolicyResult(OPTCache.apply, 8, read1, sizes1, 6), //Todo - think about why this isn't 7?
      CachePolicyResult(OPTCache.apply, 10, read1, sizes1, 10),
      CachePolicyResult(OPTCache.apply, 3, read1, sizes2, 3),
      CachePolicyResult(OPTCache.apply, 4, read1, sizes2, 4)
    )

    for (t <- tests) {
      logDebug(s"Running ${t}")
      val res = t.builder(t.cacheSize, t.sizes).hitRateMb(t.reads)
      logDebug(s"Got $res, expected ${t.hitRateMb}")
      assert(res == t.hitRateMb)
    }
  }
}