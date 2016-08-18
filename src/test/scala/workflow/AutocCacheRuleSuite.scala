package workflow

import nodes.util.Cacher
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.scalatest.FunSuite
import pipelines.Logging
import workflow.AutoCacheRule.{AggressiveCache, GreedyCache}

case class TransformerPlus(plus: Int) extends Transformer[Int, Int] {
  override def apply(in: Int): Int = {
    in + plus
  }
}

class AutoCacheRuleSuite extends FunSuite with PipelineContext with Logging {

  def estimator = new Estimator[Int, Int] with WeightedOperator {
    override def fit(data: RDD[Int]): Transformer[Int, Int] = {
      val sums = data.collect()
      TransformerPlus(sums.sum)
    }

    override val weight: Int = 4
  }

  def getPlan(sc: SparkContext): Graph = {
    val train = sc.parallelize(Seq(1, 2, 3, 4, 5, 6, 7, 8))
    Graph(
      sources = Set(SourceId(0)),
      operators = Map(
        NodeId(0) -> DatasetOperator(train),
        NodeId(1) -> TransformerPlus(1),
        NodeId(2) -> TransformerPlus(2),
        NodeId(3) -> TransformerPlus(3),
        NodeId(4) -> TransformerPlus(4),
        NodeId(5) -> TransformerPlus(5),
        NodeId(6) -> estimator,
        NodeId(7) -> new DelegatingOperator,
        NodeId(8) -> TransformerPlus(8),
        NodeId(9) -> TransformerPlus(9),
        NodeId(10) -> TransformerPlus(10),
        NodeId(11) -> TransformerPlus(11),
        NodeId(12) -> TransformerPlus(12)
      ),
      dependencies = Map(
        NodeId(0) -> Seq(),
        NodeId(1) -> Seq(NodeId(0)),
        NodeId(2) -> Seq(NodeId(1)),
        NodeId(3) -> Seq(NodeId(2)),
        NodeId(4) -> Seq(NodeId(2)),
        NodeId(5) -> Seq(NodeId(3), NodeId(4)),
        NodeId(6) -> Seq(NodeId(5)),
        NodeId(7) -> Seq(NodeId(6), NodeId(12)),
        NodeId(8) -> Seq(SourceId(0)),
        NodeId(9) -> Seq(NodeId(8)),
        NodeId(10) -> Seq(NodeId(9)),
        NodeId(11) -> Seq(NodeId(9)),
        NodeId(12) -> Seq(NodeId(10), NodeId(11))
      ),
      sinkDependencies = Map(SinkId(0) -> NodeId(7))
    )
  }

  val profiles = Map[NodeId, Profile](
    NodeId(0) -> Profile(10, Long.MaxValue, 0),
    NodeId(1) -> Profile(10, 50, 0),
    NodeId(2) -> Profile(30, 200, 0),
    NodeId(3) -> Profile(20, 1000, 0),
    NodeId(4) -> Profile(20, 1000, 0),
    NodeId(5) -> Profile(20, 100, 0)
  )

  val profiles2 = Map[NodeId, Profile](
    NodeId(0) -> Profile(10, Long.MaxValue, 0),
    NodeId(1) -> Profile(10, 50, 0),
    NodeId(2) -> Profile(30, 200, 0),
    NodeId(3) -> Profile(20, 1000, 0),
    NodeId(4) -> Profile(20, 1000, 0),
    NodeId(5) -> Profile(20, 100, 0),
    NodeId(6) -> Profile(10, 1000, 0),
    NodeId(7) -> Profile(10, 50, 0),
    NodeId(8) -> Profile(30, 200, 0),
    NodeId(9) -> Profile(20, 1000, 0),
    NodeId(10) -> Profile(20, 1000, 0),
    NodeId(11) -> Profile(20, 100, 0),
    NodeId(12) -> Profile(20, 100, 0)
  )

  test("End to end aggressive AutoCacheRule") {
    sc = new SparkContext("local", "test")
    val optimizer = new Optimizer {
      protected val batches: Seq[Batch] = Batch("Auto Cache", Once, new AutoCacheRule(GreedyCache())) :: Nil
    }
    PipelineEnv.getOrCreate.setOptimizer(optimizer)
    val pipe = new Pipeline[Int, Int](new GraphExecutor(getPlan(sc)), SourceId(0), SinkId(0))

    assert(pipe.apply(5).get() === 168)
  }

  test("End to end greedy AutoCacheRule") {
    sc = new SparkContext("local", "test")
    val optimizer = new Optimizer {
      protected val batches: Seq[Batch] = Batch("Auto Cache", Once, new AutoCacheRule(AggressiveCache)) :: Nil
    }
    PipelineEnv.getOrCreate.setOptimizer(optimizer)
    val pipe = new Pipeline[Int, Int](new GraphExecutor(getPlan(sc)), SourceId(0), SinkId(0))

    assert(pipe.apply(5).get() === 168)
  }

  test("Aggressive cacher") {
    sc = new SparkContext("local", "test")

    val autoCacheRule = new AutoCacheRule(null)
    val plan = getPlan(sc)
    val cachedInstructions = autoCacheRule.aggressiveCache(plan)

    val cachedTransformers = cachedInstructions.operators.collect {
      case (cacheNode, _: Cacher[_]) => cachedInstructions.getOperator(
        cachedInstructions.getDependencies(cacheNode).head.asInstanceOf[NodeId])
    }.toSet

    assert(cachedTransformers === Set(TransformerPlus(2), TransformerPlus(5)))
  }

  test("Greedy cacher, max mem 10") {
    sc = new SparkContext("local", "test")

    val autoCacheRule = new AutoCacheRule(null)
    val cachedInstructions = autoCacheRule.greedyCache(getPlan(sc), profiles, Some(10))

    val cachedTransformers = cachedInstructions.operators.collect {
      case (cacheNode, _: Cacher[_]) => cachedInstructions.getOperator(
        cachedInstructions.getDependencies(cacheNode).head.asInstanceOf[NodeId])
    }.toSet

    assert(cachedTransformers === Set())
  }

  test("Greedy cacher, max mem 75") {
    sc = new SparkContext("local", "test")

    val autoCacheRule = new AutoCacheRule(null)
    val cachedInstructions = autoCacheRule.greedyCache(getPlan(sc), profiles, Some(75))

    val cachedTransformers = cachedInstructions.operators.collect {
      case (cacheNode, _: Cacher[_]) => cachedInstructions.getOperator(
        cachedInstructions.getDependencies(cacheNode).head.asInstanceOf[NodeId])
    }.toSet

    assert(cachedTransformers === Set(TransformerPlus(1)))
  }

  test("Greedy cacher, max mem 125") {
    sc = new SparkContext("local", "test")

    val autoCacheRule = new AutoCacheRule(null)
    val cachedInstructions = autoCacheRule.greedyCache(getPlan(sc), profiles, Some(125))

    val cachedTransformers = cachedInstructions.operators.collect {
      case (cacheNode, _: Cacher[_]) => cachedInstructions.getOperator(
        cachedInstructions.getDependencies(cacheNode).head.asInstanceOf[NodeId])
    }.toSet

    assert(cachedTransformers === Set(TransformerPlus(5)))
  }

  test("Greedy cacher, max mem 175") {
    sc = new SparkContext("local", "test")

    val autoCacheRule = new AutoCacheRule(null)
    val cachedInstructions = autoCacheRule.greedyCache(getPlan(sc), profiles, Some(175))

    val cachedTransformers = cachedInstructions.operators.collect {
      case (cacheNode, _: Cacher[_]) => cachedInstructions.getOperator(
        cachedInstructions.getDependencies(cacheNode).head.asInstanceOf[NodeId])
    }.toSet

    assert(cachedTransformers === Set(TransformerPlus(1), TransformerPlus(5)))
  }

  test("Greedy cacher, max mem 350") {
    sc = new SparkContext("local", "test")

    val autoCacheRule = new AutoCacheRule(null)
    val cachedInstructions = autoCacheRule.greedyCache(getPlan(sc), profiles, Some(350))

    val cachedTransformers = cachedInstructions.operators.collect {
      case (cacheNode, _: Cacher[_]) => cachedInstructions.getOperator(
        cachedInstructions.getDependencies(cacheNode).head.asInstanceOf[NodeId])
    }.toSet

    assert(cachedTransformers === Set(TransformerPlus(2), TransformerPlus(5)))
  }

  test("Greedy cacher, max mem 10000") {
    sc = new SparkContext("local", "test")

    val autoCacheRule = new AutoCacheRule(null)
    val cachedInstructions = autoCacheRule.greedyCache(getPlan(sc), profiles, Some(10000))

    val cachedTransformers = cachedInstructions.operators.collect {
      case (cacheNode, _: Cacher[_]) => cachedInstructions.getOperator(
        cachedInstructions.getDependencies(cacheNode).head.asInstanceOf[NodeId])
    }.toSet

    assert(cachedTransformers === Set(TransformerPlus(2), TransformerPlus(5)))
  }

  test("Does the OPT hit rate estimator work?") {
    sc = new SparkContext("local", "test")

    val newprofs = profiles2.map(x => (x._1.asInstanceOf[GraphId], x._2))
    logInfo(s"Cache Size,optRes,lruRes")
    for (size <- Seq(100,500,600,700,800,900,1000,5000,10000)) {
      val optRes = CacheUtils.calculatePipelineHitRate(new GraphExecutor(getPlan(sc)), size, newprofs, OPTCache.apply)
      val lruRes = CacheUtils.calculatePipelineHitRate(new GraphExecutor(getPlan(sc)), size, newprofs, LRUCache.apply)
      logInfo(s"$size,$optRes,$lruRes")
    }


  }
}
