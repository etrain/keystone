package workflow

import pipelines.Logging

import scala.collection.mutable

abstract class CacheCalculator(cacheSizeMb: Long, sizesMb: Map[GraphId, Long]) extends Logging {
  val cache = mutable.Set[GraphId]()

  def hitRateMb(in: Seq[GraphId]): Long = {
    //Todo - if we cared about performance we'd use a min-heap to manage the cache.

    logDebug("---")
    logDebug(s"Starting")


    var hits = 0L
    for (i <- 0 until in.length) {
      var thisObj = in(i)
      //If object is in cache
      logDebug(s"Reading: $thisObj, Cache: ${cache.mkString(",")}")
      if (cache contains thisObj) {
        logDebug(s"Found $thisObj ")
        hits+=sizesMb(thisObj)
      } else {
        //Add this to the cache.
        logDebug(s"Cache is size ${cacheSize()}, adding $thisObj, size: ${sizesMb(thisObj)}")

        if(cacheSizeMb >= sizesMb(thisObj)) {
          add(i, thisObj)
          cache.add(thisObj)
        }

        while(cacheSizeMb < cacheSize()) {
          val evictId = evict(i, in)
          logDebug(s"Cache is size $cacheSize, removing $evictId, size ${sizesMb(evictId)}")
          cache.remove(evictId)
        }
      }
      touch(i, thisObj)
    }
    logDebug("---\n")

    //Return number bytes read from cache..
    hits
  }

  def cacheSize() = cache.toSeq.map(sizesMb).sum

  def evict(step: Int, in: Seq[GraphId]): GraphId
  def add(i: Int, thisObj: GraphId): Unit
  def touch(i: Int, thisObj: GraphId): Unit
}

case class LRUCache(cacheSizeMb: Int, sizesMb: Map[GraphId, Long])
    extends CacheCalculator(cacheSizeMb, sizesMb) with Logging {

  val usage = mutable.Map[GraphId,Int]()

  def evict(i: Int, in: Seq[GraphId]): GraphId = {
    val lruItem = usage.minBy(_._2)._1
    usage.remove(lruItem)
    lruItem
  }

  def add(i: Int, thisObj: GraphId) = {
    usage(thisObj) = i
  }

  def touch(i: Int, thisObj: GraphId) = {
    if (usage contains thisObj) {
      usage(thisObj) = i
    }
  }

}

case class OPTCache(cacheSizeMb: Int, sizesMb: Map[GraphId,Long]) extends CacheCalculator(cacheSizeMb, sizesMb) {

  def evict(i: Int, in: Seq[GraphId]) = {
    val future = in.slice(i+1, in.length)

    //build a map of the future.
    var futureUses = mutable.Map[GraphId, Int]()
    for(i <- 0 until future.length) {
      if(!(futureUses contains future(i))) {
        futureUses(future(i)) = i
      }
    }

    val unusedInFuture = cache diff futureUses.keySet

    val itemToRemove = if(unusedInFuture.isEmpty) {
      futureUses.filterKeys(cache.contains).maxBy(_._2)._1
    } else {
      unusedInFuture.head
    }
    itemToRemove
  }

  def add(i: Int, thisObj: GraphId) = Unit
  def touch(i: Int, thisObj: GraphId) = Unit
}


object CacheUtils extends Logging {
  def getNodeWeights(graph: Graph): Map[NodeId, Int] = {
    graph.operators.mapValues {
      case op: WeightedOperator => op.weight
      case _ => 1
    }
  }

  def calculatePipelineHitRate(
    graphExecutor: GraphExecutor,
    cacheSizeMb: Int,
    profiles: Map[GraphId,Profile],
    cacheMaker: (Int, Map[GraphId,Long]) => CacheCalculator): Long = {
    //Given a profile,

    val graphWeights = getNodeWeights(graphExecutor.graph)

    val reads: Seq[NodeId] = AnalysisUtils.linearize(graphExecutor.graph).flatMap { n =>
      n match {
        case nd: NodeId => {
          //Get only the Node dependencies, since we are assuming all inputs are cached.
          val deps = graphExecutor.graph.dependencies(nd).flatMap(i => i match {
            case x: NodeId => Seq(x)
            case _ => Seq[NodeId]()
          })
          (0 until graphWeights(nd)).flatMap(_ => deps)
        }
        case s: SourceId => Seq[NodeId]()
        case si: SinkId => Seq[NodeId]()
      }
    }

    logDebug(s"Reads: ${reads.mkString(",")}")

    cacheMaker(cacheSizeMb, profiles.mapValues(_.rddMem)).hitRateMb(reads)
  }
}
