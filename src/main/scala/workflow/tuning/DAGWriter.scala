package workflow.tuning


import pipelines.Logging
import workflow._

import argonaut._
import Argonaut._

object DAGWriter extends Logging {

  case class DAG(vertices: Map[String,Profile], edges: List[(String,String)], sinks: List[String], weights: Map[String, Int], names: Map[String, String])
  implicit def ProfileCodecJson = casecodec3(Profile.apply, Profile.unapply)("ns","rddMem","driverMem")
  implicit def DAGCodecJson = casecodec5(DAG.apply, DAG.unapply)("vertices","edges","sinks", "weights", "names")

  def getNames(graph: Graph): Map[String, String] = {
    graph.nodes.map(n => (n.toString, graph.getOperator(n).toString)).toMap
  }

  def toDAG[A,B](pipe: Pipeline[A, B], prof: Map[Int, Profile]): DAG = {

    val graph = pipe.executor.graph

    //Produce a list of edges from the adjacency list.
    val edges = graph.dependencies.flatMap(m => m._2.map(s => (s,m._1))).toSeq ++
      graph.sinkDependencies.map(m => (m._2,m._1)).toSeq

    val vertices = prof.map(s => (s._1.toString, s._2))

    val weights = prof.map(s => (s._1.toString, 1))

    //val names = graph.nodes.map(n => graph.getOperator

    DAG(vertices, edges.map(s => (s._1.toString, s._2.toString)).toList, graph.sinks.toList.sortBy(_.id).map(_.toString), weights, getNames(graph))
  }

  def toJson[A,B](pipe: Pipeline[A,B], prof: Map[Int, Profile]): String = {
    toDAG(pipe, prof).asJson.spaces2
  }

  def toDAG(graph: Graph, prof: Map[NodeId, Profile], weights: Map[NodeId, Int]): DAG = {
    //Produce a list of edges from the adjacency list.
    val edges = graph.dependencies.toSeq.flatMap { case (id, deps) => deps.map(s => (s,id)) } ++
      graph.sinkDependencies.toSeq.map { case (id1, id2) => (id2, id1) }

    logInfo(s"Size of edge list: ${edges.length}")


    //val vertices = prof.map {case (i, p) => (i.toString, p)}
    val vertices = graph.nodes.map(n => (n.toString, prof.getOrElse(n, Profile(0,0,0)))).toMap


    val graphWeights = weights.map {case (i, w) => (i.toString, w) }

    DAG(vertices, edges.map(s => (s._1.toString, s._2.toString)).toList, graph.sinks.toList.sortBy(_.id).map(_.toString), graphWeights, getNames(graph))

  }

  def toJson(graph: Graph, prof: Map[NodeId, Profile], weights: Map[NodeId, Int]): String = {
    toDAG(graph, prof, weights).asJson.spaces2
  }

  def toJson(profiles: Map[Int,Profile]): String = {
    profiles.map(x => (x._1.toString,x._2)).asJson.spaces2
  }

  def profilesFromJson(json: String): Map[Int, Profile] = {
    val res = json.decodeOption[Map[String,Profile]]
    res.get.map(x => (x._1.toInt,x._2))
  }
}