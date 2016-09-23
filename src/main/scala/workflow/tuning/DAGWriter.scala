package workflow.tuning


import workflow.{Pipeline, Profile}

import argonaut._
import Argonaut._

object DAGWriter {

  case class DAG(vertices: Map[String,Profile], edges: List[(String,String)], sinks: List[String])
  implicit def ProfileCodecJson = casecodec3(Profile.apply, Profile.unapply)("ns","rddMem","driverMem")
  implicit def DAGCodecJson = casecodec3(DAG.apply, DAG.unapply)("vertices","edges","sinks")

  def toDAG[A,B](pipe: Pipeline[A, B], prof: Map[Int, Profile]): DAG = {

    val graph = pipe.executor.graph

    //Produce a list of edges from the adjacency list.
    val edges = graph.dependencies.flatMap(m => m._2.map(s => (s,m._1))).toSeq ++
      graph.sinkDependencies.map(m => (m._2,m._1)).toSeq

    val vertices = prof.map(s => (s._1.toString, s._2))

    DAG(vertices, edges.map(s => (s._1.toString, s._2.toString)).toList, graph.sinks.toList.sortBy(_.id).map(_.toString))
  }

  def toJson[A,B](pipe: Pipeline[A,B], prof: Map[Int, Profile]): String = {
    toDAG(pipe, prof).asJson.spaces2
  }

  def toJson(profiles: Map[Int,Profile]): String = {
    profiles.map(x => (x._1.toString,x._2)).asJson.spaces2
  }

  def profilesFromJson(json: String): Map[Int, Profile] = {
    val res = json.decodeOption[Map[String,Profile]]
    res.get.map(x => (x._1.toInt,x._2))
  }
}