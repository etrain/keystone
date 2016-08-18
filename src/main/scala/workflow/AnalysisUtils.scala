package workflow

private[workflow] object AnalysisUtils {

  /**
   * Given a graph and a source/sink/node, output the set of all sources/sinks/nodes
   * that directly depend on the source/sink/node.
   *
   * i.e. All ids with a direct dependency on this source/sink/node.
   *
   * @param graph The graph to use
   * @param id The node/sink/source in the graph
   * @return The set of all children of that id
   */
  def getChildren(graph: Graph, id: GraphId): Set[GraphId] = {
    id match {
      case id: NodeOrSourceId => {
        // Note: this is O(VE) and could be done faster with explicit bookkeeping in
        // the Graph class pointing in both directions if this becomes an issue.
        val childrenNodes = graph.dependencies.filter(_._2.contains(id)).keySet
        val childrenSinks = graph.sinkDependencies.filter(_._2 == id).keySet
        childrenNodes ++ childrenSinks
      }
      case sinkId: SinkId => Set()
    }
  }

  /**
   * Given a graph and a source/sink/node, output the set of all sources/sinks/nodes
   * that have that source/sink/node as an ancestor.
   *
   * i.e. All ids that explicitly or implicitly depend on this source/sink/node.
   *
   * @param graph The graph to use
   * @param id The node/sink/source in the graph
   * @return The set of all descendents of that id
   */
  def getDescendants(graph: Graph, id: GraphId): Set[GraphId] = {
    val children = getChildren(graph, id)
    children.map {
      child => getDescendants(graph, child) + child
    }.fold(Set())(_ union _)
  }

  /**
   * Given a graph and a source/sink/node, output the set of all nodes
   * and sources that are parents of that source/sink/node in the graph.
   *
   * i.e. All ids that source/sink/node has as a dependency
   *
   * @param graph The graph to use
   * @param id A node/sink/source in the graph
   * @return The set of all parents of that id
   */
  def getParents(graph: Graph, id: GraphId): Set[NodeOrSourceId] = {
    id match {
      case source: SourceId => Set()
      case node: NodeId => graph.getDependencies(node).toSet
      case sink: SinkId => Set(graph.getSinkDependency(sink))
    }
  }

  /**
   * Given a graph and a source/sink/node, output the set of all nodes
   * and sources that are ancestors of that source/sink/node in the graph.
   *
   * i.e. All ids that source/sink/node has explicit or implicit dependencies on.
   *
   * @param graph The graph to use
   * @param id A node/sink/source in the graph
   * @return The set of all ancestors of that id
   */
  def getAncestors(graph: Graph, id: GraphId): Set[NodeOrSourceId] = {
    val parents = getParents(graph, id)
    parents.map {
      parent => getAncestors(graph, parent) + parent
    }.fold(Set())(_ union _)
  }

  /**
   * Given a graph and a source/sink/node, output all ancestors of the source/sink/node
   * in sorted topological order.
   *
   * @param graph The graph to use
   * @param id A node/sink/source in the graph
   * @return The ancestors of that id in sorted topological order
   */
  def linearize(graph: Graph, id: GraphId): Seq[GraphId] = {
    val deps: Seq[GraphId] = id match {
      case source: SourceId => Seq()
      case node: NodeId => graph.getDependencies(node)
      case sink: SinkId => Seq(graph.getSinkDependency(sink))
    }

    deps.foldLeft(Seq[GraphId]()) {
      case (linearization, dep) => if (!linearization.contains(dep)) {
        linearization ++ linearize(graph, dep).filter(id => !linearization.contains(id)) :+ dep
      } else {
        linearization
      }
    }
  }

  /**
   * Deterministically return a topological sorting of all sources/sinks/nodes in a graph
   *
   * @param graph  The graph to use
   * @return  A topologically sorted ordering of all sources/sinks/nodes in the graph
   */
  def linearize(graph: Graph): Seq[GraphId] = {
    // Sort sinks by id to ensure a deterministic final ordering
    val sortedSinks = graph.sinks.toSeq.sortBy(_.id)

    sortedSinks.foldLeft(Seq[GraphId]()) {
      case (linearization, sink) => {
        // We add the linearization of this sink to the existing linearization, making sure to remove
        // aready observed ids from previous sinks.
        linearization ++ linearize(graph, sink).filter(id => !linearization.contains(id)) :+ sink
      }
    }
  }


  /**
    * Helper to remove a node from a graph.
    * @param g Graph.
    * @param x Node to remove.
    * @return
    */
  def removeNode(g: Graph, x: GraphId): Graph = {
    //println(s"Removing ${x}")
    x match {
      case n: NodeId => g.removeNode(n)
      case so: SourceId => g.removeSource(so)
      case si: SinkId => g.removeSink(si)
    }
  }

  def getAllNodes(graph: Graph): Set[GraphId] = {
    (graph.dependencies.keys ++
    graph.sources.toList ++
    graph.sinks.toList).toSet
  }

  /**
    * Return an iterator of all graph linearizations.
    * Importantly we don't actually want to materialize every single one since this is factorial in the number of nodes.
    *
    * @param graph The graph to linearize.
    * @return An iterator of valid topological orderings of the graph.
    */
  def allLinearizations(graph: Graph): Seq[Seq[GraphId]] = {

    //If the graph is empty, return an empty sequence.
    val allNodes: Set[GraphId] = getAllNodes(graph)
    if(allNodes.isEmpty) {
      Seq(Seq[GraphId]())
    } else {

      val nodesWithDeps: Set[GraphId] = (graph.sinkDependencies.keys ++ graph.dependencies.filterNot(_._2.isEmpty).keys).toSet
      val initSet = (allNodes diff nodesWithDeps).toSeq

      initSet.flatMap(i => {
        val restLins = allLinearizations(removeNode(graph, i))
        restLins.map(lin => i +: lin)
      })
    }

  }
}
