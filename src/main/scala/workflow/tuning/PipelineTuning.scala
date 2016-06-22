package workflow.tuning

import org.apache.spark.rdd.RDD
import workflow._

import scala.reflect.ClassTag

object PipelineTuning {
  /**
    * A method used for tuning between several pipeline options on validation data.
    *
    * It takes multiple pipelines, validation data and labels, and an evaluation metric,
    * and returns a newpipeline that evaluates all of the inputs on the validation data, then applies
    * the best-scoring pipeline to test data.
    *
    * @param branches The sequence of pipelines to tune between.
    * @param data The validation data to pass into each branch
    * @param labels The labels for the validation data
    * @param evaluator The evaluation metric to use on the validation data
    * @tparam A The input type of the pipelines
    * @tparam B The output type of the pipelines
    * @tparam L The type of the validation labels
    * @return
    */
  def tune[A, B : ClassTag, L](
      branches: Seq[Pipeline[A, B]],
      data: RDD[A],
      labels: RDD[L],
      evaluator: (RDD[B], RDD[L]) => Double): Pipeline[A, B] = {

    //This method is based on the logic in Pipeline.gather.

    //First, we create one set of branches for each of these pipelines which uses the validation dataset.
    //Initialize to an empty graph with one source.
    val dataSetGraph = PipelineDataset(data).executor.graph
    val dataSetGraphNoSink = dataSetGraph.removeSink(dataSetGraph.sinks.head)
    val dataSource = dataSetGraph.nodes.head

    // We fold the branches together one by one, updating the graph and the overall execution state
    // to include all of the branches.
    val (graphWithDataBranches, dataBranchSinks) = branches.foldLeft(
      dataSetGraphNoSink,
      Seq[NodeOrSourceId]()) {
      case ((graph, sinks), branch) =>
        // We add the new branch to the graph containing already-processed branches
        val (graphWithBranch, sourceMapping, _, sinkMapping) = graph.addGraph(branch.executor.graph)

        // We then remove the new branch's individual source and make the branch
        // depend on the new joint source for all branches.
        // We also remove the branch's sink.
        val branchSource = sourceMapping(branch.source)
        val branchSink = sinkMapping(branch.sink)
        val branchSinkDep = graphWithBranch.getSinkDependency(branchSink)
        val nextGraph = graphWithBranch.replaceDependency(branchSource, dataSource)
          .removeSource(branchSource)
          .removeSink(branchSink)

        (nextGraph, sinks :+ branchSinkDep)
    }

    val (graphWithDataBranchesAndLabels, labelNode) = graphWithDataBranches.addNode(new DatasetOperator(labels), Seq())
    val modelSelectorDependencies = dataBranchSinks.flatMap(x => Seq(x, labelNode))

    // Next add a gather transformer with all of the branches' endpoints as dependencies,
    // and add a new sink on the model selector.
    val (graphWithSelector, selectorNode) = graphWithDataBranchesAndLabels.addNode(new ModelSelector(evaluator), modelSelectorDependencies)
    //val (dataGraph, dataSink) = graphWithSelector.addSink(selectorNode)

    //Next, we create one set of branches for each of these pipelines which uses the validation dataset.
    //Initialize to an empty graph with one source.
    val source = SourceId(0)
    val emptyGraph = Graph(Set(source), Map(), Map(), Map())

    val (graphWithEmptyBranches, emptyBranchSinks) = branches.foldLeft(
      emptyGraph,
      Seq[NodeOrSourceId]()) {
      case ((graph, sinks), branch) =>
        // We add the new branch to the graph containing already-processed branches
        val (graphWithBranch, sourceMapping, _, sinkMapping) = graph.addGraph(branch.executor.graph)

        // We then remove the new branch's individual source and make the branch
        // depend on the new joint source for all branches.
        // We also remove the branch's sink.
        val branchSource = sourceMapping(branch.source)
        val branchSink = sinkMapping(branch.sink)
        val branchSinkDep = graphWithBranch.getSinkDependency(branchSink)
        val nextGraph = graphWithBranch.replaceDependency(branchSource, source)
          .removeSource(branchSource)
          .removeSink(branchSink)

        (nextGraph, sinks :+ branchSinkDep)
    }

    // Next add a gather transformer with all of the branches' endpoints as dependencies,
    // and add a new sink on the delegating transformer.
    val (graphWithDelegatingTransformer, delegatingTransformerNode) = graphWithEmptyBranches.addNode(new DelegatingOperator(), emptyBranchSinks)
    val (delegatingGraph, delegatingSink) = graphWithDelegatingTransformer.addSink(delegatingTransformerNode)

    //Add fit dependency between delegating transformer and model selector.
    val (newGraph, newSelectorSources, newSelectorNodes, newSelectorSinks) = delegatingGraph.addGraph(graphWithSelector)

    val getNewSelectors: NodeOrSourceId => NodeOrSourceId = { case x: NodeId => newSelectorNodes(x) }

    //val res = newGraph.setDependencies(newSelector(delegatingTransformerNode),
    //  delegatingGraph.getDependencies(delegatingTransformerNode).map(getNewDelegators) :+ selectorNode)

    val res = newGraph.setDependencies(delegatingTransformerNode, newSelectorNodes(selectorNode) +: newGraph.getDependencies(delegatingTransformerNode))

    // Finally, construct & return the new "tuning" pipeline
    val executor = new GraphExecutor(res)
    new Pipeline[A, B](executor, source, delegatingSink)
  }
}