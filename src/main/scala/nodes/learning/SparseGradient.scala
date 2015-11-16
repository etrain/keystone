/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package nodes.learning

import breeze.linalg._

trait SparseGradient extends Serializable {
  
  def compute(
      data: SparseVector[Double],
      labels: DenseVector[Double],
      weights: DenseMatrix[Double],
      cumGradient: DenseMatrix[Double])
    : Double
}

class LeastSquaresSparseGradient extends SparseGradient {
  
  def compute(
      data: SparseVector[Double],
      labels: DenseVector[Double],
      weights: DenseMatrix[Double],
      cumGradient: DenseMatrix[Double])
    : Double = {

    // Least Squares Gradient is At.(Ax - b)
    val axb = weights.t * data
    axb -= labels

    var offset = 0
    while(offset < data.activeSize) {
      val index = data.indexAt(offset)
      val value = data.valueAt(offset)
      cumGradient(index, ::) += (axb.t * value)
      offset += 1
    }

    // val dataMat = new CSCMatrix[Double](
    //   data.data, data.length, 1, Array(0, data.used), data.index)
    // cumGradient += dataMat * axb.asDenseMatrix // (axb.asDenseMatrix.t * dataMat.t).t
    // Loss is 0.5 * norm(Ax - b)

    val loss = 0.5 * math.pow(norm(axb), 2)

    loss
  }

}
