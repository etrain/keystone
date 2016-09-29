package loaders

import java.io.{DataInputStream, ByteArrayInputStream, BufferedInputStream}

import breeze.linalg.DenseVector
import com.google.common.io.LittleEndianDataInputStream
import org.apache.spark.SparkContext
import org.apache.spark.input.PortableDataStream
import org.apache.spark.rdd.RDD
import org.tensorflow.example.feature._

import scala.collection.mutable

/**
  * This class loads data from TFRecord Files as described here:
  * https://www.tensorflow.org/versions/r0.10/api_docs/python/python_io.html#tfrecords-format-details
  */

object TFRecordLoader {
  /**
    * Given a block, generate an Iterator of SequenceExamples.
 *
    * @param x
    * @return
    */


  private def generateBlocks(x: PortableDataStream): Iterator[SequenceExample] = {
    val ret = mutable.Queue[SequenceExample]()

    val bis = new ByteArrayInputStream(x.toArray())
    val dis = new LittleEndianDataInputStream(bis)
    while(bis.available() > 0) {
      val recSize = dis.readLong()
      //println(s"Record size: $recSize")
      dis.readInt() //Read the CRC

      val bytes = new Array[Byte](recSize.toInt)
      dis.readFully(bytes) //Populate bytes.
      dis.readInt() //Read another CRC

      val parsed = SequenceExample.parseFrom(bytes)

      ret.enqueue(parsed)
    }
    dis.close()
    ret.iterator
  }

  /**
   * Loads the pre-featurized Timit data.
   * Expects features data to be stored as a csv of numbers,
   * and labels as "row# label" where row# is the number of the row in the data csv it is
   * referring to (starting at row #1)
   *
   * @param sc  SparkContext to use
   * @param dataLocation Name of file containing TFRecords.
   * @param numPartitions Number of partitions to load into
   * @return  An RDD containing the loaded data.
   */
  def apply(sc: SparkContext, dataLocation: String, numPartitions: Int): RDD[SequenceExample] = {
    sc.binaryFiles(dataLocation, numPartitions).flatMap{ case (file, blockStream) => generateBlocks(blockStream)}
  }
}
