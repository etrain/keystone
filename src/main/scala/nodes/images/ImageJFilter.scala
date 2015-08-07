package nodes.images

import ij.ImageStack
import ij.plugin.Filters3D
import ij.plugin.filter.PlugInFilter
import ij.process.{FloatProcessor, ImageProcessor}
import nodes.images.FilterType.FilterType
import pipelines._
import utils.{ImageUtils, ImageMetadata, RowMajorArrayVectorizedImage, Image}

/**
 * Takes an Image, applies a given ImageJ filter to it, and returns the filtered image.
 * NOTE: For chains of filters, this is inefficient because it copies a KeystoneML Image to an ImageJ image,
 * performs the operation, and returns to KeystoneML Image.
 *
 * It is useful for one-off filter application, but application of several filters should be done with ImageJFilter.
 * //TODO: Link to class above in docs.
 *
 * @param filter An ImageJ filter to apply.
 */
class Filter(filter: PlugInFilter) extends Transformer[Image, Image] {
  def apply(in: Image): Image = {
    val ip = ImageJUtils.toImageProcessor(in)
    filter.run(ip) //ImageJ operates on the image in-place.
    ImageJUtils.toImage(ip)
  }
}

object FilterType extends Enumeration {
  type FilterType = Value
  val Max, MaxLocal, Mean, Median, Min, Var = Value
}

class Filter3D(filter: PlugInFilter, filterType: FilterType) extends Transformer[Image, Image] {

  val filterMap = Map(
    FilterType.Max -> Filters3D.MAX,
    FilterType.MaxLocal -> Filters3D.MAXLOCAL,
    FilterType.Mean -> Filters3D.MEAN,
    FilterType.Median -> Filters3D.MEDIAN,
    FilterType.Min -> Filters3D.MIN,
    FilterType.Var -> Filters3D.VAR
  )

  def apply(in: Image): Image = {
    val is = ImageJUtils.toImageStack(in)
    Filters3D.filter(is, filterMap(filterType), in.metadata.yDim, in.metadata.xDim, in.metadata.numChannels)
    ImageJUtils.toImage(is)
  }
}


/**
 * This node takes transforms input images and applies ImageJ filters to them.
 * @param filter an ImageJ filter.
 */
class ImageJFilter(filter: PlugInFilter) extends Transformer[ImageProcessor, ImageProcessor] {
  def apply(in: ImageProcessor): ImageProcessor = {
    filter.run(in) //ImageJ operates on the image in-place.
    in //For compatibility with the KeystoneML programming model, we return a reference to the modified image.
  }
}


object ImageJUtils {
  def toImageProcessor(in: Image): ImageProcessor = {
    val floats = in.getSingleChannelAsFloatArray() //Todo: must make sure ordering is as ImageJ expects it.
    new FloatProcessor(in.metadata.yDim, in.metadata.xDim, floats)
  }

  def toImage(in: ImageProcessor): Image = {
    val data = in.getFloatArray
    val flatData = data.flatten.map(_.toDouble)
    new RowMajorArrayVectorizedImage(flatData, ImageMetadata(in.getHeight, in.getWidth, 1)) //Todo: verify that this went in the right direction.
  }

  def toImageStack(in: Image): ImageStack = {
    //Convert all the channels to an ImageProcessor.
    val ips = ImageUtils.splitChannels(in).map(toImageProcessor)

    //Allocate an new ImageStack
    val is = new ImageStack(in.metadata.yDim, in.metadata.xDim, in.metadata.numChannels)

    //Add a slice to the ImageStack
    ips.foreach(ip => is.addSlice(ip))

    is
  }

  def toImage(in: ImageStack): Image = {
    ImageUtils.combineChannels((0 until in.getSize).map(i => in.getProcessor(i)).map(toImage).toArray) //Todo: this could be more efficient.
  }
}