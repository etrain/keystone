package utils.images.imagej;

import java.awt.*;
import ij.*;
import ij.process.*;
import ij.gui.*;
import ij.plugin.filter.*;

public class Example3StackAverage implements PlugInFilter {
    ImagePlus imp;
    protected ImageStack stack;

    public int setup(String arg, ImagePlus imp) {
        if (arg.equals("about")) {
            showAbout();
            return DONE;
        } //convert 16 to 8 without asking!!!
        stack = imp.getStack();
        return DOES_8G+DOES_16+STACK_REQUIRED;
    }

    public void run(ImageProcessor ip) {
        ip.convertToByte(true);

        byte[] pixels;
        int dim = stack.getWidth() * stack.getHeight();
        int [] sum = new int[dim]; // array to hold the summed pixel values.
        //iterate through the slices
        for (int i=1; i<=stack.getSize(); i++) {
            pixels = (byte[]) stack.getPixels(i);//return a pixel array for the slice
            for (int j=0;j<dim;j++) {
                sum[j] += pixels[j] & 0xff; //eliminating the sign bit!
            }
        }//end for

        byte[] average = new byte[dim];
        for (int j=0;j<dim;j++) {
            average[j] = (byte) ((sum[j]/stack.getSize()) & 0xff);
        }
        stack.addSlice("Average",average);
        IJ.log("Result written as the last slice of the stack");
    }

    void showAbout() {
        IJ.showMessage("About This code_...",
                "This sample plugin was used to filter inverts 8-bit images. Look\n" +
                        "at the 'Inverter_.java' source file to see how easy it is\n" +
                        "in ImageJ to process non-rectangular ROIs, to process\n" +
                        "all the slices in a stack, and to display an About box."
        );
    }
}
