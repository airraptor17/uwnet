#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"

// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    int channel, row, col, kernelRow, kernelCol, paddingSize;
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outh = (l.height-1)/l.stride + 1;
    int outw = (l.width-1)/l.stride + 1;

    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    if(l.size % 2 == 0) { //Even
        paddingSize = 0;
    } else { //Odd
        paddingSize = l.size/2;
    }

    for (channel = 0; channel < l.channels; channel++) {
        for(row = 0; row < outh; row++) {
            for(col = 0; col < outw; col++) {
                int colInx =  (channel * outh + row) * outw + col;
                float maxPixel = -FLT_MAX;
                for(kernelRow = 0; kernelRow < l.size; kernelRow++){ //for every row in kernel
                    for(kernelCol = 0; kernelCol < l.size; kernelCol++){ //for every column in kernel
                        int curRow = (row*l.stride + kernelRow) - paddingSize;
                        int curCol = (col*l.stride + kernelCol) - paddingSize;
                        
                        float pixelVal;
                        if(curRow >= 0 && curRow < l.height && curCol >= 0 && curCol < l.width) {
                            int inx = (l.width*curRow) + (l.width*l.height*channel) + curCol;
                            pixelVal = in.data[inx];
                        } else {
                            pixelVal = -FLT_MAX;
                        }

                        if(pixelVal > maxPixel) {
                            maxPixel = pixelVal;
                        }
                    }
                }

                out.data[colInx] = maxPixel;
            }
        }
    }
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;

    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    int channel, row, col, kernelRow, kernelCol, paddingSize;
    if(l.size % 2 == 0) { //Even
        paddingSize = 0;
    } else { //Odd
        paddingSize = l.size/2;
    }

    for (channel = 0; channel < l.channels; channel++) {
        for(row = 0; row < outh; row++) {
            for(col = 0; col < outw; col++) {
                int colInx =  (channel * outh + row) * outw + col;
                float maxPixel = -FLT_MAX;
                int maxInx = 0;
                for(kernelRow = 0; kernelRow < l.size; kernelRow++){ //for every row in kernel
                    for(kernelCol = 0; kernelCol < l.size; kernelCol++){ //for every column in kernel
                        int curRow = (row*l.stride + kernelRow) - paddingSize;
                        int curCol = (col*l.stride + kernelCol) - paddingSize;

                        float pixelVal;
                        int inx = (l.width*curRow) + (l.width*l.height*channel) + curCol;
                        if (curRow >= 0 && curRow < l.height && curCol >= 0 && curCol < l.width) {
                            pixelVal = in.data[inx];
                        } else {
                            pixelVal = -FLT_MAX;
                        }

                        if(pixelVal > maxPixel) {
                            maxPixel = pixelVal;
                            maxInx = inx;
                        }
                    }
                }

                out.data[colInx] = maxInx;
            }
        }
    }

    int i;
    int h = outh;
    int w = outw;
    int c = l.channels;
    for(i = 0; i < h*w*c; i++){
        int inx = out.data[i];
        dx.data[inx] += dy.data[i];
    }

    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

