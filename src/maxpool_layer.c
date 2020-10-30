#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"

//Helper Function to get needed pixel from input or 0 if in padding 
float get_pixel_value1(matrix in, int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= in.rows || col >= in.cols) {
        return 0;
    }
    return in.data[col + in.cols*(row + in.rows*channel)];
}

// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    int channel, row, col, n, m;
    //int kernelElems = size*size;
    int paddingSize = l.size/2;
    int w_offset = -paddingSize;
    int h_offset = -paddingSize;

    for (channel = 0; channel < l.channels; channel++) {
        for(row = 0; row < outh; row++) {
            for(col = 0; col < outw; col++) {
                int col_index = (channel * outh + row) * outw + col;
                float max = (float)INT32_MIN;
                int max_i = -1;
                for(n = 0; n < l.size; ++n){
                    for(m = 0; m < l.size; ++m){
                        int cur_h = h_offset + row*l.stride + n;
                        int cur_w = w_offset + col*l.stride + m;
                        int index = cur_w + l.width*(cur_h + l.height*l.channels);
                        int valid = (cur_h >= 0 && cur_h < l.height && cur_w >= 0 && cur_w < l.width);
                        float val = (valid != 0) ? in.data[index] : -FLT_MAX;
                        max_i = (val > max) ? index : max_i;
                        max   = (val > max) ? val   : max;
                    }
                }
                out.data[col_index] = max;
            }
        }
    }
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix prev_delta: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.



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

