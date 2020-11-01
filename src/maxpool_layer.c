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
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    printf("l.width: %d\n", l.width);
    printf("l.height: %d\n", l.height);
    printf("l.stride: %d\n", l.stride);
    printf("l.size: %d\n", l.size);
    printf("l.channels: %d\n", l.channels);
    printf("outw: %d\n", outw);
    printf("outh: %d\n", outh);
    printf("in.rows: %d\n", in.rows);
    printf("in.cols: %d\n", in.cols);
    printf("outw*outh*l.channels: %d\n", outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    int channel, row, col, n, m, paddingSize;
    if(l.size % 2 == 0) { //Even
        paddingSize = 0;
    } else { //Odd
        paddingSize = l.size/2;
    }
    int w_offset = -paddingSize;
    int h_offset = -paddingSize;

    for (channel = 0; channel < l.channels; channel++) {
        for(row = 0; row < outh; row++) {
            for(col = 0; col < outw; col++) {
                int col_index = (channel * outh + row) * outw + col; //THIS COULD BE WRONG
                float max = -FLT_MAX;
                for(n = 0; n < l.size; ++n){ //for every row in kernel
                    for(m = 0; m < l.size; ++m){ //for every column in kernel
                        int cur_h = h_offset + row*l.stride + n;
                        int cur_w = w_offset + col*l.stride + m;

                        //printf("cur_h: %d\n", cur_h);
                        //printf("cur_w: %d\n", cur_w);

                        //1 if cur_h and cur_w point to something inside input, else 0
                        int valid = (cur_h >= 0 && cur_h < l.height && cur_w >= 0 && cur_w < l.width); 

                        //if valid = 1, set val to value at index, else set val to min float
                        //float val = (valid != 0) ? in.data[index] : -FLT_MAX;
                        float val;
                        if(valid == 1) {
                            int index = cur_w + l.width*(cur_h + l.height*channel); //THIS COULD BE WRONG
                            //printf("index: %d\n", index);
                            val = in.data[index];
                        } else {
                            val = -FLT_MAX;
                        }

                        //if val > max, set max to val
                        //max = (val > max) ? val : max;
                        if(val > max) {
                            max = val;
                        }
                    }
                }

                //After looking at all vals from kernel, set output index to max val seen in kernel location of input
                //printf("col_index: %d\n", col_index);
                //printf("Max: %f\n", max);
                out.data[col_index] = max; //max;
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

    printf("l.width: %d\n", l.width);
    printf("l.height: %d\n", l.height);
    printf("l.channels: %d\n", l.channels);
    printf("l.stride: %d\n", l.stride);
    printf("l.size: %d\n", l.size);
    printf("outw: %d\n", outw);
    printf("outh: %d\n", outh);
    printf("dy.rows: %d\n", dy.rows);
    printf("dy.cols: %d\n", dy.cols);
    printf("dx.rows: %d\n", dx.rows);
    printf("dx.cols: %d\n", dx.cols);
    
    //Going Forward, we found one element: the max, in each kernel (2x2 and 3x3)
    //Going Backwards (I BELIEVE BUT NOT SURE), we do dx = foreach(channel) {in[i] + (max we found for kernel of in[i])*(corresponding error in dy)}

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

