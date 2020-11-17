#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "uwnet.h"

// Take mean of matrix x over rows and spatial dimension
// matrix x: matrix with data
// int groups: number of distinct means to take, usually equal to # outputs
// after connected layers or # channels after convolutional layers
// returns: (1 x groups) matrix with means
matrix mean(matrix x, int groups)
{
    assert(x.cols % groups == 0);
    matrix m = make_matrix(1, groups);
    int n = x.cols / groups;
    int i, j;
    for(i = 0; i < x.rows; ++i){
        for(j = 0; j < x.cols; ++j){
            m.data[j/n] += x.data[i*x.cols + j];
        }
    }
    for(i = 0; i < m.cols; ++i){
        m.data[i] = m.data[i] / x.rows / n;
    }
    return m;
}

// Take variance over matrix x given mean m
matrix variance(matrix x, matrix m, int groups)
{
    //assert(x.cols % groups == 0);
    matrix v = make_matrix(1, groups);
    int n = x.cols / groups;
    int i, j;
    for(i = 0; i < x.rows; ++i){
        for(j = 0; j < x.cols; ++j){
            v.data[j/n] += ((x.data[i*x.cols + j] - m.data[j/n])*(x.data[i*x.cols + j] - m.data[j/n]));
        }
    }
    for(i = 0; i < v.cols; ++i){
        v.data[i] = v.data[i] / x.rows / n;
    }
    return v;
}

// Normalize x given mean m and variance v
// returns: y = (x-m)/sqrt(v + epsilon)
matrix normalize(matrix x, matrix m, matrix v, int groups)
{
    matrix norm = make_matrix(x.rows, x.cols);
    // TODO: 7.2 - Normalize x
    float eps = 0.00001f;
    int n = x.cols / groups;
    int i, j;
    for(i = 0; i < x.rows; ++i){
        for(j = 0; j < x.cols; ++j){
            norm.data[i*x.cols + j] += ((x.data[i*x.cols + j] - m.data[j/n])/sqrt(v.data[j/n] + eps));
        }
    }
    return norm;
}


// Run an batchnorm layer on input
// layer l: pointer to layer to run
// matrix x: input to layer
// returns: the result of running the layer y = (x - mu) / sigma
matrix forward_batchnorm_layer(layer l, matrix x)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(x);

    if(x.rows == 1){
        return normalize(x, l.rolling_mean, l.rolling_variance, l.channels);
    }

    float s = 0.1;
    matrix m = mean(x, l.channels);
    matrix v = variance(x, m, l.channels);
    matrix y = normalize(x, m, v, l.channels);

    scal_matrix(1-s, l.rolling_mean);
    axpy_matrix(s, m, l.rolling_mean);
    scal_matrix(1-s, l.rolling_variance);
    axpy_matrix(s, v, l.rolling_variance);

    free_matrix(m);
    free_matrix(v);

    return y;
}

//V is variance matrix
matrix delta_mean(matrix d, matrix v)
{
    int groups = v.cols;
    matrix dm = make_matrix(1, groups);
    
    printf("d.rows: %d, d.cols: %d\n", d.rows, d.cols);
    printf("v.rows: %d, v.cols: %d\n", v.rows, v.cols);
    printf("dm.rows: %d, dm.cols: %d\n", dm.rows, dm.cols);
    

    // TODO 7.3 - Calculate dL/dm
    float eps = 0.00001f;
    int n = d.cols / groups;
    int i, j;

    for(i = 0; i < d.rows; ++i){
        for(j = 0; j < d.cols; ++j){
            dm.data[j/n] += (d.data[i*d.cols + j] * (-1/sqrt(v.data[j/n] + eps)));
        }
    }

    return dm;
}


matrix delta_variance(matrix d, matrix x, matrix m, matrix v)
{
    int groups = m.cols;
    matrix dv = make_matrix(1, groups);
    
    printf("d.rows: %d, d.cols: %d\n", d.rows, d.cols);
    printf("x.rows: %d, x.cols: %d\n", x.rows, x.cols);
    printf("m.rows: %d, m.cols: %d\n", m.rows, m.cols);
    printf("v.rows: %d, v.cols: %d\n", v.rows, v.cols);
    printf("dv.rows: %d, dv.cols: %d\n", dv.rows, dv.cols);
    
    // TODO 7.4 - Calculate dL/dv
    float eps = 0.00001f;
    int n = d.cols / groups;
    int i, j;

    for(i = 0; i < d.rows; ++i){
        for(j = 0; j < d.cols; ++j){
            float dL_dy = d.data[i*d.cols + j];
            //printf("dL/dy: %f ", dL_dy);

            float x_val = x.data[i*d.cols + j];
            //printf("x_val: %f ", x_val);

            float mu_val = m.data[j/n];
            //printf("mu_val: %f ", mu_val);

            float var_val = v.data[j/n] + eps;
            //printf("var_val: %f ", var_val);

            float power_val = (-0.5)*powf(var_val, -1.5);
            //printf("power_val: %f ", power_val);

            float temp = (dL_dy * (x_val - mu_val) * power_val);
            dv.data[j/n] += temp;
        }
    }

    return dv;
}

matrix delta_batch_norm(matrix d, matrix dm, matrix dv, matrix m, matrix v, matrix x)
{
    matrix dx = make_matrix(d.rows, d.cols);
    // TODO 7.5 - Calculate dL/dx
    return dx;
}


// Run an batchnorm layer on input
// layer l: pointer to layer to run
// matrix dy: derivative of loss wrt output, dL/dy
// returns: derivative of loss wrt input, dL/dx
matrix backward_batchnorm_layer(layer l, matrix dy)
{
    matrix x = *l.x;

    matrix m = mean(x, l.channels);
    matrix v = variance(x, m, l.channels);

    matrix dm = delta_mean(dy, v);
    matrix dv = delta_variance(dy, x, m, v);
    matrix dx = delta_batch_norm(dy, dm, dv, m, v, x);

    free_matrix(m);
    free_matrix(v);
    free_matrix(dm);
    free_matrix(dv);

    return dx;
}

// Update batchnorm layer..... nothing happens tho
// layer l: layer to update
// float rate: SGD learning rate
// float momentum: SGD momentum term
// float decay: l2 normalization term
void update_batchnorm_layer(layer l, float rate, float momentum, float decay){}

layer make_batchnorm_layer(int groups)
{
    layer l = {0};
    l.channels = groups;
    l.x = calloc(1, sizeof(matrix));

    l.rolling_mean = make_matrix(1, groups);
    l.rolling_variance = make_matrix(1, groups);

    l.forward = forward_batchnorm_layer;
    l.backward = backward_batchnorm_layer;
    l.update = update_batchnorm_layer;
    return l;
}
