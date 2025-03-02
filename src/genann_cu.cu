/*
 * GENANN - Minimal C Artificial Neural Network
 *
 * Copyright (c) 2015-2018 Lewis Van Winkle
 *
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 */

#include "genann.cuh"
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#ifndef genann_act
#define genann_act_hidden genann_act_hidden_indirect
#define genann_act_output genann_act_output_indirect
#else
#define genann_act_hidden genann_act
#define genann_act_output genann_act
#endif

#define LOOKUP_SIZE 4096

#define GRID_SIZE 14
#define BLOCK_SIZE 128

int genann_act_hidden_indirect(const struct genann *ann, double a) {
    return ann->activation_hidden;
}

int genann_act_output_indirect(const struct genann *ann, double a) {
    return ann->activation_output;
}

double interval;
double lookup[LOOKUP_SIZE];
__device__ double d_interval;
__device__ double d_lookup[LOOKUP_SIZE];

#ifdef __GNUC__
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#define unused          __attribute__((unused))
#else
#define likely(x)       x
#define unlikely(x)     x
#define unused
#pragma warning(disable : 4996) /* For fscanf */
#endif

double start_timer() {
  struct timeval cp;
  gettimeofday(&cp, NULL);
  return ((double)cp.tv_sec + (double)cp.tv_usec * 1.e-6);
}

double stop_timer(double start_time) {
    return start_timer() - start_time;
}

double genann_act_sigmoid(const genann *ann unused, double a) {
    if (a < -45.0) return 0;
    if (a > 45.0) return 1;
    return 1.0 / (1 + exp(-a));
}

void genann_init_sigmoid_lookup(const genann *ann) {

        const double sigmoid_dom_min = -15.0;
        const double sigmoid_dom_max = 15.0;

        const double f = (sigmoid_dom_max - sigmoid_dom_min) / LOOKUP_SIZE;
        int i;

        interval = LOOKUP_SIZE / (sigmoid_dom_max - sigmoid_dom_min);
        for (i = 0; i < LOOKUP_SIZE; ++i) {
            lookup[i] = genann_act_sigmoid(ann, sigmoid_dom_min + f * i);
        }
}

double genann_act_sigmoid_cached(const genann *ann unused, double a) {
    assert(!isnan(a));

    const double sigmoid_dom_min = -15.0;
    const double sigmoid_dom_max = 15.0;

    if (a < sigmoid_dom_min) return lookup[0];
    if (a >= sigmoid_dom_max) return lookup[LOOKUP_SIZE - 1];

    size_t j = (size_t)((a-sigmoid_dom_min)*interval+0.5);

    /* Because floating point... */
    if (unlikely(j >= LOOKUP_SIZE)) return lookup[LOOKUP_SIZE - 1];

    return lookup[j];
}

double genann_act_linear(const struct genann *ann unused, double a) {
    return a;
}

double genann_act_threshold(const struct genann *ann unused, double a) {
    return a > 0;
}

// Device activation fonctions

__device__ double device_genann_act_sigmoid(double a) {
    if (a < -45.0) return 0;
    if (a > 45.0) return 1;
    return 1.0 / (1 + exp(-a));
}

void device_genann_init_sigmoid_lookup(const genann *ann) {
    const double sigmoid_dom_min = -15.0;
    const double sigmoid_dom_max = 15.0;

    const double f = (sigmoid_dom_max - sigmoid_dom_min) / LOOKUP_SIZE;

    static double lookup[LOOKUP_SIZE]; // Host array
    double interval = LOOKUP_SIZE / (sigmoid_dom_max - sigmoid_dom_min);

    // Populate the lookup array on the host
    for (int i = 0; i < LOOKUP_SIZE; ++i) {
        lookup[i] = genann_act_sigmoid(ann, sigmoid_dom_min + f * i);
    }

    cudaMemcpyToSymbol(d_lookup, lookup, LOOKUP_SIZE * sizeof(double), 0, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_interval, &interval, sizeof(double), 0, cudaMemcpyHostToDevice);
}


__device__ double device_genann_act_sigmoid_cached(double a) {
    // assert(!isnan(a));

    const double sigmoid_dom_min = -15.0;
    const double sigmoid_dom_max = 15.0;

    if (a < sigmoid_dom_min) return d_lookup[0];
    if (a >= sigmoid_dom_max) return d_lookup[LOOKUP_SIZE - 1];

    size_t j = (size_t)((a-sigmoid_dom_min)*d_interval+0.5);

    /* Because floating point... */
    if (unlikely(j >= LOOKUP_SIZE)) return d_lookup[LOOKUP_SIZE - 1];

    return d_lookup[j];
}

__device__ double device_genann_act_linear(double a) {
    return a;
}

__device__ double device_genann_act_threshold(double a) {
    return a > 0;
}

// Kernel pour appliquer la fonction d'activation
__device__ double apply_activation(double x, int activation_type) {
    switch (activation_type) {
        case 0:
            return device_genann_act_sigmoid_cached(x);
        case 1:
            return device_genann_act_threshold(x);
        case 2:
            return device_genann_act_linear(x);
        default:
            return device_genann_act_sigmoid_cached(x);
    }
}

genann *genann_init(int inputs, int hidden_layers, int hidden, int outputs) {
    if (hidden_layers < 0) return 0;
    if (inputs < 1) return 0;
    if (outputs < 1) return 0;
    if (hidden_layers > 0 && hidden < 1) return 0;

    const int hidden_weights = hidden_layers ? (inputs + 1) * hidden + (hidden_layers - 1) * (hidden + 1) * hidden : 0;
    const int output_weights = (hidden_layers ? (hidden + 1) : (inputs + 1)) * outputs;
    const int total_weights = hidden_weights + output_weights;

    const int total_neurons = inputs + hidden * hidden_layers + outputs;

    const int size = sizeof(genann) + sizeof(double) * (total_weights + total_neurons + (total_neurons - inputs));
    genann *ret = (genann *)malloc(size);
    if (!ret) return 0;

    ret->inputs = inputs;
    ret->hidden_layers = hidden_layers;
    ret->hidden = hidden;
    ret->outputs = outputs;

    ret->total_weights = total_weights;
    ret->total_neurons = total_neurons;

    ret->weight = (double *)((char *)ret + sizeof(genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;

    genann_randomize(ret);

    ret->activation_hidden = 0;
    ret->activation_output = 0;

    device_genann_init_sigmoid_lookup(ret);

    cudaMalloc((void **)&ret->gpu_d_output, sizeof(double) * ret->outputs);
    cudaMalloc((void **)&ret->gpu_d_delta, sizeof(double) * ret->outputs);
    cudaMalloc((void **)&ret->gpu_d_desired_output, sizeof(double) * ret->outputs);

    int total_weights_per_layer = ret->hidden + 1;
    cudaMalloc((void **)&ret->gpu_d_current_output, sizeof(double) * ret->hidden);
    cudaMalloc((void **)&ret->gpu_d_next_deltas, sizeof(double) * ret->outputs);
    cudaMalloc((void **)&ret->gpu_d_next_weights, sizeof(double) * total_weights_per_layer * ret->outputs);
    cudaMalloc((void **)&ret->gpu_d_current_deltas, sizeof(double) * ret->hidden);

    int layer_neurons = (ret->hidden_layers) ? ret->hidden : ret->inputs;
    cudaMalloc((void **)&ret->gpu_d_output_deltas, sizeof(double) * ret->outputs);
    cudaMalloc((void **)&ret->gpu_d_prev_outputs, sizeof(double) * layer_neurons);
    cudaMalloc((void **)&ret->gpu_d_weights, sizeof(double) * (layer_neurons + 1) * ret->outputs);

    cudaMalloc((void **)&ret->gpu_d_layer_deltas, sizeof(double) * ret->hidden);

    cudaMalloc((void **)&ret->gpu_d_prev_outputs_hl_h0, sizeof(double) * ret->inputs);
    cudaMalloc((void **)&ret->gpu_d_prev_outputs_hl, sizeof(double) * ret->hidden);
    cudaMalloc((void **)&ret->gpu_d_weights_hl_h0, sizeof(double) * (ret->inputs + 1) * ret->hidden);
    cudaMalloc((void **)&ret->gpu_d_weights_hl, sizeof(double) * (ret->hidden + 1) * ret->hidden);

    return ret;
}


genann *genann_read(FILE *in) {
    int inputs, hidden_layers, hidden, outputs;
    int rc;

    errno = 0;
    rc = fscanf(in, "%d %d %d %d", &inputs, &hidden_layers, &hidden, &outputs);
    if (rc < 4 || errno != 0) {
        perror("fscanf");
        return NULL;
    }

    genann *ann = genann_init(inputs, hidden_layers, hidden, outputs);

    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        errno = 0;
        rc = fscanf(in, " %le", ann->weight + i);
        if (rc < 1 || errno != 0) {
            perror("fscanf");
            genann_free(ann);

            return NULL;
        }
    }

    return ann;
}


genann *genann_copy(genann const *ann) {
    const int size = sizeof(genann) + sizeof(double) * (ann->total_weights + ann->total_neurons + (ann->total_neurons - ann->inputs));
    genann *ret = (genann *) malloc(size);
    if (!ret) return 0;

    memcpy(ret, ann, size);

    /* Set pointers. */
    ret->weight = (double*)((char*)ret + sizeof(genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;

    return ret;
}

genann *genann_device_copy(genann const *ann) {
	const int size = sizeof(genann) + sizeof(double) * (ann->total_weights + ann->total_neurons + (ann->total_neurons - ann->inputs));
	genann *ret;
	cudaMalloc((void **)&ret, size);
	if (!ret) return 0;

	cudaMemcpy(ret, ann, size, cudaMemcpyHostToDevice);
	return ret;
}


void genann_randomize(genann *ann) {
    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        double r = GENANN_RANDOM();
        /* Sets weights from -0.5 to 0.5. */
        ann->weight[i] = r - 0.5;
    }
}


void genann_free(genann *ann) {
    cudaFree(ann->gpu_d_output);
    cudaFree(ann->gpu_d_desired_output);
    cudaFree(ann->gpu_d_delta);

    cudaFree(ann->gpu_d_current_output);
    cudaFree(ann->gpu_d_next_deltas);
    cudaFree(ann->gpu_d_next_weights);
    cudaFree(ann->gpu_d_current_deltas);

    cudaFree(ann->gpu_d_output_deltas);
    cudaFree(ann->gpu_d_prev_outputs);
    cudaFree(ann->gpu_d_weights);

    cudaFree(ann->gpu_d_layer_deltas);

    cudaFree(ann->gpu_d_prev_outputs_hl_h0);
    cudaFree(ann->gpu_d_prev_outputs_hl);
    cudaFree(ann->gpu_d_weights_hl_h0);
    cudaFree(ann->gpu_d_weights_hl);

    /* The weight, output, and delta pointers go to the same buffer. */
    free(ann);
}


__global__ void compute_layer(
    const double* __restrict__ weights, 
    const double* __restrict__ inputs, 
    double* __restrict__ outputs, 
    const int input_size, 
    const int output_size, 
    const int layer, 
    const int layers,
    const int activ_hidden,
    const int activ_output
) {
    extern __shared__ double shared_mem[];
    double* shared_inputs = shared_mem;

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    
    // Load inputs into the shared memory
    for (int i = tid; i < input_size; i += blockDim.x) {
        shared_inputs[i] = inputs[i];
    }
    
    __syncthreads(); // Synchronize threads
    
    if (gid < output_size) {
        // Calculate the weighted sum
        double sum = weights[gid * (input_size + 1)] * -1.0; // Terme de biais
        for (int j = 0; j < input_size; j++) {
            sum += weights[gid * (input_size + 1) + j + 1] * shared_inputs[j];
        }
        
        // Apply activation function
        int activation_type = (layer == layers) ? activ_output : activ_hidden;
        outputs[gid] = apply_activation(sum, activation_type);
    }
}


double const* genann_run(genann const* ann, double const* inputs) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate GPU memory
    double *d_weights, *d_intermediate;
    cudaMalloc(&d_weights, ann->total_weights * sizeof(double));
    cudaMalloc(&d_intermediate, (ann->inputs + ann->hidden * ann->hidden_layers + ann->outputs) * sizeof(double));

    // Copy weights on the GPU
    cudaMemcpy(d_weights, ann->weight, ann->total_weights * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_intermediate, inputs, ann->inputs * sizeof(double), cudaMemcpyHostToDevice);

    double* current_input = d_intermediate;
    double* current_output = d_intermediate + ann->inputs;
    double* current_weights = d_weights;

    for (int layer = 0; layer <= ann->hidden_layers; ++layer) {
        const int input_size = (layer == 0) ? ann->inputs : ann->hidden;
        const int output_size = (layer == ann->hidden_layers) ? ann->outputs : ann->hidden;
        const int BPG = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        // const int BPG = GRID_SIZE;

        // Shared memory for inputs
        const size_t shared_mem_size = input_size * sizeof(double);

        compute_layer<<<BPG, BLOCK_SIZE, shared_mem_size, stream>>>(current_weights, current_input, current_output, input_size, output_size, layer,ann->hidden_layers,ann->activation_hidden,ann->activation_output);

        // Check kernel errors
        // cudaError_t err = cudaGetLastError();
        // if (err != cudaSuccess) {
        //     printf("Kernel error at layer %d: %s\n", layer, cudaGetErrorString(err));
        //     break;
        // }

        // Go through layers
        current_input = current_output;
        current_output += output_size;
        current_weights += (input_size + 1) * output_size;
    }

    // Copy back the result to the CPU
    cudaMemcpy(
        ann->output, 
        d_intermediate, 
        (ann->inputs + ann->hidden * ann->hidden_layers + ann->outputs) * sizeof(double), 
        cudaMemcpyDeviceToHost
    );

    // Synchronize and free
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(d_weights);
    cudaFree(d_intermediate);

    return ann->output + ann->inputs + ann->hidden * ann->hidden_layers;
}


__global__ void set_output_deltas(double *output, double *delta, const double *desired_output, int output_count, bool is_linear) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < output_count) {
        double o = output[idx];
        double t = desired_output[idx];
        if (is_linear) {
            delta[idx] = t - o;
        } else {
            delta[idx] = (t - o) * o * (1.0 - o);
        }
    }
}


__global__ void set_hidden_layer_deltas(
    const double *output, const double *next_layer_deltas, const double *next_layer_weights,
    double *current_layer_deltas, int hidden_neurons, int next_layer_neurons, int total_weights_per_layer) {

    int neuron_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (neuron_idx < hidden_neurons) {
        double o = output[neuron_idx];
        double delta_sum = 0.0;

        for (int k = 0; k < next_layer_neurons; ++k) {
            int weight_index = k * (hidden_neurons + 1) + (neuron_idx + 1);
            double weight = next_layer_weights[weight_index];
            delta_sum += next_layer_deltas[k] * weight;
        }

        current_layer_deltas[neuron_idx] = o * (1.0 - o) * delta_sum;
    }
}


__global__ void train_output_layer_weights(
    const double *output_deltas, const double *prev_layer_outputs, double *weights,
    double learning_rate, int prev_layer_neurons, int output_neurons) {

    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx < output_neurons) {
        double delta = output_deltas[output_idx];
        int weight_start_idx = output_idx * (prev_layer_neurons + 1);

        weights[weight_start_idx] += learning_rate * delta * -1.0;

        for (int k = 0; k < prev_layer_neurons; ++k) {
            weights[weight_start_idx + 1 + k] += learning_rate * delta * prev_layer_outputs[k];
        }
    }
}


__global__ void train_hidden_layer_weights(
    const double *layer_deltas, const double *prev_layer_outputs, double *weights,
    double learning_rate, int prev_layer_neurons, int current_layer_neurons) {

    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuron_idx < current_layer_neurons) {
        double delta = layer_deltas[neuron_idx];
        int weight_start_idx = neuron_idx * (prev_layer_neurons + 1);

        weights[weight_start_idx] += learning_rate * delta * -1.0;

        for (int k = 0; k < prev_layer_neurons; ++k) {
            weights[weight_start_idx + 1 + k] += learning_rate * delta * prev_layer_outputs[k];
        }
    }
}


double gpu_in_timer = 0;
double gpu_time_timer = 0;
double gpu_out_timer = 0;


void genann_train(genann const *ann, double const *inputs, double const *desired_outputs, double learning_rate) {
    // Forward pass
    genann_run(ann, inputs);

    /* First set the output layer deltas. */
    double *d_output, *d_delta, *d_desired_outputs;
    d_output = ann->gpu_d_output;
    d_delta = ann->gpu_d_delta;
    d_desired_outputs = ann->gpu_d_desired_output;

    double *output = ann->output + ann->inputs + ann->hidden * ann->hidden_layers;
    double *delta = ann->delta + ann->hidden * ann->hidden_layers;

    double start_time = start_timer();
    cudaMemcpy(d_output, output, sizeof(double) * ann->outputs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_desired_outputs, desired_outputs, sizeof(double) * ann->outputs, cudaMemcpyHostToDevice);
    gpu_in_timer += stop_timer(start_time);

    int gridSize = (ann->outputs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // int gridSize = GRID_SIZE;

    bool is_linear = (ann->activation_output == 2);

    start_time = start_timer();
    set_output_deltas<<<gridSize, BLOCK_SIZE>>>(d_output, d_delta, d_desired_outputs, ann->outputs, is_linear);
    gpu_time_timer += stop_timer(start_time);

    start_time = start_timer();
    cudaMemcpy(delta, d_delta, sizeof(double) * ann->outputs, cudaMemcpyDeviceToHost);
    gpu_out_timer += stop_timer(start_time);

    /* Set hidden layer deltas, start on last layer and work backwards. */
    double *d_current_output, *d_next_deltas, *d_next_weights, *d_current_deltas;
    d_current_output = ann->gpu_d_current_output;
    d_next_deltas = ann->gpu_d_next_deltas;
    d_next_weights = ann->gpu_d_next_weights;
    d_current_deltas = ann->gpu_d_current_deltas;

    int hidden_neurons = ann->hidden;
    int next_layer_neurons = ann->outputs;
    int total_weights_per_layer = ann->hidden + 1;

    for (int h = ann->hidden_layers - 1; h >= 0; --h) {
        double const *current_output = ann->output + ann->inputs + (h * hidden_neurons);
        double *current_deltas = ann->delta + (h * hidden_neurons);

        double const *next_deltas = (h == ann->hidden_layers - 1)
            ? (ann->delta + ann->hidden * ann->hidden_layers)
            : (ann->delta + ((h + 1) * hidden_neurons));
        double const *next_weights = (h == ann->hidden_layers - 1)
            ? (ann->weight + ((ann->inputs + 1) * hidden_neurons) + ((hidden_neurons + 1) * hidden_neurons * (h)))
            : (ann->weight + ((ann->inputs + 1) * hidden_neurons) + ((hidden_neurons + 1) * hidden_neurons * (h - 1)));

        start_time = start_timer();
        cudaMemcpy(d_current_output, current_output, sizeof(double) * hidden_neurons, cudaMemcpyHostToDevice);
        cudaMemcpy(d_next_deltas, next_deltas, sizeof(double) * next_layer_neurons, cudaMemcpyHostToDevice);
        cudaMemcpy(d_next_weights, next_weights, sizeof(double) * total_weights_per_layer * next_layer_neurons, cudaMemcpyHostToDevice);
        gpu_in_timer += stop_timer(start_time);

        int gridSize = (hidden_neurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
        // int gridSize = GRID_SIZE;

        start_time = start_timer();
        set_hidden_layer_deltas<<<gridSize, BLOCK_SIZE>>>(
            d_current_output, d_next_deltas, d_next_weights, d_current_deltas,
            hidden_neurons, next_layer_neurons, total_weights_per_layer);
        gpu_time_timer += stop_timer(start_time);

        start_time = start_timer();
        cudaMemcpy(current_deltas, d_current_deltas, sizeof(double) * hidden_neurons, cudaMemcpyDeviceToHost);
        gpu_out_timer += stop_timer(start_time);

        // next_layer_neurons = hidden_neurons;
    }

    /* Train the outputs. */
    double *d_output_deltas, *d_prev_outputs, *d_weights;
    d_output_deltas = ann->gpu_d_output_deltas;
    d_prev_outputs = ann->gpu_d_prev_outputs;
    d_weights = ann->gpu_d_weights;

    int prev_layer_neurons = (ann->hidden_layers) ? ann->hidden : ann->inputs;
    double const *prev_layer_outputs = ann->output + (ann->hidden_layers
            ? (ann->inputs + (ann->hidden) * (ann->hidden_layers - 1))
            : 0);
    double *output_deltas = ann->delta + ann->hidden * ann->hidden_layers;
    double *weights = ann->weight + (ann->hidden_layers
            ? ((ann->inputs + 1) * ann->hidden + (ann->hidden + 1) * ann->hidden * (ann->hidden_layers - 1))
            : 0);

    start_time = start_timer();
    cudaMemcpy(d_output_deltas, output_deltas, sizeof(double) * ann->outputs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_outputs, prev_layer_outputs, sizeof(double) * prev_layer_neurons, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, sizeof(double) * (prev_layer_neurons + 1) * ann->outputs, cudaMemcpyHostToDevice);
    gpu_in_timer += stop_timer(start_time);

    gridSize = (ann->outputs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // gridSize = GRID_SIZE;

    start_time = start_timer();
    train_output_layer_weights<<<gridSize, BLOCK_SIZE>>>(
        d_output_deltas, d_prev_outputs, d_weights,
        learning_rate, prev_layer_neurons, ann->outputs);
    gpu_time_timer += stop_timer(start_time);

    start_time = start_timer();
    cudaMemcpy(weights, d_weights, sizeof(double) * (prev_layer_neurons + 1) * ann->outputs, cudaMemcpyDeviceToHost);
    gpu_out_timer += stop_timer(start_time);

    /* Train the hidden layers. */
    double *d_layer_deltas;
    d_layer_deltas = ann->gpu_d_layer_deltas;

    for (int h = ann->hidden_layers - 1; h >= 0; --h) {
        int prev_layer_neurons = (h == 0) ? ann->inputs : ann->hidden;
        int current_layer_neurons = ann->hidden;

        double const *layer_deltas = ann->delta + h * ann->hidden;
        double const *prev_layer_outputs = (h == 0)
            ? inputs
            : ann->output + ann->inputs + (h - 1) * ann->hidden;
        double *weights = ann->weight + (h
                ? ((ann->inputs+1) * ann->hidden + (ann->hidden+1) * (ann->hidden) * (h-1))
                : 0);

        if (h == 0) {
            d_prev_outputs = ann->gpu_d_prev_outputs_hl_h0;
            d_weights = ann->gpu_d_weights_hl_h0;
        } else {
            d_prev_outputs = ann->gpu_d_prev_outputs_hl;
            d_weights = ann->gpu_d_weights_hl;
        }

        start_time = start_timer();
        cudaMemcpy(d_layer_deltas, layer_deltas, sizeof(double) * current_layer_neurons, cudaMemcpyHostToDevice);
        cudaMemcpy(d_prev_outputs, prev_layer_outputs, sizeof(double) * prev_layer_neurons, cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights, weights, sizeof(double) * (prev_layer_neurons + 1) * current_layer_neurons, cudaMemcpyHostToDevice);
        gpu_in_timer += stop_timer(start_time);

        gridSize = (current_layer_neurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
        // gridSize = GRID_SIZE;

        start_time = start_timer();
        train_hidden_layer_weights<<<gridSize, BLOCK_SIZE>>>(
            d_layer_deltas, d_prev_outputs, d_weights,
            learning_rate, prev_layer_neurons, current_layer_neurons);
        gpu_time_timer += stop_timer(start_time);

        start_time = start_timer();
        cudaMemcpy(weights, d_weights, sizeof(double) * (prev_layer_neurons + 1) * current_layer_neurons, cudaMemcpyDeviceToHost);
        gpu_out_timer += stop_timer(start_time);
    }
}


void genann_write(genann const *ann, FILE *out) {
    fprintf(out, "%d %d %d %d", ann->inputs, ann->hidden_layers, ann->hidden, ann->outputs);

    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        fprintf(out, " %.20e", ann->weight[i]);
    }
}


void get_gpu_timers(double *gpu_in, double *gpu_time, double *gpu_out) {
    *gpu_in = gpu_in_timer;
    *gpu_time = gpu_time_timer;
    *gpu_out = gpu_out_timer;
}
