#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include "genann.cuh"
#include "mnist.cuh"

#define SIZE 28

double _start_timer() {
    struct timeval cp;
    gettimeofday(&cp, NULL);
    return ((double)cp.tv_sec + (double)cp.tv_usec * 1.e-6);
}

void _stop_timer(double start_time, const char *device) {
    double time = _start_timer() - start_time;
    printf("Timer for %s: %lf\n", device, time);
}

void copy_to_double(mnist_data_ann *data_ann, mnist_data *data_raw, int cnt) {
    for (int i = 0; i < cnt; i++) {
        data_ann[i].label = (double)data_raw[i].label;
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                data_ann[i].data[j*28+k] = (double)data_raw[i].data[j][k];
            }
        }
    }
}

int main(int argc, char *argv[])
{
    printf("GENANN example 1.\n");
    printf("Train a mnist.\n");
    //printf("%d\n", sizeof(mnist_data_ann));

    // char* labelFileName = "t10k-labels.idx1-ubyte";
    // char* imageFileName = "t10k-images.idx3-ubyte";
    char* labelFileName = "train-labels.idx1-ubyte";
    char* imageFileName = "train-images.idx3-ubyte";

    mnist_data *data_raw;
    mnist_data_ann *data_ann;
    unsigned int cnt;
    int ret;

    if (ret = loadMNISTDataUpTo(imageFileName, labelFileName, &data_raw, &cnt)) {
        printf("An error occured: %d\n", ret);
    }

    data_ann = (mnist_data_ann*)malloc(sizeof(mnist_data_ann) * cnt);
    copy_to_double(data_ann, data_raw, cnt);

    printf("Loaded %d images.\n", cnt);

    int i;

    /* New network with 28x28 MNIST input,
    * hidden layers of HIDDEN neurons,
    * and LABELS_SIZE outputs. */
    genann *ann = genann_init(28*28, HIDDEN_LAYERS, HIDDEN, LABELS_SIZE);
    //printf("%d\n", cnt);
    double * arr = (double*)malloc(sizeof(double) * LABELS_SIZE);

    /* Train for 5 epochs using the full training dataset. */
    printf("HiddenLayers=%d, Hidden=%d, Outputs=%d\n", HIDDEN_LAYERS, HIDDEN, LABELS_SIZE);
    double start_time = _start_timer();
    for (i = 0; i < 5; ++i) {
        //printf("big i %d\n", i);
        for (int j = 0; j < 100; j++) {
            memset(arr, 0, sizeof(double) * LABELS_SIZE);
            arr[(int)data_ann[j].label] = 1;
            //printf("%d\n", j);
            genann_train(ann, data_ann[j].data, arr, 0.1);
        }
    }
    _stop_timer(start_time, "GPU");
    double gpu_in, gpu_time, gpu_out;
    get_gpu_timers(&gpu_in, &gpu_time, &gpu_out);
    printf("GPU Memory in : %f\n", gpu_in);
    printf("GPU Time : %f\n", gpu_time);
    printf("GPU Memory out : %f\n", gpu_out);

    int correct = 0;

    for (int j = 0; j < cnt; j++) {
        const double * out = genann_run(ann, data_ann[j].data);
        int max = 0;
        for (int i = 1; i < LABELS_SIZE; i++) {
            if (out[i] > out[max]) {
                max = i;
            }
        }
        if (max == (int)data_ann[j].label) correct++;
    }

    printf("Correct percentage: %lf\n", 1.0 * correct / cnt * 100);

    //char c;
    //scanf_s("%c", &c);
    genann_free(ann);
    free(data_raw);
    return 0;
}
