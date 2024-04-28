#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define FLOAT_TOLERANCE 1e-6

void matmul_forward(float* out,
                    float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                float* wrow = weight + o*C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

// matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
void matmul_cached(float* out,
                    float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC) {
    // same as above but using available cached values
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // start by computing all the Q values
            float* out_bt = out + b * T * OC + t * OC;
            float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < C; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                float* wrow = weight + o*C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }

        // now just calculate k and v last tokens
        float* out_bt_k = out + b * T * OC + (T-1) * OC + C; // Offset by C to store K
        float* out_bt_v = out + b * T * OC + (T-1) * OC + 2 * C; // Offset by 2*C to store V
        float* inp_bt = inp + b * T * C + (T-1) * C;
        
        for (int o = 0; o < C; o++) {
            // Compute K
            float val_k = (bias != NULL) ? bias[o + C] : 0.0f; // Assuming bias is also structured as QKV
            float* wrow_k = weight + (o + C) * C; // Offset in weight matrix for K
            for (int i = 0; i < C; i++) {
                val_k += inp_bt[i] * wrow_k[i];
            }
            out_bt_k[o] = val_k;
            
            // Compute V
            float val_v = (bias != NULL) ? bias[o + 2 * C] : 0.0f; // Bias offset for V
            float* wrow_v = weight + (o + 2 * C) * C; // Weight offset for V
            for (int i = 0; i < C; i++) {
                val_v += inp_bt[i] * wrow_v[i];
            }
            out_bt_v[o] = val_v;
        }
    }
}

// Function to initialize array with random values
void initialize_random(float *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = (float)rand() / RAND_MAX * 10.0; // Random float between 0 and 10
    }
}

// Function to compare arrays
int compare_arrays(float *arr1, float *arr2, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(arr1[i] - arr2[i]) > FLOAT_TOLERANCE) {
            printf("%d\n", i);
            return 0; // Arrays are not equal
        }
    }
    return 1; // Arrays are equal
}

int main() {
    int B = 5; // Batch size
    int T = 10; // Number of tokens
    int C = 64; // Channels
    int OC = 3 * C; // Output channels for Q, K, V

    // Allocate memory for inputs and outputs
    float *inp = (float *)malloc(B * T * C * sizeof(float));
    float *weight = (float *)malloc(OC * C * sizeof(float));
    float *bias = (float *)malloc(OC * sizeof(float));
    float *out1 = (float *)malloc(B * T * OC * sizeof(float));
    float *out2 = (float *)malloc(B * T * OC * sizeof(float));

    // Initialize inputs with random data
    initialize_random(inp, B * T * C);
    initialize_random(weight, OC * C);
    initialize_random(bias, OC);

    // Initialize out1
    initialize_random(out1, B * T * OC);

    // Perform matrix multiplication with matmul_forward
    matmul_forward(out1, inp, weight, bias, B, T, C, OC);

    memcpy(out2, out1, B * T * OC * sizeof(float)); // Copy initial values from out1 to out2
    // Assume out2 already has valid data and only the last token K and V need to be updated
    matmul_cached(out2, inp, weight, bias, B, T, C, OC);

    // Compare outputs
    if (compare_arrays(out1, out2, B * T * OC)) {
        printf("The outputs are identical.\n");
    } else {
        printf("The outputs differ.\n");

        printf("Inspecting outputs at critical indices:\n");
        for (int idx = 64; idx < 64 + 10; idx++) {
            printf("out1[%d] = %f, out2[%d] = %f\n", idx, out1[idx], idx, out2[idx]);
        }
    }

    // Clean up
    free(inp);
    free(weight);
    free(bias);
    free(out1);
    free(out2);

    return 0;
}
