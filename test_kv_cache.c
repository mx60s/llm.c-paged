#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void matmul_cached(float* out,
                    float* inp, float* weight, float* bias,
                    float* kv_cache, int B, int T, int C, int OC, int max_length) {
    // OC should ideally be 3 times the number of actual output channels per type (Q, K, V)
    // assuming the first third of OC is for Q, the second third for K, and the last third for V
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    int C2 = 2 * C;

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            float* inp_bt = inp + b * T * C + t * C;

            // Process query for all tokens
            for (int o = 0; o < C; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                float* wrow = weight + o * C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }

            // Fetch the latest key and value from KV cache unless it's the last token
            if (t == T - 1) {
                // Compute new K and V for the last token and update the cache
                for (int o = C; o < OC; o++) {
                    float val = (bias != NULL) ? bias[o] : 0.0f;
                    float* wrow = weight + o * C;
                    for (int i = 0; i < C; i++) {
                        val += inp_bt[i] * wrow[i];
                    }
                    out_bt[o] = val;
                    kv_cache[b * max_length * C2 + t * C2 + o - C] = val;
                }
            } else {
                // Use cached K and V
                float* kv_ptr = kv_cache + b * max_length * C2 + t * C2;
                for (int i = 0; i < C2; i++) {
                    out_bt[C + i] = kv_ptr[i];
                }
            }
        }
    }
}

// Function to compare arrays and print results
void assert_arrays(float* expected, float* actual, int size, char* test_name) {
    for (int i = 0; i < size; i++) {
        if (fabs(expected[i] - actual[i]) > 0.0001) {
            printf("%s failed at index %d: expected %f, got %f\n", test_name, i, expected[i], actual[i]);
            return;
        }
    }
    printf("%s passed.\n", test_name);
}

// Mock the main function to run tests
int main() {
    // Test variables
    int B = 1; // Batch size
    int T = 2; // Sequence length
    int C = 2; // Channels per token
    int OC = 3*C; // Output channels (3 times C, assuming Q, K, V)
    int max_length = 5;

    // Allocate memory for inputs and outputs
    float inp[4] = {1.0, 2.0, 3.0, 4.0}; // Shape (B, T, C)
    float weight[12] = {1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0}; // Shape (OC, C)
    float bias[6] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    float out[12]; // Output buffer
    float kv_cache[20]; // KV cache with padding (B, T+max_gen_length, 2C)
    memset(kv_cache, 0, sizeof(float) * 20);

    // Initialize KV cache with dummy values
    for (int i = 0; i < 8; i++) {
        kv_cache[i] = i + 0.5;  // Preset values
    }

    // Expected output array for validation
    float expected_output[12] = {
        2.5, 3.5, 1.0, 1.5, 2.0, 2.5,  // First token Q, K, V from input; K, V from cache
        7.0, 8.5, 13.0, 15.0, 17.0, 19.0 // Second token Q, K, V all computed fresh
    };

    // Run the function
    matmul_cached(out, inp, weight, bias, kv_cache, B, T, C, OC, max_length);

    // Assert output
    assert_arrays(expected_output, out, 12, "Test Output");

    return 0;
}
