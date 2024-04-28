#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define FLOAT_TOLERANCE 1e-2

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

void populate_kv_cache(float* kv_cache, float* out, int B, int T, int C, int idx) {
    //#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = idx; t < T; t++) {
            float* out_bt_k = out + b * T * 3 * C + t * 3 * C + C;
            float* out_bt_v = out + b * T * 3 * C + t * 3 * C + 2 * C;
            float* cache_k = kv_cache + b * T * 2 * C + t * 2 * C;
            float* cache_v = kv_cache + b * T * 2 * C + t * 2 * C + C;
            for (int i = 0; i < C; i++) {
                cache_k[i] = out_bt_k[i];
                cache_v[i] = out_bt_v[i];
            }
        }
    }
}

void fill_from_kv_cache(float* out, float* kv_cache, int B, int T, int C) {
    //#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt_k = out + b * T * 3 * C + t * 3 * C + C;
            float* out_bt_v = out + b * T * 3 * C + t * 3 * C + 2 * C;
            float* cache_k = kv_cache + b * T * 2 * C + t * 2 * C;
            float* cache_v = kv_cache + b * T * 2 * C + t * 2 * C + C;
            for (int i = 0; i < C; i++) {
                out_bt_k[i] = cache_k[i];
                out_bt_v[i] = cache_v[i];
            }
        }
    }
}

void attention_cached(float* out, float* preatt, float* att,
                       float* queries, float* kv_cache,
                       int B, int T, int C, int NH) {
    // Implementation assumes kv_cache holds both keys and values for all layers
    int hs = C / NH; // head size
    int C3 = C*3;
    float scale = 1.0 / sqrtf(hs);
    int base_index_k = B * T * 2 * C + B * T * C; // Offset for keys
    int base_index_v = B * T * 2 * C + B * T * 2 * C; // Offset for values, i.e., after all keys

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* query_t = queries + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
                float* att_bth = att + b * NH * T * T + h * T * T + t * T;

                // Calculate query dot key and maxval
                float maxval = -10000.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = kv_cache + b * T * 2 * C + t2 * C + h * hs; // Direct access to keys

                    // Dot product of query_t and key_t2
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }
                    preatt_bth[t2] = val;
                }

                // Calculate the exp and keep track of sum
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // Normalize to get the softmax
                for (int t2 = 0; t2 <= t; t2++) {
                    att_bth[t2] *= expsum_inv;
                }

                // Accumulate weighted values into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) {
                    out_bth[i] = 0.0f;
                }
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = kv_cache + b * T * 2 * C + T * C + t2 * C + h * hs;
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

void attention_forward(float* out, float* preatt, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f; // TODO something better
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
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
    int equal = 1;
    for (int i = 0; i < size; i++) {
        if (fabs(arr1[i] - arr2[i]) > FLOAT_TOLERANCE) {
            //printf("Noual: out1[%d] = %f, out2[%d] = %f\n", i, arr1[i], i, arr2[i]);
            equal = 0;
        }
        else {
            //printf("Equal: out1[%d] = %f, out2[%d] = %f\n", i, arr1[i], i, arr2[i]);
        }
    }
    return equal;
}

int main() {
    int B = 5; // Batch size
    int T = 10; // Number of tokens
    int C = 64; // Channels
    int OC = 3 * C; // Output channels for Q, K, V
    int NH = 8;

    // Allocate memory for inputs and outputs
    float *inp = (float *)malloc(B * T * C * sizeof(float));
    float *weight = (float *)malloc(OC * C * sizeof(float));
    float *bias = (float *)malloc(OC * sizeof(float));
    float *preatt = (float*) malloc(B * NH * T * T * sizeof(float));
    float *att = (float*) malloc(B * NH * T * T * sizeof(float));
    float *attproj = (float*) malloc(B * T * C * sizeof(float));

    float *out1 = (float *)malloc(B * T * OC * sizeof(float));
    float *out2 = (float *)malloc(B * T * OC * sizeof(float));
    float *out_a1 = (float *)malloc(B * T * OC * sizeof(float));
    float *out_a2 = (float *)malloc(B * T * OC * sizeof(float));

    float *kv_cache = (float *)malloc(B * T * 2 * C * sizeof(float));

    // Initialize inputs with random data
    initialize_random(inp, B * T * C);
    initialize_random(weight, OC * C);
    initialize_random(bias, OC);

    // Initialize out1
    initialize_random(out1, B * T * OC);

    // Perform matrix multiplication with matmul_forward
    matmul_forward(out1, inp, weight, bias, B, T, C, OC);
    populate_kv_cache(kv_cache, out1, B, T, C, 0);

    //memcpy(out2, out1, B * T * OC * sizeof(float)); // Copy initial values from out1 to out2
    // Assume out2 already has valid data and only the last token K and V need to be updated
    fill_from_kv_cache(out2, kv_cache, B, T, C);
    matmul_cached(out2, inp, weight, bias, B, T, C, OC);

    // Compare outputs
    if (compare_arrays(out1, out2, B * T * OC)) {
        printf("The outputs are identical.\n");
    } else {
        printf("The outputs differ.\n");
    }

    attention_forward(out_a1, preatt, att, out1, B, T, C, NH);
    attention_cached(out_a2, preatt, att, out2, kv_cache, B, T, C, NH);

    if (compare_arrays(out_a1, out_a2, B * T * OC)) {
        printf("The attention outputs are identical.\n");
    } else {
        printf("The attention outputs differ.\n");
        for (int i = 0; i < 200; i++) {
            printf("out_a1[%d] = %f, out_a2[%d] = %f\n", i, out_a1[i], i, out_a2[i]);
        }
    }

    // Clean up
    free(inp);
    free(weight);
    free(bias);
    free(preatt);
    free(att);
    free(attproj);

    free(out1);
    free(out2);
    free(out_a1);
    free(out_a2);

    return 0;
}
