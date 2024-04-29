#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FLOAT_TOLERANCE 1e-2

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

void attention_paged(float* out, float* preatt, float* att,
                       float* inp,
                       float** key_blocks, float** value_blocks,
                       int B, int T, int C, int NH, int num_blocks) {
    // Input is (B, T, C) holding the query vectors
    // key_blocks and value_blocks are arrays of pointers to the key and value blocks
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // Output is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size per attention head
    float scale = 1.0 / sqrtf(hs);
    int block_size = T / num_blocks;

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int block_index_t = t / block_size; // block index for the query token

            for (int h = 0; h < NH; h++) {
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }

                for (int block_index = 0; block_index <= block_index_t; block_index++) {
                    float* key_block = key_blocks[block_index];
                    float* value_block = value_blocks[block_index];

                    float maxval = -10000.0f;
                    float expsum = 0.0f;

                    for (int t2 = 0; t2 < block_size; t2++) {
                        float* key_t2 = key_block + t2 * hs;
                        float val = 0.0f;
                        for (int i = 0; i < hs; i++) {
                            val += query_t[i] * key_t2[i];
                        }
                        val *= scale;
                        preatt_bth[t2 + block_index * block_size] = val;
                        if (val > maxval) maxval = val;
                    }

                    for (int t2 = 0; t2 < block_size; t2++) {
                        float expv = expf(preatt_bth[t2 + block_index * block_size] - maxval);
                        expsum += expv;
                        att_bth[t2 + block_index * block_size] = expv;
                    }

                    float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                    for (int t2 = 0; t2 < block_size; t2++) {
                        att_bth[t2 + block_index * block_size] *= expsum_inv;
                        float* value_t2 = value_block + t2 * hs;
                        for (int i = 0; i < hs; i++) {
                            out_bth[i] += att_bth[t2 + block_index * block_size] * value_t2[i];
                        }
                    }
                }
            }
        }
    }
}


int compare_arrays(float *arr1, float *arr2, int size) {
    int equal = 1;
    for (int i = 0; i < size; i++) {
        if (fabs(arr1[i] - arr2[i]) > FLOAT_TOLERANCE) {
            printf("Noual: out1[%d] = %f, out2[%d] = %f\n", i, arr1[i], i, arr2[i]);
            equal = 0;
        }
        else {
            //printf("Equal: out1[%d] = %f, out2[%d] = %f\n", i, arr1[i], i, arr2[i]);
        }
    }
    return equal;
}

void print_array(float* array, int size) {
    for (int i = 0; i < size; i++) {
        printf("%f ", array[i]);
    }
    printf("\n");
}


int main() {
    int B = 1;  
    int T = 5; 
    int C = 1; 
    int NH = 1; 
    int num_blocks = 5;

    float* inp = (float*)malloc(B * T * 3 * C * sizeof(float));
    float* out_standard = (float*)malloc(B * T * C * sizeof(float));
    float* out_block = (float*)malloc(B * T * C * sizeof(float));
    float* preatt = (float*)malloc(B * NH * T * T * sizeof(float));
    float* att = (float*)malloc(B * NH * T * T * sizeof(float));
    
    for (int i = 0; i < B * T * 3 * C; i++) {
        inp[i] = (float)1.0;
    }

    float** key_blocks = malloc(num_blocks * sizeof(float*));
    float** value_blocks = malloc(num_blocks * sizeof(float*));

    int block_size = T / num_blocks;
    for (int i = 0; i < num_blocks; i++) {
        key_blocks[i] = malloc(block_size * C * sizeof(float));
        value_blocks[i] = malloc(block_size * C * sizeof(float));

        for (int j = 0; j < block_size * C; j++) {
            key_blocks[i][j] = inp[i * block_size * C + C + j]; 
            value_blocks[i][j] = inp[i * block_size * C + 2 * C + j]; 
        }
    }

    attention_forward(out_standard, preatt, att, inp, B, T, C, NH);

    attention_paged(out_block, preatt, att, inp, key_blocks, value_blocks, B, T, C, NH, num_blocks);

    if (compare_arrays(out_standard, out_block, B * T * C)) {
        printf("equivalent\n");
    } else {
        printf("differ\n");
        printf("Standard\n");
        print_array(out_standard, B * T * C);
        printf("Block\n");
        print_array(out_block, B * T * C);
    }

    free(inp);
    free(out_standard);
    free(out_block);
    free(preatt);
    free(att);
    for (int i = 0; i < num_blocks; i++) {
        free(key_blocks[i]);
        free(value_blocks[i]);
    }
    free(key_blocks);
    free(value_blocks);

    return 0;
}
