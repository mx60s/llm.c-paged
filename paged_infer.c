#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include "block_manager.c"
#ifdef OMP
#include <omp.h>
#endif


#define MAX_PROMPTS 100 // should end up being equal to batch size I guess
#define MAX_BLOCKS  100
#define BLOCK_SIZE  32 

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

void encoder_forward(float* out,
                   int* inp, float* wte, float* wpe,
                   int B, int T, int C) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            int ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            float* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            float* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalize
                float o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

// matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
void matmul_forward(float* out,
                    float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    #pragma omp parallel for collapse(2)
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
    #pragma omp parallel for
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


void attention_paged(float* out, float* preatt, float* att,
                             float* inp, float** key_blocks, float** value_blocks,
                             int B, int T, int C, int NH, int offset) {

    // NOTE: for now it is important to keep in mind that this function was written with the assumption
    // that starting in the key_blocks and value_blocks references (starting from the offset), 
    // the sequence is contiguous meaning that the blocks are ordered properly.
    // they blocks lists are basicallly the global sequence and the offset tells us where to run on.
    // the float* inp is everything else other than the key and value blocks.
    int C3 = C*3;
    int hs = C / NH; // head size per attention head
    float scale = 1.0 / sqrtf(hs);
    // int block_size = 32; 

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) { // t is the index of the token within the sequence from the point of view of the attention
            for (int h = 0; h < NH; h++) {

                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f; // TODO something better
                for (int t2 = 0; t2 <= t; t2++) { // t2 is the index of the token in the global sequence
                    // float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    float* key_t2 = key_blocks[(t2+offset) / BLOCK_SIZE] + ((t2+offset) % BLOCK_SIZE) * C + h * hs;

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
                    // float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float* value_t2 = value_blocks[(t2+offset) / BLOCK_SIZE] + ((t2+offset) % BLOCK_SIZE) * C + h * hs;
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}


#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float* out, float* inp, int N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

void residual_forward(float* out, float* inp1, float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

void softmax_forward(float* probs, float* logits, int B, int T, int V) {
    // output: probs are (B,T,V) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,V) of the unnormalized log probabilities
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // probs <- softmax(logits)
            float* logits_bt = logits + b * T * V + t * V;
            float* probs_bt = probs + b * T * V + t * V;

            // maxval is only calculated and subtracted for numerical stability
            float maxval = -10000.0f; // TODO something better
            for (int i = 0; i < V; i++) {
                if (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            for (int i = 0; i < V; i++) {
                probs_bt[i] /= sum;
            }
        }
    }
}

void crossentropy_forward(float* losses,
                          float* probs, int* targets,
                          int B, int T, int V) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,V) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // loss = -log(probs[target])
            float* probs_bt = probs + b * T * V + t * V;
            int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float* wte; // (V, C)
    float* wpe; // (maxT, C)
    float* ln1w; // (L, C)
    float* ln1b; // (L, C)
    float* qkvw; // (L, 3*C, C)
    float* qkvb; // (L, 3*C)
    float* attprojw; // (L, C, C)
    float* attprojb; // (L, C)
    float* ln2w; // (L, C)
    float* ln2b; // (L, C)
    float* fcw; // (L, 4*C, C)
    float* fcb; // (L, 4*C)
    float* fcprojw; // (L, C, 4*C)
    float* fcprojb; // (L, C)
    float* lnfw; // (C)
    float* lnfb; // (C)
} ParameterTensors;

// allocate memory for the parameters and point the individual tensors to the right places
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    float* params_memory = (float*)malloc(num_parameters * sizeof(float));
    // assign all the tensors
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 23
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* qkv; // (L, B, T, 3*C)
    float* atty; // (L, B, T, C)
    float* preatt; // (L, B, NH, T, T)
    float* att; // (L, B, NH, T, T)
    float* attproj; // (L, B, T, C)
    float* residual2; // (L, B, T, C)
    float* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, 4*C)
    float* fch_gelu; // (L, B, T, 4*C)
    float* fcproj; // (L, B, T, C)
    float* residual3; // (L, B, T, C)
    float* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* logits; // (B, T, V)
    float* probs; // (B, T, V)
    float* losses; // (B, T)
} ActivationTensors;

float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }
    float* acts_memory = (float*)malloc(num_activations * sizeof(float));
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;

typedef struct {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    size_t num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    float* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    size_t num_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    float* grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
    // kv cache stuff
    BlockManager* manager;
} GPT2;

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopen(checkpoint_path, "rb");
    if (model_file == NULL) { printf("Error opening model file\n"); exit(1); }
    int model_header[256];
    fread(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file"); exit(1); }
    if (model_header[1] != 1) { printf("Bad version in model file"); exit(1); }

    // read in hyperparameters
    size_t maxT, V, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    printf("[GPT-2]\n");
    printf("max_seq_len: %zu\n", maxT);
    printf("vocab_size: %zu\n", V);
    printf("num_layers: %zu\n", L);
    printf("num_heads: %zu\n", NH);
    printf("channels: %zu\n", C);

    // allocate space for all the parameters and read them in
    model->param_sizes[0] = V * C; // wte
    model->param_sizes[1] = maxT * C; // wpe
    model->param_sizes[2] = L * C; // ln1w
    model->param_sizes[3] = L * C; // ln1b
    model->param_sizes[4] = L * (3 * C) * C; // qkvw
    model->param_sizes[5] = L * (3 * C); // qkvb
    model->param_sizes[6] = L * C * C; // attprojw
    model->param_sizes[7] = L * C; // attprojb
    model->param_sizes[8] = L * C; // ln2w
    model->param_sizes[9] = L * C; // ln2b
    model->param_sizes[10] = L * (4 * C) * C; // fcw
    model->param_sizes[11] = L * (4 * C); // fcb
    model->param_sizes[12] = L * C * (4 * C); // fcprojw
    model->param_sizes[13] = L * C; // fcprojb
    model->param_sizes[14] = C; // lnfw
    model->param_sizes[15] = C; // lnfb

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    // read in all the parameters from file
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    fread(model->params_memory, sizeof(float), num_parameters, model_file);
    fclose(model_file);

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

// ok the only real reason this is wrapped up is because later when we have a B > 1 it's cleaner
void add_to_cache(BlockManager* manager, float* qkv, int B, int T, int C, int how_many_tokens_to_copy_from_the_end_of_sequence) {
    // this function takes in the outputs from our matmul_forward or matmul_cached pass
    // because the output can come from either pass, our input can be of variable size
    // idx keeps track of where we should start grabbing keys and values for each token
    // manager is our block manager
    // B and T and C are just used for indexing
    // after this function, we're going to be doing a separate function before attention, where
    // we grab the keys and values within the context window and pass it to attention.
    

    int b = 0; // placeholder

    KVBlock* curr_block;
    if ((curr_block = get_current_block(manager, b))) {
        printf("Got current block\n");
        if (curr_block->filled >= BLOCK_SIZE) { // need to think more carefully about this
            printf("Current block full, getting new one.\n");
            curr_block = request_block(manager, b); // TODO handle null case
        } else {
            curr_block->lru_counter = ++manager->lru_epoch;
        }
    } else {
        printf("No blocks, requesting one\n");
        curr_block = request_block(manager, b); // TODO handle null case
    }

    printf("how_many_tokens_to_copy_from_the_end_of_sequence: %d \n", how_many_tokens_to_copy_from_the_end_of_sequence);
    print_state(manager, b);

    float* block_keys = curr_block->keys;
    float* block_values = curr_block->values;

    int last_block_offset_inside_it = curr_block->filled * C;

    printf("Block keys address: %p, Block values address: %p\n", block_keys, block_values);
    printf("Offset: %d, Filled: %d\n", last_block_offset_inside_it, curr_block->filled);

    // need to add logic where if we can't fill up the whole block we loop and fill blocks until we're done
    // that being said will we ever need to do that if our block size
    // and our T are the same or T < BLOCK_SIZE?
    // let's just work under that assumption for now

    //#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) { // we're totally assuming b is 1 right now lol
        int position_in_cache = 0;
        for (int t = T-how_many_tokens_to_copy_from_the_end_of_sequence; t < T; t++) {
            float* out_bt_k = qkv + b * T * 3 * C + t * 3 * C + C;
            float* out_bt_v = qkv + b * T * 3 * C + t * 3 * C + 2 * C;
            float* cache_k = block_keys + last_block_offset_inside_it + position_in_cache * C;
            float* cache_v = block_values + last_block_offset_inside_it + position_in_cache * C;
            printf("t = %d\n", t);
            for (int i = 0; i < C; i++) {
                //printf("is this access the issue? %f\n", cache_k[i]);
                //printf("or this access the issue? %f\n", cache_v[i]);
                //printf("or this? %f\n", out_bt_k[i]);
                //printf("or this?? %f\n", out_bt_v[i]);
                cache_k[i] = out_bt_k[i];
                cache_v[i] = out_bt_v[i];
            }
            position_in_cache++;
        }
    }

    // printf("segfault here?\n");

    curr_block->filled += how_many_tokens_to_copy_from_the_end_of_sequence;
    printf("AFTER: Offset: %d, Filled: %d\n", last_block_offset_inside_it, curr_block->filled);
    print_state(manager, b);
}

void gpt2_forward(GPT2 *model, int* inputs, int* targets, size_t B, size_t T, size_t max_total, int offset) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    // convenience parameters (size_t to help prevent int overflow)
    size_t V = model->config.vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    for(int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) {
            assert(0 <= targets[i] && targets[i] < V);
        }
    }

    int first_pass = 0;
    if(model->acts_memory == NULL) {
        first_pass = 1; // TODO don't forget that we'll need to change this for more than one prompt
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        model->act_sizes[0] = B * T * C; // encoded
        model->act_sizes[1] = L * B * T * C; // ln1
        model->act_sizes[2] = L * B * T;  // ln1_mean
        model->act_sizes[3] = L * B * T;  // ln1_rstd
        model->act_sizes[4] = L * B * T * 3*C; // qkv
        model->act_sizes[5] = L * B * T * C;  // atty
        model->act_sizes[6] = L * B * NH * T * T;  // preatt
        model->act_sizes[7] = L * B * NH * T * T;  // att
        model->act_sizes[8] = L * B * T * C; // attproj
        model->act_sizes[9] = L * B * T * C; // residual2
        model->act_sizes[10] = L * B * T * C; // ln2
        model->act_sizes[11] = L * B * T; // ln2_mean
        model->act_sizes[12] = L * B * T; // ln2_rstd
        model->act_sizes[13] = L * B * T * 4*C; // fch
        model->act_sizes[14] = L * B * T * 4*C; // fch_gelu
        model->act_sizes[15] = L * B * T * C; // fcproj
        model->act_sizes[16] = L * B * T * C; // residual3
        model->act_sizes[17] = B * T * C; // lnf
        model->act_sizes[18] = B * T; // lnf_mean
        model->act_sizes[19] = B * T; // lnf_rstd
        model->act_sizes[20] = B * T * V; // logits
        model->act_sizes[21] = B * T * V; // probs
        model->act_sizes[22] = B * T; // losses
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        // printf("num_activations: %zu\n", num_activations);
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        model->inputs = (int*)malloc(B * T * sizeof(int));
        model->targets = (int*)malloc(B * T * sizeof(int)); // might be unused if we never have targets but it's small
    } else {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        if (B != model->batch_size || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (int)B, (int)T);
            exit(EXIT_FAILURE);
        }
    }

    // cache the inputs/targets
    memcpy(model->inputs, inputs, B * T * sizeof(int));
    if (targets != NULL) {
        memcpy(model->targets, targets, B * T * sizeof(int));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    float* residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
    //printf("starting layers (just one for now)\n");
    for (int l = 0; l < 1; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcb = params.fcb + l * 4*C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        float* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_preatt = acts.preatt + l * B * NH * T * T;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_attproj = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        float* l_fcproj = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;

        // now do the forward pass
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);

        int how_many_tokens_to_copy_from_the_end_of_sequence;
        if (first_pass) { // need to update first pass logic because won't work with multiple rounds of prompts 
        // and actually I think this is probably why we need to have the first pass be like, a full first pass so we can set it appropriately
        // maybe you could just set the flag per new prompt I guess
            // only called the first time, to populate the cache
            matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
            how_many_tokens_to_copy_from_the_end_of_sequence = T;
        } else {
            matmul_cached(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
            how_many_tokens_to_copy_from_the_end_of_sequence = 1;
        }

        add_to_cache(model->manager, l_qkv, B, T, C, how_many_tokens_to_copy_from_the_end_of_sequence);

        int num_blocks; // I'm leaving this alone but don't think I need it for now
        float*** kv_blocks = collect_kv_blocks(model->manager, 0, &num_blocks);

        attention_paged(l_atty, l_preatt, l_att, l_qkv, kv_blocks[0], kv_blocks[1], B, T, C, NH, offset);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }

    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, V);
    softmax_forward(acts.probs, acts.logits, B, T, V);
}

void gpt2_zero_grad(GPT2 *model) {
    if(model->grads_memory != NULL) { memset(model->grads_memory, 0, model->num_parameters * sizeof(float)); }
    if(model->grads_acts_memory != NULL) { memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float)); }
}

void gpt2_free(GPT2 *model) {
    free(model->params_memory);
    free(model->grads_memory);
    free(model->m_memory);
    free(model->v_memory);
    free(model->acts_memory);
    free(model->grads_acts_memory);
    free(model->inputs);
    free(model->targets);
}

// if we are TESTING (see test_gpt2.c), we'll skip the int main below

// ----------------------------------------------------------------------------
// data loader lite
// returns random batches of data from a file of integers

typedef struct {
    // hyperparameters
    int B; // batch size
    int T; // sequence length
    // input handling and its state
    FILE* tokens_file;
    long file_size;
    long current_position;
    // output memory
    int* batch;
    int* inputs;
    int* targets;
    // convenience variables
    int num_batches;
} DataLoader;

void dataloader_init(DataLoader *loader, const char* filename, int B, int T) {
    loader->B = B;
    loader->T = T;

    // open the input file for reading
    loader->tokens_file = fopen(filename, "rb");
    if (loader->tokens_file == NULL) {
        printf("Error opening tokens file\n");
        exit(1);
    }

    // determine the file size
    fseek(loader->tokens_file, 0, SEEK_END);
    loader->file_size = ftell(loader->tokens_file);
    fseek(loader->tokens_file, 0, SEEK_SET);
    if (loader->file_size < (B * T + 1) * sizeof(int)) {
        printf("Error: file size is too small for the batch size and sequence length\n");
        exit(1);
    }
    loader->current_position = 0; // start at the beginning

    // allocate space for B*T + 1 integers to store the inputs and targets
    loader->batch = (int*) malloc((B * T + 1) * sizeof(int));
    loader->inputs = loader->batch;
    loader->targets = loader->batch + 1; // targets are shifted by one
    loader->num_batches = loader->file_size / (B * T * sizeof(int));
}

void dataloader_reset(DataLoader *loader) {
    loader->current_position = 0;
}

void dataloader_next_batch(DataLoader *loader) {
    int B = loader->B;
    int T = loader->T;
    // if we are at the end of the file, loop back to the beginning
    if (loader->current_position + (B*T+1) * sizeof(int) > loader->file_size) {
        loader->current_position = 0;
    }
    // read the B*T+1 integers from the file into batch
    fseek(loader->tokens_file, loader->current_position, SEEK_SET);
    fread(loader->batch, sizeof(int), B*T+1, loader->tokens_file);
    // advance the current position by B*T integers
    loader->current_position += B*T * sizeof(int);
}

void dataloader_free(DataLoader *loader) {
    fclose(loader->tokens_file);
    free(loader->batch);
}

// ----------------------------------------------------------------------------
// sampler

// the GPT-2 end-of-text token id
#define GPT2_EOT 50256

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// Tokenizer (only supports decoding)

typedef struct {
    uint32_t vocab_size;
    char **token_table;
    int init_ok;
} Tokenizer;

void safe_printf(const char *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // weird byte, don't print it
        }
    }
    printf("%s", piece);
}

void tokenizer_init(Tokenizer *tokenizer, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        // try to be more helpful as we just added this feature, erase later
        printf("---\n");
        printf("WARNING: Failed to open the tokenizer file %s\n", filename);
        printf("The Tokenizer is a new feature added April 14 2024.\n");
        printf("Re-run `python train_gpt2.py` to write it\n");
        printf("---\n");
        tokenizer->init_ok = 0;
        return;
    }
    // read in the header
    uint32_t header[256];
    fread(header, sizeof(uint32_t), 256, file);
    assert(header[0] == 20240328);
    assert(header[1] == 1);
    tokenizer->vocab_size = header[2];
    // read in all the tokens
    unsigned char length;
    tokenizer->token_table = (char **)malloc(tokenizer->vocab_size * sizeof(char *));
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        fread(&length, sizeof(unsigned char), 1, file);
        assert(length > 0); // every token should be at least one character
        char *token_bytes = (char *)malloc(length + 1);
        fread(token_bytes, sizeof(char), length, file);
        token_bytes[length] = '\0';  // Add null terminator for printing
        tokenizer->token_table[i] = token_bytes;
    }
    // cleanups
    fclose(file);
    tokenizer->init_ok = 1;
}

const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    if (tokenizer->init_ok == 0) {
        return NULL;
    }
    if (token_id < tokenizer->vocab_size) {
        return tokenizer->token_table[token_id];
    } else {
        printf("invalid token id %d!\n", token_id);
        return NULL;
    }
}

void tokenizer_free(Tokenizer *tokenizer) {
    if (tokenizer->init_ok) {
        for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            free(tokenizer->token_table[i]);
        }
        free(tokenizer->token_table);
    }
}

void print_generated_sequence(int* tokens, int B, int T) {
    for (int i = 0; i < B * T; i++) {
        printf("%d ", tokens[i]);
        if ((i+1) % T == 0) printf("\n");
    }
}

int* generate_tokens_from_logits(float* probs, int B, int T, int V) {
    int* tokens = malloc(B * T * sizeof(int));
    for (int i = 0; i < B * T; i++) {
        int max_idx = 0;
        float max_val = probs[i * V];
        for (int v = 1; v < V; v++) {
            if (probs[i * V + v] > max_val) {
                max_val = probs[i * V + v];
                max_idx = v;
            }
        }
        tokens[i] = max_idx;
    }
    return tokens;
}

int main(int arc, char **argv) {
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    int T = 32; // sequence length, fits in one block size
    int P = 5; // parallel generations, not using this for now
    int B = 1;
    int PROMPT_SIZE = 32;

    char* tiny_stories_val = "data/TinyStories_val.bin";
    char* tiny_shakespeare_val = "data/tiny_shakespeare_val.bin";
    char* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    DataLoader val_loader;
    dataloader_init(&val_loader, val_tokens, B, T);
    printf("val dataset num_batches: %d\n", val_loader.num_batches);
    //int val_num_batches = 10;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    unsigned long long rng_state = 1337;
    const int totalSize = 50;//300; // number of steps of inference we will do
    int* gen_tokens = (int*)malloc(B * totalSize * sizeof(int));
    // genT can be larger than T if we slide the window

    int C = model.config.channels;

    printf("T (sequence length): %d\n", T);
    printf("PROMPT_SIZE: %d\n", PROMPT_SIZE);
    printf("totalSize: %d\n", totalSize);

    BlockManager* block_manager = create_block_manager(C);
    model.manager = block_manager;
    printf(" ------ State before running -------\n");
    print_state(block_manager, 0);
    
    // prompt idx = idx within batch
    // for now, it's always 1
    // maybe need to consider how this works with more in the dataloader

    dataloader_next_batch(&val_loader);

    // need to make the dataloader output prompts of different sizes
    // or just take a random amount bite out of the end

    // B SHOULD BE 1 for now
    // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
    for(int i = 0; i < B * PROMPT_SIZE; ++i) {
        gen_tokens[i] = val_loader.inputs[i];
    }
    for(int i = B*PROMPT_SIZE; i < B * totalSize; ++i) {
        gen_tokens[i] = GPT2_EOT;
    }

    printf("==============Prompt:==================\n");
    for (int i = 0; i < B * totalSize; i++) {
        const char* token_str = tokenizer_decode(&tokenizer, gen_tokens[i]);
        safe_printf(token_str);
    }
    printf("\n========================================\n");
    fflush(stdout);


    // now sample from the model autoregressively
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    printf("\ngenerating:\n---\n");

    //for (int i = 0; i < PROMPT_SIZE; i++) {
    //    const char* token_str = tokenizer_decode(&tokenizer, gen_tokens[i]);
    //    safe_printf(token_str);
    //}
    // first loop we fill up the context
    for (int t = PROMPT_SIZE; t < T; t++) {
        gpt2_forward(&model, gen_tokens, NULL, B, T, totalSize, 0);
        // printf("finished forward\n");
        float* probs = model.acts.probs + (t-1) * model.config.vocab_size;
        float coin = random_f32(&rng_state);
        int next_token = sample_mult(probs, model.config.vocab_size, coin);
        gen_tokens[t] = next_token;

        // print the generated token, either using the Tokenizer or a fallback
        if (tokenizer.init_ok) {
            // for (int i = 0; i < B * totalSize; i++) {
            //     const char* token_str = tokenizer_decode(&tokenizer, gen_tokens[i]);
            //     safe_printf(token_str);
            // }
            const char* token_str = tokenizer_decode(&tokenizer, next_token);
            safe_printf(token_str);
            fflush(stdout);
        } else {
            // fall back to printing the token id
            printf("%d ", next_token);
        }
        fflush(stdout);
    }

    printf("\nStart sliding the window\n\n");

    // now continue with a sliding window past the context length
    for (int t = T; t < totalSize; t++) {
        //gpt2_forward(&model, gen_tokens + (t - T), NULL, B, T, totalSize, t - T);
        gpt2_forward(&model, gen_tokens + (t - T), NULL, B, T, totalSize, t - T);
        // printf("finished forward\n");
        // either T-1 or T not sure
        float* probs = model.acts.probs + (T-1) * model.config.vocab_size;
        float coin = random_f32(&rng_state);
        int next_token = sample_mult(probs, model.config.vocab_size, coin);
        gen_tokens[t] = next_token;

        // print the generated token, either using the Tokenizer or a fallback
        if (tokenizer.init_ok) {
            // for (int i = 0; i < B * totalSize; i++) {
            //     const char* token_str = tokenizer_decode(&tokenizer, gen_tokens[i]);
            //     safe_printf(token_str);
            // }
            const char* token_str = tokenizer_decode(&tokenizer, next_token);
            safe_printf(token_str);
            printf("\n"); // this might fuck it up be careful
            // printf("\n======================================\n");
        } else {
            // fall back to printing the token id
            printf("%d", next_token);
        }
        fflush(stdout);
    }

    printf("\n---\n");
    printf("Finished!\n");

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Generating with paged caching took (took %f ms)\n", time_elapsed_s * 1000);

    for (int i = 0; i < totalSize; i++) {
        const char* token_str = tokenizer_decode(&tokenizer, gen_tokens[i]);
        safe_printf(token_str);
    }    

    // free
    // dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    free(gen_tokens);
    return 0;
}