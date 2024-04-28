#define INFERENCE
#include "train_gpt2.c"


int main(int arc, char **argv) {
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    int T = 30; // sequence length
    int P = 5; // parallel generations, not using this for now
    int B = 1;

    char* tiny_stories_val = "data/TinyStories_val.bin";
    char* tiny_shakespeare_val = "data/tiny_shakespeare_val.bin";
    char* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    DataLoader val_loader;
    dataloader_init(&val_loader, val_tokens, B, T);
    printf("val dataset num_batches: %d\n", val_loader.num_batches);
    int val_num_batches = 10;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    unsigned long long rng_state = 1337;
    int* gen_tokens = (int*)malloc(B * T * sizeof(int));
    const int totalSize = 80; // number of steps of inference we will do
    // genT can be larger than T if we slide the window


    int L = model.config.num_layers;
    int C = model.config.channels;
    int C3 = C * 3;
    int C2 = C * 2;
    
    // not stochastic 

    int PROMPT_SIZE = 14;

    dataloader_next_batch(&val_loader);
    //dataloader_next_batch(&val_loader);

    // need to make the dataloader output prompts of different sizes
    // or just take a random amount bite out of the end

    //model.kv_cache = malloc_kvcache(B, L, model.config.channels, gen_max_length);
    
    // populate the kv cache with the prompt data
    //populate_kv_cache(model.acts.qkv, model.kv_cache, B, T, C, gen_max_length);

    // B SHOULD BE 1
    // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
    for(int i = 0; i < PROMPT_SIZE; i++) {
        gen_tokens[i] = val_loader.inputs[i];
    }
    for(int i = PROMPT_SIZE; i < totalSize; i++) {
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
        gpt2_forward(&model, gen_tokens, NULL, B, T);
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
        } else {
            // fall back to printing the token id
            printf("%d ", next_token);
        }
        fflush(stdout);
    }

    printf("\nStart sliding the window\n\n");

    // now continue with a sliding window past the context length
    for (int t = T; t < totalSize; t++) {
        gpt2_forward(&model, gen_tokens + (t - T), NULL, B, T);
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
            // printf("\n======================================\n");
        } else {
            // fall back to printing the token id
            printf("%d ", next_token);
        }
        fflush(stdout);
    }

    printf("\n---\n");
    printf("Finished!\n");

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Generating with no caching took (took %f ms)\n", time_elapsed_s * 1000);
    

    // free
    // dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    free(gen_tokens);
    return 0;
}