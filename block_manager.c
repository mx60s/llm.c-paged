#include <stdio.h>
#include <stdlib.h>

#define MAX_PROMPTS 100 // should end up being equal to batch size I guess
#define MAX_BLOCKS  100
#define BLOCK_SIZE  32 // vary


typedef struct {
    float* keys;
    float* values;
    int filled;
    int prompt_id; 
    int lru_counter;
} KVBlock;

typedef struct {
    int C;
    KVBlock blocks[MAX_BLOCKS];
    int prompt_block_list[MAX_PROMPTS][MAX_BLOCKS];
    int prompt_block_count[MAX_PROMPTS];
    int lru_epoch;
} BlockManager;

void print_state(BlockManager* manager, int prompt) {
    printf("Block manager llru %d\n", manager->lru_epoch);

    int prompt_blocks = manager->prompt_block_count[prompt];
    printf("Prompt %d block count: %d\n", prompt, prompt_blocks);

    int block_id;
    for (int i = 0; i < prompt_blocks; i++) {
        block_id = manager->prompt_block_list[prompt][i];
        printf("Block %d: filled %d, llru %d\n", block_id, manager->blocks[block_id].filled, manager->blocks[block_id].lru_counter);
    }
}

BlockManager* create_block_manager(int channels) {
    BlockManager* manager = (BlockManager*) malloc(sizeof(BlockManager));
    manager->C = channels;

    for (int i = 0; i < MAX_PROMPTS; i++) {
        manager->prompt_block_count[i] = 0;
    }

    for (int i = 0; i < MAX_BLOCKS; i++) {
        manager->blocks[i].keys = NULL;
        manager->blocks[i].values = NULL;
        manager->blocks[i].prompt_id = -1;
    }
    return manager;
}

int get_next_block_id(BlockManager *manager, int prompt, int block_id) {
    int* prompt_blocks = manager->prompt_block_list[prompt];

    for (int i = 0; i < MAX_BLOCKS; i++) {
        if (prompt_blocks[i] == block_id && i + 1 < MAX_BLOCKS) {
            return prompt_blocks[i+1];
        }
    }
    return -1;
}

KVBlock* get_current_block(BlockManager* manager, int prompt_id) {
    int num_blocks = manager->prompt_block_count[prompt_id];
    printf("Current num_blocks in prompt is %d\n", num_blocks);
    if (num_blocks == 0) {
        return NULL;
    }
    
    int curr_block_idx = manager->prompt_block_list[prompt_id][num_blocks - 1];
    printf("Current block idx is %d\n", curr_block_idx);

    return &manager->blocks[curr_block_idx];
}

void free_blocks_for_prompt(BlockManager* manager, int prompt_id) {
    printf("Freeing all blocks for prompt\n");
    for (int i = 0; i < manager->prompt_block_count[prompt_id]; i++) {
        int block_index = manager->prompt_block_list[prompt_id][i];
        free(manager->blocks[block_index].keys);
        free(manager->blocks[block_index].values);
        manager->blocks[block_index].keys = NULL;
        manager->blocks[block_index].values = NULL;
        manager->blocks[block_index].filled = 0;
        manager->blocks[block_index].prompt_id = -1;
    }
    manager->prompt_block_count[prompt_id] = 0;
}

int find_least_recently_used_block(BlockManager* manager) {
    int lru_index = -1;
    int min_lru_counter = manager->lru_epoch;
    for (int i = 0; i < MAX_BLOCKS; i++) {
        if (manager->blocks[i].prompt_id != -1 && manager->blocks[i].lru_counter < min_lru_counter) {
            min_lru_counter = manager->blocks[i].lru_counter;
            lru_index = i;
        }
    }
    return lru_index;
}

void page_out_lru_block(BlockManager* manager) {
    printf("Paging out lru block\n");
    int lru_index = find_least_recently_used_block(manager);
    if (lru_index != -1) {
        int prompt_id = manager->blocks[lru_index].prompt_id;
        free_blocks_for_prompt(manager, prompt_id); // they clear out the entire prompt blocks in the paper
        // doubt we need to contend with that in this scope but w/e
        // if you do hashing you need to change up this policy
    }
}

KVBlock* request_block(BlockManager* manager, int prompt_id) {
    if (prompt_id < 0 || prompt_id >= MAX_PROMPTS) {
        fprintf(stderr, "Invalid prompt ID.\n");
        return NULL;
    }
    
    // find first available block
    int block_index = -1;
    for (int i = 0; i < MAX_BLOCKS; i++) {
        if (manager->blocks[i].prompt_id == -1) { 
            block_index = i;
            break;
        }
    }

    if (block_index == -1) {
        page_out_lru_block(manager);
        for (int i = 0; i < MAX_BLOCKS; i++) {
            if (manager->blocks[i].prompt_id == -1) {
                block_index = i;
                break;
            }
        }
        if (block_index == -1) {
            fprintf(stderr, "No blocks available.\n");
            return NULL;
        }
    }


    manager->blocks[block_index].keys = (float*)malloc(BLOCK_SIZE * manager->C * sizeof(float));
    manager->blocks[block_index].values = (float*)malloc(BLOCK_SIZE * manager->C * sizeof(float));

    if (manager->blocks[block_index].keys == NULL || manager->blocks[block_index].values == NULL) {
        fprintf(stderr, "Failed to allocate memory for block keys/values.\n");
        return NULL;  // Ensure you handle this NULL return properly in the caller.
    }

    manager->blocks[block_index].prompt_id = prompt_id;
    manager->blocks[block_index].filled = 0;
    manager->blocks[block_index].lru_counter = ++manager->lru_epoch;

    int prompt_block_count = manager->prompt_block_count[prompt_id];
    manager->prompt_block_list[prompt_id][prompt_block_count] = block_index;
    manager->prompt_block_count[prompt_id]++;

    return &manager->blocks[block_index];
}

// a really artless way to do this but whatever for now
float*** collect_kv_blocks(BlockManager* manager, int prompt_id, int* num_blocks) {
    if (prompt_id < 0 || prompt_id >= MAX_PROMPTS) {
        fprintf(stderr, "Invalid prompt ID.\n");
        return NULL;
    }

    *num_blocks = manager->prompt_block_count[prompt_id];
    if (*num_blocks == 0) {
        return NULL;
    }

    float*** kv_pointers = (float***)malloc(2 * sizeof(float**));
    if (kv_pointers == NULL) {
        fprintf(stderr, "Memory allocation failed for kv_pointers.\n");
        return NULL;
    }

    // keys, values in order
    kv_pointers[0] = (float**)malloc(*num_blocks * sizeof(float*));
    kv_pointers[1] = (float**)malloc(*num_blocks * sizeof(float*));

    if (kv_pointers[0] == NULL || kv_pointers[1] == NULL) {
        fprintf(stderr, "Memory allocation failed for key/value pointers.\n");
        free(kv_pointers[0]);
        free(kv_pointers[1]);
        free(kv_pointers);
        return NULL;
    }

    for (int i = 0; i < *num_blocks; i++) {
        int block_idx = manager->prompt_block_list[prompt_id][i];
        kv_pointers[0][i] = manager->blocks[block_idx].keys;
        kv_pointers[1][i] = manager->blocks[block_idx].values;
    }

    return kv_pointers;
}