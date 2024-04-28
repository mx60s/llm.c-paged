#include <stdio.h>
#include <stdlib.h>

#define MAX_PROMPTS 100 // should end up being equal to batch size I guess
#define MAX_BLOCKS  1000
#define BLOCK_SIZE  32 // vary

typedef struct {
    float* keys;
    float* values;
    int filled;
    int prompt_id; 
    int lru_counter;
} kv_block;

typedef struct {
    kv_block blocks[MAX_BLOCKS];
    int prompt_block_list[MAX_PROMPTS][MAX_BLOCKS];
    int prompt_block_count[MAX_PROMPTS];
    int lru_epoch;
} BlockManager;


BlockManager* create_block_manager() {
    BlockManager* manager = (BlockManager*) malloc(sizeof(BlockManager));
    for (int i = 0; i < MAX_BLOCKS; i++) {
        manager->blocks[i].keys = NULL;
        manager->blocks[i].values = NULL;
        manager->blocks[i].prompt_id = -1;
    }
    return manager;
}

kv_block* request_block(BlockManager* manager, int prompt_id) {
    if (prompt_id < 0 || prompt_id >= MAX_PROMPTS) {
        fprintf(stderr, "Invalid prompt ID.\n");
        return NULL;
    }
    
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

    if (manager->blocks[block_index].keys == NULL) {
        manager->blocks[block_index].keys = (float*)malloc(BLOCK_SIZE * sizeof(float));
        manager->blocks[block_index].values = (float*)malloc(BLOCK_SIZE * sizeof(float));
    }
    manager->blocks[block_index].prompt_id = prompt_id;
    manager->blocks[block_index].filled = 0;
    manager->blocks[block_index].lru_counter = ++manager->lru_epoch;

    int prompt_block_count = manager->prompt_block_count[prompt_id];
    manager->prompt_block_list[prompt_id][prompt_block_count] = block_index;
    manager->prompt_block_count[prompt_id]++;

    return &manager->blocks[block_index];
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
    int lru_index = find_least_recently_used_block(manager);
    if (lru_index != -1) {
        int prompt_id = manager->blocks[lru_index].prompt_id;
        free_blocks_for_prompt(manager, prompt_id); // they clear out the entire prompt blocks in the paper
        // doubt we need to contend with that in this scope but w/e
    }
}

void free_blocks_for_prompt(BlockManager* manager, int prompt_id) {
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
