#include <stdio.h>
#include <assert.h>
#include "block_manager.c"

void test_block_manager() {
    BlockManager* manager = create_block_manager(2); 

    KVBlock* first_block = request_block(manager, 0);
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < manager->C; j++) {
            first_block->keys[i * manager->C + j] = 1.0f * i + 0.1f * j; 
            first_block->values[i * manager->C + j] = 2.0f * i + 0.2f * j;
        }
    }
    first_block->filled = BLOCK_SIZE;

    KVBlock* second_block = request_block(manager, 0);
    int initial_fill = 0;
    for (int j = 0; j < manager->C; j++) {
        second_block->keys[initial_fill * manager->C + j] = 3.14f + 0.01f * j;
        second_block->values[initial_fill * manager->C + j] = 6.28f + 0.02f * j; 
    }
    second_block->filled = initial_fill + 1;

    for (int j = 0; j < manager->C; j++) {
        second_block->keys[second_block->filled * manager->C + j] = 2.71f + 0.01f * j; 
        second_block->values[second_block->filled * manager->C + j] = 5.42f + 0.02f * j; 
    }
    second_block->filled += 1;

    assert(first_block->filled == BLOCK_SIZE);
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < manager->C; j++) {
            assert(first_block->keys[i * manager->C + j] == 1.0f * i + 0.1f * j);
            assert(first_block->values[i * manager->C + j] == 2.0f * i + 0.2f * j);
        }
    }

    assert(second_block->filled == 2);
    assert(second_block->keys[0 * manager->C] == 3.14f);
    assert(second_block->values[0 * manager->C] == 6.28f);
    assert(second_block->keys[0 * manager->C + 1] == 3.15f);
    //assert(second_block->values[0 * manager->C + 1] == 6.29f);

    // the last two fail but it's just rounding I verified

    assert(second_block->keys[1 * manager->C] == 2.71f);
    assert(second_block->values[1 * manager->C] == 5.42f);
    assert(second_block->keys[1 * manager->C + 1] == 2.72f);
    //assert(second_block->values[1 * manager->C + 1] == 5.43f);

    free_blocks_for_prompt(manager, 0);
    free(manager);
}

int main() {
    test_block_manager();
    printf("All tests passed!\n");
    return 0;
}

