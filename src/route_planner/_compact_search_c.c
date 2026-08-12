#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

typedef struct {
    int64_t node_count;
    int64_t edge_count;
    int64_t traversal_count;
    const int64_t* forward_indptr;
    const int32_t* forward_indices;
    const int64_t* reverse_indptr;
    const int32_t* reverse_indices;
    const int64_t* reverse_positions;
    const double* trav_travel_time;
    const uint8_t* trav_highway_mask;
    const double* trav_scenic_score;
    const uint8_t* trav_scenic_byway_mask;
    const int32_t* trav_edge_rank;
    const uint8_t* trav_reverse;
} CompactGraphSpec;

typedef struct {
    double scenic_weight;
    int strict_highways;
    double highway_preference;
    double travel_weight;
    double scenic_reward;
    double highway_penalty;
    double scenic_byway_bonus;
    double lagrangian_multiplier;
    double cost_limit;
} CostSpec;

typedef struct {
    int32_t node_rank;
    double dist;
} HeapItem;

typedef struct {
    HeapItem* data;
    int32_t capacity;
    int32_t size;
} MinHeap;

static MinHeap* heap_create(int32_t capacity) {
    MinHeap* h = (MinHeap*)malloc(sizeof(MinHeap));
    h->capacity = capacity > 16 ? capacity : 16;
    h->size = 0;
    h->data = (HeapItem*)malloc(sizeof(HeapItem) * h->capacity);
    return h;
}

static void heap_free(MinHeap* h) {
    if (h) {
        free(h->data);
        free(h);
    }
}

static void heap_push(MinHeap* h, int32_t node_rank, double dist) {
    if (h->size >= h->capacity) {
        h->capacity *= 2;
        h->data = (HeapItem*)realloc(h->data, sizeof(HeapItem) * h->capacity);
    }
    int32_t i = h->size++;
    while (i > 0) {
        int32_t p = (i - 1) / 2;
        if (h->data[p].dist <= dist) break;
        h->data[i] = h->data[p];
        i = p;
    }
    h->data[i].node_rank = node_rank;
    h->data[i].dist = dist;
}

static bool heap_pop(MinHeap* h, int32_t* out_node, double* out_dist) {
    if (h->size == 0) return false;
    *out_node = h->data[0].node_rank;
    *out_dist = h->data[0].dist;
    HeapItem last = h->data[--h->size];
    if (h->size > 0) {
        int32_t i = 0;
        while (i * 2 + 1 < h->size) {
            int32_t left = i * 2 + 1;
            int32_t right = left + 1;
            int32_t smallest = left;
            if (right < h->size && h->data[right].dist < h->data[left].dist) {
                smallest = right;
            }
            if (last.dist <= h->data[smallest].dist) break;
            h->data[i] = h->data[smallest];
            i = smallest;
        }
        h->data[i] = last;
    }
    return true;
}

typedef struct {
    int32_t node_rank;
    int32_t parent_rank;
    int64_t trav_pos;
    double dist;
} NodeState;

typedef struct {
    NodeState* entries;
    int32_t capacity;
    int32_t count;
    int32_t mask;
} HashTable;

static HashTable* hash_create(int32_t capacity) {
    HashTable* ht = (HashTable*)malloc(sizeof(HashTable));
    int32_t cap = 16;
    while (cap < capacity) cap *= 2;
    ht->capacity = cap;
    ht->mask = cap - 1;
    ht->count = 0;
    ht->entries = (NodeState*)calloc(cap, sizeof(NodeState));
    for (int32_t i = 0; i < cap; i++) {
        ht->entries[i].node_rank = -1;
    }
    return ht;
}

static void hash_free(HashTable* ht) {
    if (ht) {
        free(ht->entries);
        free(ht);
    }
}

static NodeState* hash_get(HashTable* ht, int32_t node_rank) {
    uint32_t h = (uint32_t)node_rank * 0x9e3779b9u;
    int32_t idx = h & ht->mask;
    while (ht->entries[idx].node_rank != -1) {
        if (ht->entries[idx].node_rank == node_rank) {
            return &ht->entries[idx];
        }
        idx = (idx + 1) & ht->mask;
    }
    return NULL;
}

static void hash_grow(HashTable* ht);

static NodeState* hash_put(HashTable* ht, int32_t node_rank, int32_t parent_rank, int64_t trav_pos, double dist) {
    if (ht->count * 2 >= ht->capacity) {
        hash_grow(ht);
    }
    uint32_t h = (uint32_t)node_rank * 0x9e3779b9u;
    int32_t idx = h & ht->mask;
    while (ht->entries[idx].node_rank != -1) {
        if (ht->entries[idx].node_rank == node_rank) {
            ht->entries[idx].parent_rank = parent_rank;
            ht->entries[idx].trav_pos = trav_pos;
            ht->entries[idx].dist = dist;
            return &ht->entries[idx];
        }
        idx = (idx + 1) & ht->mask;
    }
    ht->entries[idx].node_rank = node_rank;
    ht->entries[idx].parent_rank = parent_rank;
    ht->entries[idx].trav_pos = trav_pos;
    ht->entries[idx].dist = dist;
    ht->count++;
    return &ht->entries[idx];
}

static void hash_grow(HashTable* ht) {
    int32_t old_cap = ht->capacity;
    NodeState* old_entries = ht->entries;
    ht->capacity *= 2;
    ht->mask = ht->capacity - 1;
    ht->entries = (NodeState*)calloc(ht->capacity, sizeof(NodeState));
    for (int32_t i = 0; i < ht->capacity; i++) {
        ht->entries[i].node_rank = -1;
    }
    ht->count = 0;
    for (int32_t i = 0; i < old_cap; i++) {
        if (old_entries[i].node_rank != -1) {
            hash_put(ht, old_entries[i].node_rank, old_entries[i].parent_rank, old_entries[i].trav_pos, old_entries[i].dist);
        }
    }
    free(old_entries);
}

static inline double compute_edge_cost(const CompactGraphSpec* g, const CostSpec* c, int64_t pos) {
    if (c->strict_highways && g->trav_highway_mask[pos]) {
        return INFINITY;
    }
    double duration = g->trav_travel_time[pos];
    if (duration < 0.0 || !isfinite(duration)) return INFINITY;

    if (c->scenic_weight == 0.0 && c->highway_preference == 0.0 &&
        c->travel_weight == 1.0 && c->scenic_reward == 0.0 &&
        c->highway_penalty == 0.0 && c->scenic_byway_bonus == 0.0) {
        double cost = duration;
        if (c->lagrangian_multiplier > 0.0) {
            cost += c->lagrangian_multiplier * duration;
        }
        return cost;
    }

    double raw_score = g->trav_scenic_score ? g->trav_scenic_score[pos] : 5.0;
    if (!isfinite(raw_score)) raw_score = 0.0;
    if (raw_score < 0.0) raw_score = 0.0;
    if (raw_score > 10.0) raw_score = 10.0;

    double weighted_base = (1.0 - c->scenic_weight) * duration * c->travel_weight
        + c->scenic_weight * duration * (10.0 - raw_score) / 10.0 * c->scenic_reward;
    double base = weighted_base > 1e-6 * duration ? weighted_base : 1e-6 * duration;

    if (g->trav_scenic_byway_mask && g->trav_scenic_byway_mask[pos]) {
        double bonus = c->scenic_byway_bonus > 0.5 ? 0.5 : c->scenic_byway_bonus;
        if (bonus < 0.0) bonus = 0.0;
        base *= (1.0 - bonus);
    }

    double adjustment = 0.0;
    if (c->highway_preference > 0.0 && g->trav_highway_mask[pos]) {
        adjustment = duration * c->highway_preference;
    } else if (c->strict_highways && g->trav_highway_mask[pos]) {
        adjustment = duration * c->highway_penalty;
    }

    double cost = base + adjustment;
    if (cost < 1e-6) cost = 1e-6;
    if (!isfinite(cost)) cost = 1e-6;

    if (c->lagrangian_multiplier > 0.0) {
        cost += c->lagrangian_multiplier * duration;
    }
    return cost;
}

int32_t compact_bidirectional_search_alloc(
    const CompactGraphSpec* graph,
    const CostSpec* cost_spec,
    const int32_t* fwd_seed_nodes,
    const double* fwd_seed_costs,
    int32_t fwd_seed_count,
    const int32_t* rev_seed_nodes,
    const double* rev_seed_costs,
    int32_t rev_seed_count,
    int64_t** out_trav_positions,
    double* out_total_cost,
    int32_t* out_fwd_seed_index,
    int32_t* out_rev_seed_index
) {
    *out_trav_positions = NULL;
    MinHeap* heap_f = heap_create(128);
    MinHeap* heap_r = heap_create(128);
    HashTable* map_f = hash_create(256);
    HashTable* map_r = hash_create(256);

    for (int32_t i = 0; i < fwd_seed_count; i++) {
        int32_t node = fwd_seed_nodes[i];
        double cost = fwd_seed_costs[i];
        if (cost_spec->cost_limit > 0.0 && cost > cost_spec->cost_limit) continue;
        hash_put(map_f, node, -1, -1, cost);
        heap_push(heap_f, node, cost);
    }
    for (int32_t i = 0; i < rev_seed_count; i++) {
        int32_t node = rev_seed_nodes[i];
        double cost = rev_seed_costs[i];
        if (cost_spec->cost_limit > 0.0 && cost > cost_spec->cost_limit) continue;
        hash_put(map_r, node, -1, -1, cost);
        heap_push(heap_r, node, cost);
    }

    double best_cost = INFINITY;
    int32_t best_meeting_node = -1;

    while (heap_f->size > 0 && heap_r->size > 0) {
        int32_t top_node_f = heap_f->data[0].node_rank;
        double top_dist_f = heap_f->data[0].dist;
        int32_t top_node_r = heap_r->data[0].node_rank;
        double top_dist_r = heap_r->data[0].dist;

        if (top_dist_f + top_dist_r >= best_cost) {
            break;
        }

        bool expand_f = top_dist_f <= top_dist_r;
        int32_t u;
        double curr_dist;
        if (expand_f) {
            heap_pop(heap_f, &u, &curr_dist);
            NodeState* state_f = hash_get(map_f, u);
            if (!state_f || state_f->dist < curr_dist) continue;

            NodeState* state_r = hash_get(map_r, u);
            if (state_r) {
                double total = curr_dist + state_r->dist;
                if (total < best_cost) {
                    best_cost = total;
                    best_meeting_node = u;
                }
            }

            int64_t row_start = graph->forward_indptr[u];
            int64_t row_end = graph->forward_indptr[u + 1];
            for (int64_t pos = row_start; pos < row_end; pos++) {
                double w = compute_edge_cost(graph, cost_spec, pos);
                if (!isfinite(w)) continue;
                double next_cost = curr_dist + w;
                if (cost_spec->cost_limit > 0.0 && next_cost > cost_spec->cost_limit) continue;
                int32_t v = graph->forward_indices[pos];
                NodeState* state_v = hash_get(map_f, v);
                if (!state_v || next_cost < state_v->dist) {
                    hash_put(map_f, v, u, pos, next_cost);
                    heap_push(heap_f, v, next_cost);
                }
            }
        } else {
            heap_pop(heap_r, &u, &curr_dist);
            NodeState* state_r = hash_get(map_r, u);
            if (!state_r || state_r->dist < curr_dist) continue;

            NodeState* state_f = hash_get(map_f, u);
            if (state_f) {
                double total = curr_dist + state_f->dist;
                if (total < best_cost) {
                    best_cost = total;
                    best_meeting_node = u;
                }
            }

            int64_t row_start = graph->reverse_indptr[u];
            int64_t row_end = graph->reverse_indptr[u + 1];
            for (int64_t rev_pos = row_start; rev_pos < row_end; rev_pos++) {
                int64_t pos = graph->reverse_positions[rev_pos];
                double w = compute_edge_cost(graph, cost_spec, pos);
                if (!isfinite(w)) continue;
                double next_cost = curr_dist + w;
                if (cost_spec->cost_limit > 0.0 && next_cost > cost_spec->cost_limit) continue;
                int32_t v = graph->reverse_indices[rev_pos];
                NodeState* state_v = hash_get(map_r, v);
                if (!state_v || next_cost < state_v->dist) {
                    hash_put(map_r, v, u, pos, next_cost);
                    heap_push(heap_r, v, next_cost);
                }
            }
        }
    }

    int32_t total_positions = -1;
    if (best_meeting_node != -1 && isfinite(best_cost)) {
        *out_total_cost = best_cost;
        int32_t fwd_pos_capacity = 1024;
        int64_t* fwd_pos_buffer = (int64_t*)malloc(
            sizeof(int64_t) * (size_t)fwd_pos_capacity
        );
        int32_t fwd_pos_count = 0;
        int32_t fwd_start_seed_idx = -1;
        bool reconstruction_failed = fwd_pos_buffer == NULL;
        int32_t curr = best_meeting_node;
        int64_t fwd_steps = 0;
        while (!reconstruction_failed && curr != -1) {
            if (++fwd_steps > graph->node_count) {
                reconstruction_failed = true;
                break;
            }
            NodeState* st = hash_get(map_f, curr);
            if (!st) break;
            if (st->parent_rank == -1) {
                for (int32_t s = 0; s < fwd_seed_count; s++) {
                    if (fwd_seed_nodes[s] == curr) {
                        fwd_start_seed_idx = s;
                        break;
                    }
                }
                break;
            }
            if (fwd_pos_count >= fwd_pos_capacity) {
                if (fwd_pos_capacity > INT32_MAX / 2) {
                    reconstruction_failed = true;
                    break;
                }
                int32_t next_capacity = fwd_pos_capacity * 2;
                int64_t* next_buffer = (int64_t*)realloc(
                    fwd_pos_buffer,
                    sizeof(int64_t) * (size_t)next_capacity
                );
                if (!next_buffer) {
                    reconstruction_failed = true;
                    break;
                }
                fwd_pos_buffer = next_buffer;
                fwd_pos_capacity = next_capacity;
            }
            fwd_pos_buffer[fwd_pos_count++] = st->trav_pos;
            curr = st->parent_rank;
        }

        int32_t rev_pos_capacity = 1024;
        int64_t* rev_pos_buffer = (int64_t*)malloc(
            sizeof(int64_t) * (size_t)rev_pos_capacity
        );
        int32_t rev_pos_count = 0;
        int32_t rev_start_seed_idx = -1;
        reconstruction_failed = reconstruction_failed || rev_pos_buffer == NULL;
        curr = best_meeting_node;
        int64_t rev_steps = 0;
        while (!reconstruction_failed && curr != -1) {
            if (++rev_steps > graph->node_count) {
                reconstruction_failed = true;
                break;
            }
            NodeState* st = hash_get(map_r, curr);
            if (!st) break;
            if (st->parent_rank == -1) {
                for (int32_t s = 0; s < rev_seed_count; s++) {
                    if (rev_seed_nodes[s] == curr) {
                        rev_start_seed_idx = s;
                        break;
                    }
                }
                break;
            }
            if (rev_pos_count >= rev_pos_capacity) {
                if (rev_pos_capacity > INT32_MAX / 2) {
                    reconstruction_failed = true;
                    break;
                }
                int32_t next_capacity = rev_pos_capacity * 2;
                int64_t* next_buffer = (int64_t*)realloc(
                    rev_pos_buffer,
                    sizeof(int64_t) * (size_t)next_capacity
                );
                if (!next_buffer) {
                    reconstruction_failed = true;
                    break;
                }
                rev_pos_buffer = next_buffer;
                rev_pos_capacity = next_capacity;
            }
            rev_pos_buffer[rev_pos_count++] = st->trav_pos;
            curr = st->parent_rank;
        }

        if (!reconstruction_failed) {
            int64_t total = (int64_t)fwd_pos_count + (int64_t)rev_pos_count;
            if (total > INT32_MAX) {
                reconstruction_failed = true;
            } else {
                total_positions = (int32_t)total;
                if (total > 0) {
                    int64_t* result_buffer = (int64_t*)malloc(
                        sizeof(int64_t) * (size_t)total
                    );
                    if (result_buffer == NULL) {
                        reconstruction_failed = true;
                    } else {
                        int32_t idx = 0;
                        for (int32_t i = fwd_pos_count - 1; i >= 0; i--) {
                            result_buffer[idx++] = fwd_pos_buffer[i];
                        }
                        for (int32_t i = 0; i < rev_pos_count; i++) {
                            result_buffer[idx++] = rev_pos_buffer[i];
                        }
                        *out_trav_positions = result_buffer;
                    }
                }
            }
        }
        free(fwd_pos_buffer);
        free(rev_pos_buffer);
        if (!reconstruction_failed) {
            *out_fwd_seed_index = fwd_start_seed_idx;
            *out_rev_seed_index = rev_start_seed_idx;
        } else {
            total_positions = -1;
        }
    }

    heap_free(heap_f);
    heap_free(heap_r);
    hash_free(map_f);
    hash_free(map_r);
    return total_positions;
}

void compact_free_positions(int64_t* positions) {
    free(positions);
}
