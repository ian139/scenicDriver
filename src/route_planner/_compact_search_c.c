#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

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

static bool checked_size_mul(size_t count, size_t element_size, size_t* out_bytes) {
    if (element_size != 0 && count > SIZE_MAX / element_size) return false;
    *out_bytes = count * element_size;
    return true;
}

static MinHeap* heap_create(int32_t capacity) {
    MinHeap* h = (MinHeap*)malloc(sizeof(MinHeap));
    if (!h) return NULL;
    h->capacity = capacity > 16 ? capacity : 16;
    h->size = 0;
    size_t bytes = 0;
    if (!checked_size_mul((size_t)h->capacity, sizeof(HeapItem), &bytes)) {
        free(h);
        return NULL;
    }
    h->data = (HeapItem*)malloc(bytes);
    if (!h->data) {
        free(h);
        return NULL;
    }
    return h;
}

static void heap_free(MinHeap* h) {
    if (h) {
        free(h->data);
        free(h);
    }
}

static bool heap_push(MinHeap* h, int32_t node_rank, double dist) {
    if (!h || h->size < 0 || h->capacity <= 0) return false;
    if (h->size >= h->capacity) {
        if (h->capacity > INT32_MAX / 2) return false;
        int32_t next_capacity = h->capacity * 2;
        size_t bytes = 0;
        if (!checked_size_mul((size_t)next_capacity, sizeof(HeapItem), &bytes)) {
            return false;
        }
        HeapItem* next_data = (HeapItem*)realloc(h->data, bytes);
        if (!next_data) return false;
        h->data = next_data;
        h->capacity = next_capacity;
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
    return true;
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

/* Per-direction search state, indexed directly by node rank in
 * [0, node_count).  Four parallel arrays hold the settled/touched records
 * that the previous open-addressed HashTable stored: an explicit seen
 * stamp, the best distance, the reconstruction parent rank, and the
 * traversal position used to rebuild the winning path.
 *
 * The stamp array is calloc'd: its zero-filled sentinel is the explicit
 * "never touched this search" state.  Large calloc regions are backed by
 * OS zero pages faulted in lazily, so no eager O(node_count) memset runs
 * and physical pages are committed only for ranks the search reaches.
 * The value arrays (dist/parent/trav_pos) are plain malloc'd: every read
 * of them is gated by a matching seen stamp, which is only ever written
 * together with the value, so stale bytes are never observed.
 *
 * The sentinel is a per-call constant: 0 (from calloc) never equals it,
 * and written slots always do.  No global generation counter is needed,
 * so the state is thread-safe and there is nothing to reset.
 */
typedef struct {
    int64_t node_count;
    uint32_t* stamp;
    double* dist;
    int32_t* parent;
    int64_t* trav_pos;
} DirectionState;

/* Any non-zero sentinel works (see above); 1 is used for clarity. */
static const uint32_t k_search_gen = 1u;

static bool direction_state_alloc(DirectionState* ds, int64_t node_count) {
    memset(ds, 0, sizeof(*ds));
    ds->node_count = node_count;
    size_t n = (size_t)node_count;
    size_t bytes = 0;
    if (!checked_size_mul(n, sizeof(uint32_t), &bytes)) return false;
    /* calloc: zero-filled seen sentinel, backed by lazily faulted zero
     * pages for large counts (see the struct comment). */
    ds->stamp = (uint32_t*)calloc(n, sizeof(uint32_t));
    if (!ds->stamp) return false;
    if (!checked_size_mul(n, sizeof(double), &bytes)) {
        free(ds->stamp);
        ds->stamp = NULL;
        return false;
    }
    ds->dist = (double*)malloc(bytes);
    if (!ds->dist) {
        free(ds->stamp);
        ds->stamp = NULL;
        return false;
    }
    if (!checked_size_mul(n, sizeof(int32_t), &bytes)) {
        free(ds->stamp);
        free(ds->dist);
        ds->stamp = NULL;
        ds->dist = NULL;
        return false;
    }
    ds->parent = (int32_t*)malloc(bytes);
    if (!ds->parent) {
        free(ds->stamp);
        free(ds->dist);
        ds->stamp = NULL;
        ds->dist = NULL;
        return false;
    }
    if (!checked_size_mul(n, sizeof(int64_t), &bytes)) {
        free(ds->stamp);
        free(ds->dist);
        free(ds->parent);
        ds->stamp = NULL;
        ds->dist = NULL;
        ds->parent = NULL;
        return false;
    }
    ds->trav_pos = (int64_t*)malloc(bytes);
    if (!ds->trav_pos) {
        free(ds->stamp);
        free(ds->dist);
        free(ds->parent);
        ds->stamp = NULL;
        ds->dist = NULL;
        ds->parent = NULL;
        return false;
    }
    return true;
}

static void direction_state_free(DirectionState* ds) {
    if (!ds) return;
    free(ds->stamp);
    free(ds->dist);
    free(ds->parent);
    free(ds->trav_pos);
    memset(ds, 0, sizeof(*ds));
}

/* Explicit seen-state check with a bounds guard; the only gate before any
 * dist/parent/trav_pos read. */
static inline bool state_seen(const DirectionState* ds, int32_t rank, uint32_t gen) {
    return rank >= 0 && (int64_t)rank < ds->node_count && ds->stamp[rank] == gen;
}

/* Callers must have validated rank in [0, node_count) (state_seen above or
 * the seed/neighbor guards in the search). */
static inline void state_set(DirectionState* ds, int32_t rank, int32_t parent_rank, int64_t trav_pos, double dist, uint32_t gen) {
    ds->stamp[rank] = gen;
    ds->dist[rank] = dist;
    ds->parent[rank] = parent_rank;
    ds->trav_pos[rank] = trav_pos;
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

/* Bounded deadline enforcement.  ``deadline_seconds < 0.0`` means unlimited;
 * otherwise it is the remaining wall-clock budget for the whole native call,
 * measured on the same monotonic time base as the Python RoutingDeadline
 * (time.monotonic / CLOCK_MONOTONIC), so a search that would run past the
 * request deadline fails closed instead of completing.  The check is
 * amortized over batches of 1024 expansions so no per-node overhead is
 * added; overshoot is bounded by one batch plus a few clock reads.  A
 * cancel_event cannot interrupt the native call mid-search; the Python
 * boundary check surfaces cancellation before and after the call.
 */
static inline bool deadline_exceeded(double deadline_seconds, struct timespec started) {
    if (deadline_seconds < 0.0) return false;
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return true; /* fail closed */
    double elapsed = (double)(now.tv_sec - started.tv_sec)
        + (double)(now.tv_nsec - started.tv_nsec) / 1000000000.0;
    return elapsed >= deadline_seconds;
}

/* Negative returns: -1 generic failure, -2 deadline exceeded. */
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
    int32_t* out_rev_seed_index,
    double deadline_seconds
) {
    *out_trav_positions = NULL;
    int32_t total_positions = -1;
    if (!graph || !cost_spec || !out_trav_positions || !out_total_cost ||
        !out_fwd_seed_index || !out_rev_seed_index) {
        return -1;
    }
    if (graph->node_count <= 0 || graph->node_count > (int64_t)INT32_MAX) {
        return -1;
    }
    if (fwd_seed_count < 0 || rev_seed_count < 0) return -1;
    if (fwd_seed_count > 0 && (!fwd_seed_nodes || !fwd_seed_costs)) return -1;
    if (rev_seed_count > 0 && (!rev_seed_nodes || !rev_seed_costs)) return -1;
    const int64_t node_count = graph->node_count;

    struct timespec started_clock;
    if (clock_gettime(CLOCK_MONOTONIC, &started_clock) != 0) {
        /* Clock unavailable: cannot bound the search, fail closed only when a
         * deadline was actually requested (the loop check below retries). */
        started_clock.tv_sec = 0;
        started_clock.tv_nsec = 0;
    }
    MinHeap* heap_f = heap_create(128);
    MinHeap* heap_r = heap_create(128);
    DirectionState ds_f;
    DirectionState ds_r;
    bool ds_f_ok = false;
    bool ds_r_ok = false;
    if (heap_f && heap_r) {
        ds_f_ok = direction_state_alloc(&ds_f, node_count);
    }
    if (heap_f && heap_r && ds_f_ok) {
        ds_r_ok = direction_state_alloc(&ds_r, node_count);
    }
    if (!heap_f || !heap_r || !ds_f_ok || !ds_r_ok) {
        heap_free(heap_f);
        heap_free(heap_r);
        if (ds_f_ok) direction_state_free(&ds_f);
        if (ds_r_ok) direction_state_free(&ds_r);
        return -1;
    }

    for (int32_t i = 0; i < fwd_seed_count; i++) {
        int32_t node = fwd_seed_nodes[i];
        double cost = fwd_seed_costs[i];
        if (cost_spec->cost_limit > 0.0 && cost > cost_spec->cost_limit) continue;
        if (node < 0 || (int64_t)node >= node_count) continue; /* reject invalid seed ranks */
        state_set(&ds_f, node, -1, -1, cost, k_search_gen);
        if (!heap_push(heap_f, node, cost)) goto cleanup;
    }
    for (int32_t i = 0; i < rev_seed_count; i++) {
        int32_t node = rev_seed_nodes[i];
        double cost = rev_seed_costs[i];
        if (cost_spec->cost_limit > 0.0 && cost > cost_spec->cost_limit) continue;
        if (node < 0 || (int64_t)node >= node_count) continue; /* reject invalid seed ranks */
        state_set(&ds_r, node, -1, -1, cost, k_search_gen);
        if (!heap_push(heap_r, node, cost)) goto cleanup;
    }

    double best_cost = INFINITY;
    int32_t best_meeting_node = -1;
    bool timed_out = false;
    int64_t outer_steps = 0;

    while (heap_f->size > 0 && heap_r->size > 0) {
        if ((outer_steps++ & 1023) == 0) {
            if (deadline_exceeded(deadline_seconds, started_clock)) {
                timed_out = true;
                break;
            }
        }
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
            if (!state_seen(&ds_f, u, k_search_gen) || ds_f.dist[u] < curr_dist) continue;

            if (state_seen(&ds_r, u, k_search_gen)) {
                double total = curr_dist + ds_r.dist[u];
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
                if (v < 0 || (int64_t)v >= node_count) continue; /* reject invalid neighbor ranks */
                if (!state_seen(&ds_f, v, k_search_gen) || next_cost < ds_f.dist[v]) {
                    state_set(&ds_f, v, u, pos, next_cost, k_search_gen);
                    if (!heap_push(heap_f, v, next_cost)) goto cleanup;
                }
            }
        } else {
            heap_pop(heap_r, &u, &curr_dist);
            if (!state_seen(&ds_r, u, k_search_gen) || ds_r.dist[u] < curr_dist) continue;

            if (state_seen(&ds_f, u, k_search_gen)) {
                double total = curr_dist + ds_f.dist[u];
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
                if (v < 0 || (int64_t)v >= node_count) continue; /* reject invalid neighbor ranks */
                if (!state_seen(&ds_r, v, k_search_gen) || next_cost < ds_r.dist[v]) {
                    state_set(&ds_r, v, u, pos, next_cost, k_search_gen);
                    if (!heap_push(heap_r, v, next_cost)) goto cleanup;
                }
            }
        }
    }

    if (timed_out) {
        total_positions = -2;
    } else if (best_meeting_node != -1 && isfinite(best_cost)) {
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
            if (!state_seen(&ds_f, curr, k_search_gen)) break;
            if (ds_f.parent[curr] == -1) {
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
            fwd_pos_buffer[fwd_pos_count++] = ds_f.trav_pos[curr];
            curr = ds_f.parent[curr];
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
            if (!state_seen(&ds_r, curr, k_search_gen)) break;
            if (ds_r.parent[curr] == -1) {
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
            rev_pos_buffer[rev_pos_count++] = ds_r.trav_pos[curr];
            curr = ds_r.parent[curr];
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

cleanup:
    heap_free(heap_f);
    heap_free(heap_r);
    direction_state_free(&ds_f);
    direction_state_free(&ds_r);
    return total_positions;
}

void compact_free_positions(int64_t* positions) {
    free(positions);
}
