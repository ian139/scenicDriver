#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <float.h>

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
    /* Canonical edge id string table (rank -> byte range), used to compare
     * path keys exactly like Python's lexicographic edge-id strings.  Both
     * pointers may be NULL (legacy/tampered spec); comparison then falls
     * back to numeric edge ranks. */
    const uint8_t* edge_id_strings;
    const int64_t* edge_id_offsets;
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
    double cost_limit; /* -1 means absent; zero is an active bound. */
} CostSpec;

typedef struct {
    double dist;
    uint64_t sequence;
    int32_t node_rank;
    int32_t rank_primary;
    int32_t rank_secondary;
    int32_t label_id;
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

static int heap_item_compare(const HeapItem* a, const HeapItem* b) {
    int order = (a->dist > b->dist) - (a->dist < b->dist);
    if (order != 0) return order;
    order = (a->rank_primary > b->rank_primary)
        - (a->rank_primary < b->rank_primary);
    if (order != 0) return order;
    order = (a->rank_secondary > b->rank_secondary)
        - (a->rank_secondary < b->rank_secondary);
    if (order != 0) return order;
    return (a->sequence > b->sequence) - (a->sequence < b->sequence);
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

static bool heap_push(
    MinHeap* h,
    int32_t node_rank,
    double dist,
    int32_t rank_primary,
    int32_t rank_secondary,
    uint64_t sequence,
    int32_t label_id
) {
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
    HeapItem item = {
        .dist = dist,
        .sequence = sequence,
        .node_rank = node_rank,
        .rank_primary = rank_primary,
        .rank_secondary = rank_secondary,
        .label_id = label_id,
    };
    int32_t i = h->size++;
    while (i > 0) {
        int32_t p = (i - 1) / 2;
        if (heap_item_compare(&h->data[p], &item) <= 0) break;
        h->data[i] = h->data[p];
        i = p;
    }
    h->data[i] = item;
    return true;
}

static bool heap_pop(MinHeap* h, HeapItem* out_item) {
    if (!h || !out_item || h->size <= 0) return false;
    *out_item = h->data[0];
    HeapItem last = h->data[--h->size];
    if (h->size > 0) {
        int32_t i = 0;
        while (i * 2 + 1 < h->size) {
            int32_t left = i * 2 + 1;
            int32_t right = left + 1;
            int32_t smallest = left;
            if (
                right < h->size
                && heap_item_compare(&h->data[right], &h->data[left]) < 0
            ) {
                smallest = right;
            }
            if (heap_item_compare(&last, &h->data[smallest]) <= 0) break;
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
 *
 * Three more per-rank arrays carry the ranked-search metadata that the
 * Python `_bidirectional_search_core` keeps on each label:
 *   - label_id: stable insertion id of the winning label at the rank.
 *   - seed_idx: compact index of the seed this label descends from
 *     (-1 until the seed itself is recorded).
 *   - rank_primary/rank_secondary: the seed's (start, direction) rank
 *     pair, propagated verbatim by every relaxation (Python passes the
 *     seed's `rank` tuple through the whole label chain).
 * The seed index is written exactly once, when the seed is installed;
 * later equal-cost replacements carry their own seed identity.  All of
 * these are gated by the same seen stamp as the value arrays.
 *
 * Persistent path records (PathRecord below) back the Python
 * `_PersistentPathKey` tie-break: each label points at its parent record
 * and stores the numeric `trav_edge_rank` of the edge that produced it,
 * so equal-cost comparisons compare canonical edge-rank sequences instead
 * of CSR insertion order.  Records are chained in a per-call arena that
 * lives for the whole search and is freed at cleanup.
 */
typedef struct {
    int64_t node_count;
    uint32_t* stamp;
    double* dist;
    int32_t* parent;
    int64_t* trav_pos;
    int32_t* label_id;
    int32_t* seed_idx;
    int32_t* rank_primary;
    int32_t* rank_secondary;
    int32_t* path_id;
} DirectionState;

/* One linked path record: the edge-rank of the step that produced the
 * label plus the record id of its parent label's path.  -1 parent means
 * "seed" (empty path).  The edge ranks are the canonical `trav_edge_rank`
 * values, so sequences compare exactly like Python's lexicographic
 * `_PersistentPathKey` over canonical edge ids (rank order is the
 * canonical edge insertion order, and every traversal of the same edge
 * carries the same rank).  Forward expansions append at the tail; reverse
 * expansions prepend at the head; both use `parent` as the earlier part,
 * matching `prepend` semantics in the Python key.
 */
typedef struct {
    int32_t parent;
    int32_t edge_rank;
} PathRecord;

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
    if (!checked_size_mul(n, sizeof(int32_t), &bytes)) {
        free(ds->stamp);
        free(ds->dist);
        free(ds->parent);
        free(ds->trav_pos);
        ds->stamp = NULL;
        ds->dist = NULL;
        ds->parent = NULL;
        ds->trav_pos = NULL;
        return false;
    }
    ds->label_id = (int32_t*)malloc(bytes);
    if (!ds->label_id) {
        free(ds->stamp);
        free(ds->dist);
        free(ds->parent);
        free(ds->trav_pos);
        ds->stamp = NULL;
        ds->dist = NULL;
        ds->parent = NULL;
        ds->trav_pos = NULL;
        return false;
    }
    ds->seed_idx = (int32_t*)malloc(bytes);
    if (!ds->seed_idx) {
        free(ds->stamp);
        free(ds->dist);
        free(ds->parent);
        free(ds->trav_pos);
        free(ds->label_id);
        ds->stamp = NULL;
        ds->dist = NULL;
        ds->parent = NULL;
        ds->trav_pos = NULL;
        ds->label_id = NULL;
        return false;
    }
    ds->rank_primary = (int32_t*)malloc(bytes);
    if (!ds->rank_primary) {
        free(ds->stamp);
        free(ds->dist);
        free(ds->parent);
        free(ds->trav_pos);
        free(ds->label_id);
        free(ds->seed_idx);
        ds->stamp = NULL;
        ds->dist = NULL;
        ds->parent = NULL;
        ds->trav_pos = NULL;
        ds->label_id = NULL;
        ds->seed_idx = NULL;
        return false;
    }
    ds->rank_secondary = (int32_t*)malloc(bytes);
    if (!ds->rank_secondary) {
        free(ds->stamp);
        free(ds->dist);
        free(ds->parent);
        free(ds->trav_pos);
        free(ds->label_id);
        free(ds->seed_idx);
        free(ds->rank_primary);
        ds->stamp = NULL;
        ds->dist = NULL;
        ds->parent = NULL;
        ds->trav_pos = NULL;
        ds->label_id = NULL;
        ds->seed_idx = NULL;
        ds->rank_primary = NULL;
        return false;
    }
    ds->path_id = (int32_t*)malloc(bytes);
    if (!ds->path_id) {
        free(ds->stamp);
        free(ds->dist);
        free(ds->parent);
        free(ds->trav_pos);
        free(ds->label_id);
        free(ds->seed_idx);
        free(ds->rank_primary);
        free(ds->rank_secondary);
        ds->stamp = NULL;
        ds->dist = NULL;
        ds->parent = NULL;
        ds->trav_pos = NULL;
        ds->label_id = NULL;
        ds->seed_idx = NULL;
        ds->rank_primary = NULL;
        ds->rank_secondary = NULL;
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
    free(ds->label_id);
    free(ds->seed_idx);
    free(ds->rank_primary);
    free(ds->rank_secondary);
    free(ds->path_id);
    memset(ds, 0, sizeof(*ds));
}

/* Explicit seen-state check with a bounds guard; the only gate before any
 * dist/parent/trav_pos read. */
static inline bool state_seen(const DirectionState* ds, int32_t rank, uint32_t gen) {
    return rank >= 0 && (int64_t)rank < ds->node_count && ds->stamp[rank] == gen;
}

/* Callers must have validated rank in [0, node_count) (state_seen above or
 * the seed/neighbor guards in the search). */
static inline void state_set(
    DirectionState* ds,
    int32_t rank,
    int32_t parent_rank,
    int64_t trav_pos,
    double dist,
    int32_t label_id,
    int32_t seed_idx,
    int32_t rank_primary,
    int32_t rank_secondary,
    int32_t path_id,
    uint32_t gen
) {
    ds->stamp[rank] = gen;
    ds->dist[rank] = dist;
    ds->parent[rank] = parent_rank;
    ds->trav_pos[rank] = trav_pos;
    ds->label_id[rank] = label_id;
    ds->seed_idx[rank] = seed_idx;
    ds->rank_primary[rank] = rank_primary;
    ds->rank_secondary[rank] = rank_secondary;
    ds->path_id[rank] = path_id;
}

/* One absolute deadline for the whole native call.  ``budget < 0.0`` means
 * unlimited; otherwise it is the remaining wall-clock budget, measured from
 * the first statement of the call on the same monotonic time base as the
 * Python RoutingDeadline (time.monotonic / CLOCK_MONOTONIC).  Every phase --
 * sidecar validation, allocation, seeding, expansion, path-key comparison
 * and path reconstruction -- charges work units against it, and the clock
 * is read once per k_deadline_check_interval units, so no per-node clock
 * overhead is added.  Overshoot is bounded by one interval plus one
 * indivisible step (a single allocation or path-chain walk).  A final
 * unconditional check runs before success is committed, and a clock failure
 * fails closed.  A cancel_event cannot interrupt the native call; the Python
 * boundary check surfaces cancellation before and after the call. */
typedef struct {
    double budget;
    struct timespec started;
    int64_t work;
    int64_t next_check;
    bool expired;
} SearchDeadline;

static const int64_t k_deadline_check_interval = 1024;

static bool deadline_start(SearchDeadline* d, double budget) {
    d->budget = budget;
    d->started.tv_sec = 0;
    d->started.tv_nsec = 0;
    d->work = 0;
    d->next_check = 0;
    d->expired = false;
    if (budget < 0.0) return true;
    if (clock_gettime(CLOCK_MONOTONIC, &d->started) != 0) {
        d->expired = true;
        return false;
    }
    return true;
}

static bool deadline_check_now(SearchDeadline* d) {
    if (d->budget < 0.0 || d->expired) return d->expired;
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        d->expired = true; /* fail closed */
        return true;
    }
    double elapsed = (double)(now.tv_sec - d->started.tv_sec)
        + (double)(now.tv_nsec - d->started.tv_nsec) / 1000000000.0;
    if (elapsed >= d->budget) d->expired = true;
    return d->expired;
}

/* Charge ``units`` (>= 0) of completed work; returns true once expired. */
static inline bool deadline_charge(SearchDeadline* d, int64_t units) {
    if (d->budget < 0.0) return false;
    if (d->expired) return true;
    d->work += units;
    if (d->work < d->next_check) return false;
    d->next_check = d->work + k_deadline_check_interval;
    return deadline_check_now(d);
}

/* Validate the canonical-edge-rank score sidecar under the call deadline.
 * Returns 0 when valid, -1 for a malformed rank table, -2 when the deadline
 * expires, and -4 when any score is non-finite (rejected rather than
 * silently coerced, so a corrupt report sidecar can never be misread). */
static int32_t validate_edge_score_sidecar(
    const CompactGraphSpec* graph,
    const double* edge_scenic_score_by_rank,
    SearchDeadline* deadline
) {
    if (!graph || !edge_scenic_score_by_rank) return -1;
    if (graph->edge_count < 0 || graph->traversal_count < 0) return -1;
    if (graph->traversal_count > 0 && !graph->trav_edge_rank) return -1;
    for (int64_t pos = 0; pos < graph->traversal_count; pos++) {
        if (deadline_charge(deadline, 1)) return -2;
        int32_t edge_rank = graph->trav_edge_rank[pos];
        if (edge_rank < 0 || (int64_t)edge_rank >= graph->edge_count) {
            return -1;
        }
    }
    for (int64_t rank = 0; rank < graph->edge_count; rank++) {
        if (deadline_charge(deadline, 1)) return -2;
        if (!isfinite(edge_scenic_score_by_rank[rank])) return -4;
    }
    return 0;
}

static inline double compute_edge_cost(
    const CompactGraphSpec* g,
    const CostSpec* c,
    int64_t pos,
    const double* edge_scenic_score_by_rank,
    bool* invalid_edge_rank,
    bool* cost_overflow
) {
    *invalid_edge_rank = false;
    int32_t edge_rank = -1;
    if (edge_scenic_score_by_rank) {
        if (!g->trav_edge_rank) {
            *invalid_edge_rank = true;
            return INFINITY;
        }
        edge_rank = g->trav_edge_rank[pos];
        if (edge_rank < 0 || (int64_t)edge_rank >= g->edge_count) {
            *invalid_edge_rank = true;
            return INFINITY;
        }
    }
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
        if (!isfinite(cost)) *cost_overflow = true;
        return cost;
    }

    /* Do not turn an inactive overflowing component into 0 * infinity (NaN). */
    double weighted_base = 0.0;
    if (c->scenic_weight < 1.0 && c->travel_weight > 0.0) {
        weighted_base = (1.0 - c->scenic_weight) * (duration * c->travel_weight);
    }
    if (c->scenic_weight > 0.0 && c->scenic_reward > 0.0) {
        double raw_score = edge_scenic_score_by_rank
            ? edge_scenic_score_by_rank[edge_rank]
            : (g->trav_scenic_score ? g->trav_scenic_score[pos] : 5.0);
        if (!isfinite(raw_score)) raw_score = 0.0;
        if (raw_score < 0.0) raw_score = 0.0;
        if (raw_score > 10.0) raw_score = 10.0;
        weighted_base += c->scenic_weight
            * (duration * ((10.0 - raw_score) / 10.0) * c->scenic_reward);
    }
    double base = isfinite(weighted_base)
        ? (weighted_base > 1e-6 * duration ? weighted_base : 1e-6 * duration)
        : weighted_base;

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
    if (!isfinite(cost)) cost = DBL_MAX;

    if (c->lagrangian_multiplier > 0.0) {
        cost += c->lagrangian_multiplier * duration;
    }
    if (!isfinite(cost)) *cost_overflow = true;
    return cost;
}

/* ---- ranked-search ordering helpers ------------------------------------ */

/* Lexicographic compare of two canonical edge ids (Python string order),
 * resolved through the compact edge id string table.  Falls back to numeric
 * edge-rank comparison when the table is unavailable.  Python compares
 * ``str`` by code points and the ids are UTF-8, so byte order == string
 * order. */
static int edge_key_compare(
    const CompactGraphSpec* graph,
    int32_t left_rank,
    int32_t right_rank
) {
    if (graph->edge_id_strings && graph->edge_id_offsets) {
        int64_t left_start = graph->edge_id_offsets[left_rank];
        int64_t left_end = graph->edge_id_offsets[left_rank + 1];
        int64_t right_start = graph->edge_id_offsets[right_rank];
        int64_t right_end = graph->edge_id_offsets[right_rank + 1];
        int64_t left_len = left_end - left_start;
        int64_t right_len = right_end - right_start;
        int64_t common = left_len < right_len ? left_len : right_len;
        if (common > 0) {
            int cmp = memcmp(
                graph->edge_id_strings + left_start,
                graph->edge_id_strings + right_start,
                (size_t)common
            );
            if (cmp != 0) return cmp < 0 ? -1 : 1;
        }
        if (left_len == right_len) return 0;
        return left_len < right_len ? -1 : 1;
    }
    if (left_rank != right_rank) return left_rank < right_rank ? -1 : 1;
    return 0;
}

/* Count a path chain iteratively, rejecting malformed parent links before
 * any record is indexed.  ``record_count`` is the current arena size, not a
 * graph-node bound: rejected labels may leave orphaned records in the arena. */
static bool path_record_length(
    const PathRecord* records,
    int32_t record_count,
    int32_t id,
    int32_t* out_length
) {
    if (!out_length || record_count < 0 || id < -1) return false;
    int32_t length = 0;
    int32_t current = id;
    while (current != -1) {
        if (!records || current < 0 || current >= record_count) return false;
        if (length == INT32_MAX) return false;
        ++length;
        current = records[current].parent;
    }
    *out_length = length;
    return true;
}

/* Materialize one parent-linked chain in chronological order without using
 * recursion.  The array is deliberately transient: path records remain the
 * only persistent per-label storage, and callers free this scratch before
 * every return (including allocation and malformed-chain failures). */
static bool path_record_materialize_forward(
    const PathRecord* records,
    int32_t record_count,
    int32_t id,
    int32_t length,
    int32_t** out_edges
) {
    if (!out_edges || length < 0) return false;
    *out_edges = NULL;
    if (length == 0) return true;
    size_t bytes = 0;
    if (!checked_size_mul((size_t)length, sizeof(int32_t), &bytes)) {
        return false;
    }
    int32_t* edges = (int32_t*)malloc(bytes);
    if (!edges) return false;
    int32_t current = id;
    for (int32_t i = length; i > 0; --i) {
        if (!records || current < 0 || current >= record_count) {
            free(edges);
            return false;
        }
        edges[i - 1] = records[current].edge_rank;
        current = records[current].parent;
    }
    if (current != -1) {
        free(edges);
        return false;
    }
    *out_edges = edges;
    return true;
}

/* Compare two persistent path chains.  ``newest_first`` selects the Python
 * orientation: forward labels store chronological edge order (oldest record
 * first), reverse labels store the reversed order (newest record first, i.e.
 * the meeting-side edge first).  The forward path is materialized in
 * transient heap scratch so this comparison is iterative and has bounded C
 * stack use even for very deep equal-cost prefixes. */
static int path_record_compare(
    const CompactGraphSpec* graph,
    int32_t left_id,
    int32_t right_id,
    const PathRecord* records,
    int32_t record_count,
    bool newest_first,
    SearchDeadline* deadline,
    bool* compare_ok
) {
    if (!compare_ok) return 0;
    *compare_ok = true;
    int32_t left_length = 0;
    int32_t right_length = 0;
    if (!path_record_length(
            records, record_count, left_id, &left_length
        ) ||
        !path_record_length(
            records, record_count, right_id, &right_length
        )) {
        *compare_ok = false;
        return 0;
    }
    /* The chain walks above are the dominant O(path) work; charge them so a
     * deep equal-cost comparison cannot outrun the deadline unchecked. */
    if (deadline_charge(
            deadline, (int64_t)left_length + (int64_t)right_length
        )) {
        *compare_ok = false;
        return 0;
    }

    if (newest_first) {
        int32_t l = left_id;
        int32_t r = right_id;
        while (l != -1 && r != -1) {
            int cmp = edge_key_compare(
                graph, records[l].edge_rank, records[r].edge_rank
            );
            if (cmp != 0) return cmp;
            l = records[l].parent;
            r = records[r].parent;
        }
        if (l == r) return 0;
        return l == -1 ? -1 : 1; /* shorter prefix is smaller */
    }

    int32_t* left_edges = NULL;
    int32_t* right_edges = NULL;
    if (!path_record_materialize_forward(
            records, record_count, left_id, left_length, &left_edges
        ) ||
        !path_record_materialize_forward(
            records, record_count, right_id, right_length, &right_edges
        )) {
        free(left_edges);
        free(right_edges);
        *compare_ok = false;
        return 0;
    }

    int32_t common = left_length < right_length ? left_length : right_length;
    int result = 0;
    for (int32_t i = 0; i < common; ++i) {
        result = edge_key_compare(graph, left_edges[i], right_edges[i]);
        if (result != 0) break;
    }
    free(left_edges);
    free(right_edges);
    if (result != 0) return result;
    if (left_length == right_length) return 0;
    return left_length < right_length ? -1 : 1; /* shorter prefix is smaller */
}

/* Compare the Python ``middle_key`` for a meeting candidate: the forward
 * path (chronological, oldest first) concatenated with the reverse path
 * (reversed, newest first).  Every forward edge is compared before moving
 * to the reverse side; this matters when candidates meet at different nodes
 * and one candidate's forward path ends while the other's continues. */
static int middle_key_compare(
    const CompactGraphSpec* graph,
    int32_t fwd_a,
    int32_t rev_a,
    int32_t fwd_b,
    int32_t rev_b,
    const PathRecord* records,
    int32_t record_count,
    SearchDeadline* deadline,
    bool* compare_ok
) {
    if (!compare_ok) return 0;
    *compare_ok = true;
    int32_t fwd_length_a = 0;
    int32_t fwd_length_b = 0;
    int32_t rev_length_a = 0;
    int32_t rev_length_b = 0;
    if (!path_record_length(
            records, record_count, fwd_a, &fwd_length_a
        ) ||
        !path_record_length(
            records, record_count, fwd_b, &fwd_length_b
        ) ||
        !path_record_length(
            records, record_count, rev_a, &rev_length_a
        ) ||
        !path_record_length(
            records, record_count, rev_b, &rev_length_b
        )) {
        *compare_ok = false;
        return 0;
    }
    if (deadline_charge(
            deadline,
            (int64_t)fwd_length_a + (int64_t)fwd_length_b +
                (int64_t)rev_length_a + (int64_t)rev_length_b
        )) {
        *compare_ok = false;
        return 0;
    }

    int32_t* fwd_edges_a = NULL;
    int32_t* fwd_edges_b = NULL;
    if (!path_record_materialize_forward(
            records, record_count, fwd_a, fwd_length_a, &fwd_edges_a
        ) ||
        !path_record_materialize_forward(
            records, record_count, fwd_b, fwd_length_b, &fwd_edges_b
        )) {
        free(fwd_edges_a);
        free(fwd_edges_b);
        *compare_ok = false;
        return 0;
    }

    int32_t fwd_index_a = 0;
    int32_t fwd_index_b = 0;
    int32_t rev_index_a = 0;
    int32_t rev_index_b = 0;
    int32_t rev_current_a = rev_a;
    int32_t rev_current_b = rev_b;
    int result = 0;
    while (
        fwd_index_a < fwd_length_a || rev_index_a < rev_length_a ||
        fwd_index_b < fwd_length_b || rev_index_b < rev_length_b
    ) {
        bool a_has = fwd_index_a < fwd_length_a || rev_index_a < rev_length_a;
        bool b_has = fwd_index_b < fwd_length_b || rev_index_b < rev_length_b;
        if (!a_has) {
            result = b_has ? -1 : 0;
            break;
        }
        if (!b_has) {
            result = 1;
            break;
        }

        int32_t edge_a;
        int32_t edge_b;
        if (fwd_index_a < fwd_length_a) {
            edge_a = fwd_edges_a[fwd_index_a++];
        } else {
            if (
                rev_current_a < 0 ||
                rev_current_a >= record_count ||
                rev_index_a >= rev_length_a
            ) {
                result = 0;
                *compare_ok = false;
                break;
            }
            edge_a = records[rev_current_a].edge_rank;
            rev_current_a = records[rev_current_a].parent;
            ++rev_index_a;
        }
        if (fwd_index_b < fwd_length_b) {
            edge_b = fwd_edges_b[fwd_index_b++];
        } else {
            if (
                rev_current_b < 0 ||
                rev_current_b >= record_count ||
                rev_index_b >= rev_length_b
            ) {
                result = 0;
                *compare_ok = false;
                break;
            }
            edge_b = records[rev_current_b].edge_rank;
            rev_current_b = records[rev_current_b].parent;
            ++rev_index_b;
        }
        result = edge_key_compare(graph, edge_a, edge_b);
        if (result != 0) break;
    }
    free(fwd_edges_a);
    free(fwd_edges_b);
    return result;
}

/* Does a new label at ``rank`` beat the current best label there?  Mirrors
 * the Python add_label rejection conditions: strictly cheaper, or equal cost
 * with a strictly smaller (rank pair, path key).  The first label with a
 * given key wins equal-key ties, matching Python's ``<`` comparison.
 *
 * ``compare_ok`` is set false when the path-key comparison cannot be
 * completed (transient scratch allocation failed, a malformed chain was
 * detected, or ``deadline`` expired); callers must then abort the search
 * instead of trusting the boolean result. */
static bool label_candidate_better(
    const CompactGraphSpec* graph,
    const DirectionState* ds,
    int32_t rank,
    double cost,
    int32_t rank_primary,
    int32_t rank_secondary,
    int32_t new_path_id,
    const PathRecord* records,
    int32_t record_count,
    bool newest_first,
    SearchDeadline* deadline,
    bool* compare_ok
) {
    if (!compare_ok) return false;
    *compare_ok = true;
    if (!state_seen(ds, rank, k_search_gen)) return true;
    if (cost < ds->dist[rank]) return true;
    if (cost > ds->dist[rank]) return false;
    if (rank_primary != ds->rank_primary[rank]) {
        return rank_primary < ds->rank_primary[rank];
    }
    if (rank_secondary != ds->rank_secondary[rank]) {
        return rank_secondary < ds->rank_secondary[rank];
    }
    int cmp = path_record_compare(
        graph, new_path_id, ds->path_id[rank], records, record_count,
        newest_first, deadline, compare_ok
    );
    if (!*compare_ok) return false;
    return cmp < 0;
}

/* Consider the meeting candidate at ``node``: total = dist_f + dist_r, and
 * the Python rank_key ``(f_primary, r_primary, f_secondary, r_secondary,
 * middle_key)``.  Strictly better (cost, rank_key) replaces the incumbent;
 * the first candidate wins full ties.  Returns false when the middle-key
 * comparison cannot be completed (transient scratch allocation failure,
 * malformed chain, or expired ``deadline``) or the total overflows; callers
 * must then abort the search. */
static bool consider_meeting(
    const CompactGraphSpec* graph,
    const CostSpec* cost_spec,
    bool* cost_overflow,
    const DirectionState* ds_f,
    const DirectionState* ds_r,
    int32_t node,
    const PathRecord* records,
    int32_t record_count,
    SearchDeadline* deadline,
    double* best_cost,
    int32_t* best_node,
    int32_t* best_f_primary,
    int32_t* best_r_primary,
    int32_t* best_f_secondary,
    int32_t* best_r_secondary,
    int32_t* best_f_path,
    int32_t* best_r_path
) {
    double total = ds_f->dist[node] + ds_r->dist[node];
    if (!isfinite(total)) {
        *cost_overflow = true;
        return false;
    }
    if (cost_spec->cost_limit >= 0.0 && total > cost_spec->cost_limit) return true;
    if (*best_node != -1 && total > *best_cost) return true;
    if (*best_node != -1 && total == *best_cost) {
        if (ds_f->rank_primary[node] != *best_f_primary) {
            if (ds_f->rank_primary[node] > *best_f_primary) return true;
        } else if (ds_r->rank_primary[node] != *best_r_primary) {
            if (ds_r->rank_primary[node] > *best_r_primary) return true;
        } else if (ds_f->rank_secondary[node] != *best_f_secondary) {
            if (ds_f->rank_secondary[node] > *best_f_secondary) return true;
        } else if (ds_r->rank_secondary[node] != *best_r_secondary) {
            if (ds_r->rank_secondary[node] > *best_r_secondary) return true;
        } else {
            bool compare_ok = false;
            int cmp = middle_key_compare(
                graph,
                ds_f->path_id[node],
                ds_r->path_id[node],
                *best_f_path,
                *best_r_path,
                records,
                record_count,
                deadline,
                &compare_ok
            );
            if (!compare_ok) return false;
            if (cmp >= 0) return true;
        }
    }
    *best_cost = total;
    *best_node = node;
    *best_f_primary = ds_f->rank_primary[node];
    *best_r_primary = ds_r->rank_primary[node];
    *best_f_secondary = ds_f->rank_secondary[node];
    *best_r_secondary = ds_r->rank_secondary[node];
    *best_f_path = ds_f->path_id[node];
    *best_r_path = ds_r->path_id[node];
    return true;
}

/* Grow-on-demand arena for PathRecord chains.  Records are immutable once
 * written; orphaned records from rejected labels are harmless. */
static bool path_arena_push(
    PathRecord** records,
    int32_t* capacity,
    int32_t* count,
    int32_t parent,
    int32_t edge_rank,
    int32_t* out_id
) {
    if (*count >= *capacity) {
        if (*capacity > INT32_MAX / 2) return false;
        int32_t next_capacity = *capacity > 0 ? *capacity * 2 : 1024;
        size_t bytes = 0;
        if (!checked_size_mul((size_t)next_capacity, sizeof(PathRecord), &bytes)) {
            return false;
        }
        PathRecord* next = (PathRecord*)realloc(*records, bytes);
        if (!next) return false;
        *records = next;
        *capacity = next_capacity;
    }
    PathRecord record;
    record.parent = parent;
    record.edge_rank = edge_rank;
    (*records)[*count] = record;
    *out_id = *count;
    (*count)++;
    return true;
}

/* Walk one direction's parent chain from ``meeting`` back to its seed label,
 * collecting traversal positions meeting-first into a fresh heap buffer
 * (NULL for an empty chain) and the winning compact seed index recorded on
 * the seed label (duplicate-node seeds resolve by cost/rank/key, not by a
 * rescan of the input arrays).  Fails -- leaving no allocation -- on a
 * chain longer than node_count, a broken link, allocation failure, or
 * deadline expiry. */
static bool reconstruct_side(
    const DirectionState* ds,
    int32_t meeting,
    SearchDeadline* deadline,
    int64_t** out_positions,
    int32_t* out_count,
    int32_t* out_seed_idx
) {
    *out_positions = NULL;
    *out_count = 0;
    *out_seed_idx = -1;
    int64_t* buffer = NULL;
    int32_t capacity = 0;
    int32_t count = 0;
    int32_t curr = meeting;
    for (int64_t steps = 0; steps < ds->node_count; steps++) {
        if (deadline_charge(deadline, 1)) break;
        if (!state_seen(ds, curr, k_search_gen)) break;
        if (ds->parent[curr] == -1) {
            if (ds->seed_idx[curr] < 0) break;
            *out_positions = buffer;
            *out_count = count;
            *out_seed_idx = ds->seed_idx[curr];
            return true;
        }
        if (count >= capacity) {
            if (capacity > INT32_MAX / 2) break;
            int32_t next_capacity = capacity > 0 ? capacity * 2 : 64;
            size_t bytes = 0;
            if (!checked_size_mul((size_t)next_capacity, sizeof(int64_t), &bytes)) {
                break;
            }
            int64_t* next_buffer = (int64_t*)realloc(buffer, bytes);
            if (!next_buffer) break;
            buffer = next_buffer;
            capacity = next_capacity;
        }
        buffer[count++] = ds->trav_pos[curr];
        curr = ds->parent[curr];
    }
    free(buffer);
    return false;
}

/* Seed one search direction from its ranked seed arrays.  Returns 0 on
 * success, -3 for a non-finite or negative seed cost, and -1 for any other
 * failure (including an expired ``deadline``, which the caller detects via
 * ``deadline->expired``). */
static int32_t seed_direction(
    const CompactGraphSpec* graph,
    const CostSpec* cost_spec,
    DirectionState* ds,
    MinHeap* heap,
    const int32_t* seed_nodes,
    const double* seed_costs,
    const int32_t* seed_rank_primary,
    const int32_t* seed_rank_secondary,
    int32_t seed_count,
    bool newest_first,
    const PathRecord* records,
    int32_t record_count,
    uint64_t* next_sequence,
    int32_t* next_label,
    SearchDeadline* deadline
) {
    for (int32_t i = 0; i < seed_count; i++) {
        if (deadline_charge(deadline, 1)) return -1;
        int32_t node = seed_nodes[i];
        double cost = seed_costs[i];
        if (!isfinite(cost) || cost < 0.0) return -3;
        if (cost_spec->cost_limit >= 0.0 && cost > cost_spec->cost_limit) continue;
        if (node < 0 || (int64_t)node >= ds->node_count) continue; /* reject invalid seed ranks */
        int32_t rank_primary = seed_rank_primary[i];
        int32_t rank_secondary = seed_rank_secondary[i];
        bool compare_ok = true;
        if (!label_candidate_better(
                graph, ds, node, cost, rank_primary, rank_secondary, -1,
                records, record_count, newest_first, deadline, &compare_ok
            )) {
            if (!compare_ok) return -1;
            continue;
        }
        state_set(
            ds, node, -1, -1, cost, (*next_label)++, i, rank_primary,
            rank_secondary, -1, k_search_gen
        );
        if (!heap_push(
                heap, node, cost, rank_primary, rank_secondary,
                (*next_sequence)++, ds->label_id[node]
            )) {
            return -1;
        }
    }
    return 0;
}

/* Shared implementation for the ranked ABI entries.  Seed rank pairs are
 * required for every non-empty seed list so the search follows the Python
 * ranked ordering exactly.
 *
 * ``edge_scenic_score_by_rank`` selects the scored-sidecar mode: scenic
 * scores are then indexed by canonical edge rank
 * ``edge_scenic_score_by_rank[trav_edge_rank[pos]]`` instead of by traversal
 * position (``trav_scenic_score[pos]``).  Every rank in the traversal table
 * is validated against ``graph->edge_count`` up front and again at cost
 * evaluation; a malformed rank fails the whole search with -1 and a
 * non-finite score fails it with -4, so a corrupt sidecar can never be
 * silently misread.  NULL keeps the traversal-indexed scoring (or the 5.0
 * default when ``trav_scenic_score`` is NULL).
 *
 * Output pointers are validated before any dereference; on every failure
 * ``*out_trav_positions`` is NULL and both seed indices are -1.  The single
 * call-wide deadline (see SearchDeadline) starts before validation and is
 * re-checked unconditionally before success is committed.
 *
 * Negative returns: -1 generic failure, -2 deadline exceeded, -3 cost
 * overflow, -4 non-finite sidecar score. */
static int32_t compact_search_impl(
    const CompactGraphSpec* graph,
    const CostSpec* cost_spec,
    const double* edge_scenic_score_by_rank,
    const int32_t* fwd_seed_nodes,
    const double* fwd_seed_costs,
    const int32_t* fwd_seed_rank_primary,
    const int32_t* fwd_seed_rank_secondary,
    int32_t fwd_seed_count,
    const int32_t* rev_seed_nodes,
    const double* rev_seed_costs,
    const int32_t* rev_seed_rank_primary,
    const int32_t* rev_seed_rank_secondary,
    int32_t rev_seed_count,
    int64_t** out_trav_positions,
    double* out_total_cost,
    int32_t* out_fwd_seed_index,
    int32_t* out_rev_seed_index,
    double deadline_seconds
) {
    SearchDeadline deadline;
    const bool clock_ok = deadline_start(&deadline, deadline_seconds);
    if (!out_trav_positions || !out_total_cost || !out_fwd_seed_index ||
        !out_rev_seed_index) {
        return -1;
    }
    *out_trav_positions = NULL;
    *out_fwd_seed_index = -1;
    *out_rev_seed_index = -1;
    if (isnan(deadline_seconds)) return -1;
    if (!clock_ok) return -2; /* deadline requested but unmeasurable */
    if (!graph || !cost_spec) return -1;
    if (graph->node_count <= 0 || graph->node_count > (int64_t)INT32_MAX) {
        return -1;
    }
    if (!graph->forward_indptr || !graph->forward_indices ||
        !graph->reverse_indptr || !graph->reverse_indices ||
        !graph->reverse_positions || !graph->trav_travel_time ||
        !graph->trav_highway_mask) {
        return -1;
    }
    if (fwd_seed_count < 0 || rev_seed_count < 0) return -1;
    if (fwd_seed_count > 0 &&
        (!fwd_seed_nodes || !fwd_seed_costs || !fwd_seed_rank_primary ||
         !fwd_seed_rank_secondary)) {
        return -1;
    }
    if (rev_seed_count > 0 &&
        (!rev_seed_nodes || !rev_seed_costs || !rev_seed_rank_primary ||
         !rev_seed_rank_secondary)) {
        return -1;
    }
    if (deadline_check_now(&deadline)) return -2;
    if (edge_scenic_score_by_rank) {
        int32_t sidecar_status = validate_edge_score_sidecar(
            graph, edge_scenic_score_by_rank, &deadline
        );
        if (sidecar_status != 0) return sidecar_status;
    }
    const int64_t node_count = graph->node_count;

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
        return deadline_check_now(&deadline) ? -2 : -1;
    }

    int32_t total_positions = -1;
    PathRecord* path_records = NULL;
    int32_t path_capacity = 0;
    int32_t path_count = 0;
    bool compare_ok = true; /* path-key comparison scratch failures */
    bool cost_overflow = false;
    uint64_t next_sequence = 0; /* shared across both sides, like Python */
    int32_t next_label_f = 0;
    int32_t next_label_r = 0;

    /* The O(node_count) state allocation is indivisible; charge it here. */
    if (deadline_check_now(&deadline)) goto cleanup;

    int32_t seed_status = seed_direction(
        graph, cost_spec, &ds_f, heap_f, fwd_seed_nodes, fwd_seed_costs,
        fwd_seed_rank_primary, fwd_seed_rank_secondary, fwd_seed_count, false,
        path_records, path_count, &next_sequence, &next_label_f, &deadline
    );
    if (seed_status == 0) {
        seed_status = seed_direction(
            graph, cost_spec, &ds_r, heap_r, rev_seed_nodes, rev_seed_costs,
            rev_seed_rank_primary, rev_seed_rank_secondary, rev_seed_count, true,
            path_records, path_count, &next_sequence, &next_label_r, &deadline
        );
    }
    if (seed_status != 0) {
        cost_overflow = seed_status == -3;
        goto cleanup;
    }

    double best_cost = INFINITY;
    int32_t best_meeting_node = -1;
    int32_t best_f_primary = 0;
    int32_t best_r_primary = 0;
    int32_t best_f_secondary = 0;
    int32_t best_r_secondary = 0;
    int32_t best_f_path = -1;
    int32_t best_r_path = -1;

    while (heap_f->size > 0 && heap_r->size > 0) {
        if (deadline_charge(&deadline, 1)) goto cleanup;
        const HeapItem* top_f = &heap_f->data[0];
        const HeapItem* top_r = &heap_r->data[0];
        if (top_f->dist + top_r->dist > best_cost) {
            break;
        }

        bool expand_f = top_f->dist <= top_r->dist;
        HeapItem item;
        if (expand_f) {
            if (!heap_pop(heap_f, &item)) break;
            int32_t u = item.node_rank;
            /* Stale heap entries: a cheaper or better-key label replaced the
             * one that was pushed.  The label_id identifies the exact label,
             * mirroring Python's best_at_node/label cost checks. */
            if (!state_seen(&ds_f, u, k_search_gen) ||
                ds_f.label_id[u] != item.label_id ||
                ds_f.dist[u] != item.dist) {
                continue;
            }

            if (state_seen(&ds_r, u, k_search_gen)) {
                if (!consider_meeting(
                        graph, cost_spec, &cost_overflow, &ds_f, &ds_r, u, path_records, path_count, &deadline,
                        &best_cost, &best_meeting_node, &best_f_primary,
                        &best_r_primary, &best_f_secondary, &best_r_secondary,
                        &best_f_path, &best_r_path
                    )) {
                    goto cleanup;
                }
            }

            int64_t row_start = graph->forward_indptr[u];
            int64_t row_end = graph->forward_indptr[u + 1];
            for (int64_t pos = row_start; pos < row_end; pos++) {
                if (deadline_charge(&deadline, 1)) goto cleanup;
                bool invalid_edge_rank = false;
                double w = compute_edge_cost(
                    graph, cost_spec, pos, edge_scenic_score_by_rank,
                    &invalid_edge_rank, &cost_overflow
                );
                if (invalid_edge_rank || cost_overflow) goto cleanup;
                if (!isfinite(w)) continue;
                double next_cost = item.dist + w;
                if (!isfinite(next_cost)) {
                    cost_overflow = true;
                    goto cleanup;
                }
                if (cost_spec->cost_limit >= 0.0 && next_cost > cost_spec->cost_limit) continue;
                int32_t v = graph->forward_indices[pos];
                if (v < 0 || (int64_t)v >= node_count) continue; /* reject invalid neighbor ranks */
                int32_t edge_rank = graph->trav_edge_rank ? graph->trav_edge_rank[pos] : (int32_t)pos;
                int32_t new_path_id = -1;
                if (!path_arena_push(
                        &path_records, &path_capacity, &path_count,
                        ds_f.path_id[u], edge_rank, &new_path_id
                    )) {
                    goto cleanup;
                }
                if (!label_candidate_better(
                        graph, &ds_f, v, next_cost, ds_f.rank_primary[u],
                        ds_f.rank_secondary[u], new_path_id, path_records,
                        path_count, false, &deadline, &compare_ok
                    )) {
                    if (!compare_ok) goto cleanup;
                    continue;
                }
                state_set(
                    &ds_f, v, u, pos, next_cost, next_label_f++,
                    ds_f.seed_idx[u], ds_f.rank_primary[u],
                    ds_f.rank_secondary[u], new_path_id, k_search_gen
                );
                if (!heap_push(
                        heap_f, v, next_cost, ds_f.rank_primary[v],
                        ds_f.rank_secondary[v], next_sequence++,
                        ds_f.label_id[v]
                    )) {
                    goto cleanup;
                }
                if (state_seen(&ds_r, v, k_search_gen)) {
                    if (!consider_meeting(
                            graph, cost_spec, &cost_overflow, &ds_f, &ds_r, v, path_records, path_count, &deadline,
                            &best_cost, &best_meeting_node, &best_f_primary,
                            &best_r_primary, &best_f_secondary,
                            &best_r_secondary, &best_f_path, &best_r_path
                        )) {
                        goto cleanup;
                    }
                }
            }
        } else {
            if (!heap_pop(heap_r, &item)) break;
            int32_t u = item.node_rank;
            if (!state_seen(&ds_r, u, k_search_gen) ||
                ds_r.label_id[u] != item.label_id ||
                ds_r.dist[u] != item.dist) {
                continue;
            }

            if (state_seen(&ds_f, u, k_search_gen)) {
                if (!consider_meeting(
                        graph, cost_spec, &cost_overflow, &ds_f, &ds_r, u, path_records, path_count, &deadline,
                        &best_cost, &best_meeting_node, &best_f_primary,
                        &best_r_primary, &best_f_secondary, &best_r_secondary,
                        &best_f_path, &best_r_path
                    )) {
                    goto cleanup;
                }
            }

            int64_t row_start = graph->reverse_indptr[u];
            int64_t row_end = graph->reverse_indptr[u + 1];
            for (int64_t rev_pos = row_start; rev_pos < row_end; rev_pos++) {
                if (deadline_charge(&deadline, 1)) goto cleanup;
                int64_t pos = graph->reverse_positions[rev_pos];
                bool invalid_edge_rank = false;
                double w = compute_edge_cost(
                    graph, cost_spec, pos, edge_scenic_score_by_rank,
                    &invalid_edge_rank, &cost_overflow
                );
                if (invalid_edge_rank || cost_overflow) goto cleanup;
                if (!isfinite(w)) continue;
                double next_cost = item.dist + w;
                if (!isfinite(next_cost)) {
                    cost_overflow = true;
                    goto cleanup;
                }
                if (cost_spec->cost_limit >= 0.0 && next_cost > cost_spec->cost_limit) continue;
                int32_t v = graph->reverse_indices[rev_pos];
                if (v < 0 || (int64_t)v >= node_count) continue; /* reject invalid neighbor ranks */
                int32_t edge_rank = graph->trav_edge_rank ? graph->trav_edge_rank[pos] : (int32_t)pos;
                int32_t new_path_id = -1;
                if (!path_arena_push(
                        &path_records, &path_capacity, &path_count,
                        ds_r.path_id[u], edge_rank, &new_path_id
                    )) {
                    goto cleanup;
                }
                if (!label_candidate_better(
                        graph, &ds_r, v, next_cost, ds_r.rank_primary[u],
                        ds_r.rank_secondary[u], new_path_id, path_records,
                        path_count, true, &deadline, &compare_ok
                    )) {
                    if (!compare_ok) goto cleanup;
                    continue;
                }
                state_set(
                    &ds_r, v, u, pos, next_cost, next_label_r++,
                    ds_r.seed_idx[u], ds_r.rank_primary[u],
                    ds_r.rank_secondary[u], new_path_id, k_search_gen
                );
                if (!heap_push(
                        heap_r, v, next_cost, ds_r.rank_primary[v],
                        ds_r.rank_secondary[v], next_sequence++,
                        ds_r.label_id[v]
                    )) {
                    goto cleanup;
                }
                if (state_seen(&ds_f, v, k_search_gen)) {
                    if (!consider_meeting(
                            graph, cost_spec, &cost_overflow, &ds_f, &ds_r, v, path_records, path_count, &deadline,
                            &best_cost, &best_meeting_node, &best_f_primary,
                            &best_r_primary, &best_f_secondary,
                            &best_r_secondary, &best_f_path, &best_r_path
                        )) {
                        goto cleanup;
                    }
                }
            }
        }
    }

    if (best_meeting_node != -1 && isfinite(best_cost)) {
        int64_t* fwd_positions = NULL;
        int64_t* rev_positions = NULL;
        int64_t* result_buffer = NULL;
        int32_t fwd_count = 0;
        int32_t rev_count = 0;
        int32_t fwd_start_seed_idx = -1;
        int32_t rev_start_seed_idx = -1;
        bool ok = reconstruct_side(
                      &ds_f, best_meeting_node, &deadline, &fwd_positions,
                      &fwd_count, &fwd_start_seed_idx
                  ) &&
                  reconstruct_side(
                      &ds_r, best_meeting_node, &deadline, &rev_positions,
                      &rev_count, &rev_start_seed_idx
                  );
        int64_t total = (int64_t)fwd_count + (int64_t)rev_count;
        if (ok && total > INT32_MAX) ok = false;
        if (ok && total > 0) {
            size_t bytes = 0;
            ok = checked_size_mul((size_t)total, sizeof(int64_t), &bytes);
            if (ok) {
                result_buffer = (int64_t*)malloc(bytes);
                ok = result_buffer != NULL;
            }
            if (ok) {
                /* The forward chain was collected meeting-first; emit it
                 * seed-first, then the reverse chain meeting-first. */
                int32_t idx = 0;
                for (int32_t i = fwd_count - 1; i >= 0; i--) {
                    result_buffer[idx++] = fwd_positions[i];
                }
                for (int32_t i = 0; i < rev_count; i++) {
                    result_buffer[idx++] = rev_positions[i];
                }
            }
        }
        free(fwd_positions);
        free(rev_positions);
        /* Final unconditional check: success is never committed past the
         * deadline, however little amortized work remained. */
        if (ok && deadline_check_now(&deadline)) ok = false;
        if (ok) {
            *out_trav_positions = result_buffer;
            *out_total_cost = best_cost;
            *out_fwd_seed_index = fwd_start_seed_idx;
            *out_rev_seed_index = rev_start_seed_idx;
            total_positions = (int32_t)total;
        } else {
            free(result_buffer);
        }
    }

cleanup:
    if (cost_overflow) {
        total_positions = -3;
    } else if (deadline.expired) {
        total_positions = -2;
    }
    heap_free(heap_f);
    heap_free(heap_r);
    direction_state_free(&ds_f);
    direction_state_free(&ds_r);
    free(path_records);
    return total_positions;
}

/* Ranked ABI entry.  Per-seed (rank_primary, rank_secondary) pairs for both
 * directions reproduce the Python ranked search's seed (index, direction)
 * rank keys so duplicate-node seeds and equal-cost paths resolve
 * identically; they are required whenever the matching seed count is > 0.
 *
 * Negative returns: -1 generic failure, -2 deadline exceeded, -3 cost
 * overflow. */
int32_t compact_bidirectional_search_alloc_ranked(
    const CompactGraphSpec* graph,
    const CostSpec* cost_spec,
    const int32_t* fwd_seed_nodes,
    const double* fwd_seed_costs,
    const int32_t* fwd_seed_rank_primary,
    const int32_t* fwd_seed_rank_secondary,
    int32_t fwd_seed_count,
    const int32_t* rev_seed_nodes,
    const double* rev_seed_costs,
    const int32_t* rev_seed_rank_primary,
    const int32_t* rev_seed_rank_secondary,
    int32_t rev_seed_count,
    int64_t** out_trav_positions,
    double* out_total_cost,
    int32_t* out_fwd_seed_index,
    int32_t* out_rev_seed_index,
    double deadline_seconds
) {
    return compact_search_impl(
        graph,
        cost_spec,
        NULL,
        fwd_seed_nodes,
        fwd_seed_costs,
        fwd_seed_rank_primary,
        fwd_seed_rank_secondary,
        fwd_seed_count,
        rev_seed_nodes,
        rev_seed_costs,
        rev_seed_rank_primary,
        rev_seed_rank_secondary,
        rev_seed_count,
        out_trav_positions,
        out_total_cost,
        out_fwd_seed_index,
        out_rev_seed_index,
        deadline_seconds
    );
}

/* Edge-score sidecar ABI entry: identical to compact_bidirectional_search_
 * alloc_ranked plus the canonical-edge-rank-indexed scenic score array
 * ``edge_scenic_score_by_rank`` (length ``graph->edge_count``).  This is the
 * entry the wrapper selects only when the active CompactRoadGraph carries a
 * score sidecar; the ranked entry keeps its traversal-indexed scoring
 * contract unchanged.  The sidecar is read-only: no remap or E-sized copy is
 * ever made.  A rank outside [0, edge_count) fails with -1 and a non-finite
 * score with -4, both before any search allocation.
 *
 * Negative returns: -1 generic failure (including malformed ranks),
 * -2 deadline exceeded, -3 cost overflow, -4 non-finite sidecar score. */
int32_t compact_bidirectional_search_alloc_ranked_edge_scores(
    const CompactGraphSpec* graph,
    const CostSpec* cost_spec,
    const double* edge_scenic_score_by_rank,
    const int32_t* fwd_seed_nodes,
    const double* fwd_seed_costs,
    const int32_t* fwd_seed_rank_primary,
    const int32_t* fwd_seed_rank_secondary,
    int32_t fwd_seed_count,
    const int32_t* rev_seed_nodes,
    const double* rev_seed_costs,
    const int32_t* rev_seed_rank_primary,
    const int32_t* rev_seed_rank_secondary,
    int32_t rev_seed_count,
    int64_t** out_trav_positions,
    double* out_total_cost,
    int32_t* out_fwd_seed_index,
    int32_t* out_rev_seed_index,
    double deadline_seconds
) {
    return compact_search_impl(
        graph,
        cost_spec,
        edge_scenic_score_by_rank,
        fwd_seed_nodes,
        fwd_seed_costs,
        fwd_seed_rank_primary,
        fwd_seed_rank_secondary,
        fwd_seed_count,
        rev_seed_nodes,
        rev_seed_costs,
        rev_seed_rank_primary,
        rev_seed_rank_secondary,
        rev_seed_count,
        out_trav_positions,
        out_total_cost,
        out_fwd_seed_index,
        out_rev_seed_index,
        deadline_seconds
    );
}

void compact_free_positions(int64_t* positions) {
    free(positions);
}
