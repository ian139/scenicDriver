# Infrastructure Architecture

This document outlines the evolving infrastructure plan for the Scenic Route Planner. It is written to be both **generally readable** and **production-oriented**.

The system is intentionally designed as a hybrid **ML + systems** project: reproducible experiments, efficient storage, fast scoring, and low-latency routing.

---

# Production-Ready Architecture Spec (Readable)

## Goals

* **Correctness & reproducibility:** anyone can rerun experiments and reproduce outputs from a config + dataset version.
* **Fast iteration:** avoid re-downloading tiles and re-embedding images.
* **Low-latency routing:** interactive route results (target: sub-second for typical queries).
* **Scalability:** support expanding regions and zoom levels without rewriting the pipeline.
* **Cost-aware:** minimize GPU recomputation and unnecessary storage.

## Non-Goals (for now)

* Real-time, worldwide coverage
* Personalized scenic taste modeling (per-user preferences)
* Fully distributed training at scale

## System Overview

**Offline pipeline** produces caches and indexes. **Online service** uses those artifacts to score routes quickly.

### Offline Artifacts (what we precompute)

* **Tiles** (satellite + terrain)
* **Terrain features** per tile
* **Satellite embeddings** per tile (Step 3)
* **Road → tile mapping index** (Step 2)
* Optional: **Road segment scenic score** (pre-aggregated)

### Online Responsibilities (what must be fast)

* Route query → candidate paths
* Scenic aggregation (from cached data)
* Return a small set of routes with tradeoffs (fastest vs scenic)

## Data Flow (High Level)

1. Download tiles for region (satellite + terrain)
2. Compute terrain features
3. Compute satellite embeddings and cache
4. Train scenic model (regression or ranking)
5. Build road mapping index (road segments → tile_ids)
6. Serve routing queries using cached artifacts

## Interfaces (contracts)

These interfaces are what keep the system modular.

### Tile Identity

A tile must be uniquely addressable:

* `tile_id` = `(z, x, y, source)` or stable hash of those

### Embedding Store

* **Input:** `tile_id`
* **Output:** `embedding_vector` + metadata (encoder version)

### Feature Store

* **Input:** `tile_id`
* **Output:** terrain feature vector + any derived stats

### Road Index

* **Input:** `road_segment_id`
* **Output:** ordered `tile_id[]` (and optionally weights/coverage)

### Scenic Model

* **Input:** fused vector (`embedding` + `terrain_features` + optional classifier logits)
* **Output:** `scenic_score` (0–10) or ranking score

### Routing Engine

* **Input:** start/end coordinates + parameters (α, β)
* **Output:** N routes + metrics (ETA, distance, scenic)

## Performance Targets (initial)

### Offline Targets

* Tile download throughput: network-bound, but batching should avoid API throttling
* Terrain feature computation: scalable linearly with tile count
* Embedding generation: GPU-saturated batches (target >80% utilization)
* Full region preprocessing (≈5k–10k tiles): minutes, not hours

### Online Targets

* Typical route query (city-scale): **< 1 second** end-to-end
* Scenic aggregation per candidate route: **< 200ms**
* Graph search (A* or Dijkstra over filtered graph): **< 300ms**
* Scenic scoring for a single tile (cached embedding): **< 2ms**

### Memory & Storage Targets

* Embedding size per tile: 256–1024 dims (float16 preferred)
* Embedding storage per 10k tiles: manageable (< ~50–100MB range)
* Road index lookup: O(1) or O(log n) via hashed or spatial index

### Scalability Expectations

* System should handle incremental region additions without full rebuild
* Re-embedding required only when encoder version changes
* Road index rebuild required only when tile coverage changes

---

## Full Data Flow Diagram (Offline + Online)

### OFFLINE PIPELINE

```
            ┌────────────────────┐
            │  Mapbox Tile APIs  │
            └─────────┬──────────┘
                      │
                      ▼
            ┌────────────────────┐
            │  Raw Tile Storage  │
            │ (satellite/terrain)│
            └─────────┬──────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌──────────────────┐     ┌──────────────────┐
│ Terrain Decoder  │     │ Satellite Encoder│
│ (RGB → features) │     │ (CNN / ViT)      │
└─────────┬────────┘     └─────────┬────────┘
          │                          │
          ▼                          ▼
┌──────────────────┐     ┌──────────────────┐
│ Terrain Feature  │     │ Embedding Store  │
│ Store            │     │ (tile → vector)  │
└─────────┬────────┘     └─────────┬────────┘
          └──────────┬──────────────┘
                     ▼
          ┌────────────────────┐
          │ Scenic Model Train │
          │ (regression/rank)  │
          └─────────┬──────────┘
                    ▼
          ┌────────────────────┐
          │ Road → Tile Index  │
          │ (spatial mapping)  │
          └────────────────────┘
```

Artifacts produced:

* Raw tiles
* Terrain features per tile
* Satellite embeddings per tile
* Scenic model checkpoint
* Road segment → tile_id index

---

### ONLINE QUERY FLOW

```
User Request (start, end, α, β)
            │
            ▼
┌────────────────────┐
│ Baseline Route     │
│ Candidate Generator│
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Routing Engine     │
│ (A* / Dijkstra)    │
└─────────┬──────────┘
          │ needs scenic edge weights
          ▼
┌─────────────────────────────┐
│ Road Segment Lookup         │
│ (road_id → tile_ids)        │
└─────────┬───────────────────┘
          │
          ▼
┌─────────────────────────────┐
│ Embedding Store             │
│ + Terrain Feature Store     │
└─────────┬───────────────────┘
          │
          ▼
┌─────────────────────────────┐
│ Scenic Model Inference      │
│ (fused vector → score)      │
└─────────┬───────────────────┘
          ▼
   Aggregated Scenic Score
          │
          ▼
     Ranked Route Output
```

Key Insight:

* **No raw image decoding happens online.**
* Online path only touches: road index → cached embeddings → lightweight model head.
* Heavy compute is entirely offline.

---

## Storage Layout (recommended)

```
data/
  raw/
    images/
      satellite/
      terrain/
    manifests/
  processed/
    features/
      terrain/
      embeddings/
    indexes/
      road_to_tiles/
    runs/
models/
```

## Versioning Rules (keep sanity)

* Raw data is immutable (append-only)
* Processed caches are versioned by:

  * dataset manifest hash
  * preprocessing version
  * encoder checkpoint hash
* Models are versioned with:

  * training config
  * dataset version
  * code commit hash

---

# 1. Local ML Infrastructure (Reproducibility First)

## Dataset Versioning

* Track dataset regions, bounding boxes, zoom levels, and timestamps
* Store metadata alongside tiles
* Maintain a dataset manifest file (JSON/YAML)
* Hash tile batches for reproducibility

## Run Tracking

* Log every training run:

  * Dataset version
  * Model architecture
  * Hyperparameters
  * Git commit hash
  * Metrics
* Store logs in `data/processed/runs/`

## Deterministic Configs

* Centralized config files (e.g. YAML or Pydantic-based)
* No hardcoded paths or magic numbers
* Every experiment must be reproducible from config

---

# 2. Storage & Data Lifecycle

## Raw vs Processed Separation

```
data/raw/images/
data/processed/
models/
```

Raw data is immutable.
Processed data is regenerable.
Models are versioned with metadata.

## S3 Lifecycle Strategy

* Raw tiles → Standard → Infrequent Access → Glacier
* Processed features cached for faster iteration
* No unnecessary local duplication

## Embedding Cache Layer

Instead of recomputing embeddings every time:

```
tile_id → embedding_vector
```

Design considerations:

* Storage format (Parquet / NumPy / Torch tensor files)
* Memory mapping for fast reads
* Version embeddings by encoder checkpoint
* Batch loading to minimize disk I/O

---

# 3. Embedding Layer (Core ML Systems Component)

## Learned Embedding Cache (Step 3)

Introduce a dedicated embedding layer between raw tiles and regression scoring:

```
tile_id → satellite_embedding_vector
```

This embedding is produced by a frozen (or versioned) encoder such as a ResNet or ViT trained on satellite imagery.

Why this is a strong architectural decision:

* **Decouples representation from scoring**
  The scenic regressor can change without recomputing raw image processing.

* **Enables rapid experimentation**
  You can swap regression heads, ranking models, or aggregation strategies while reusing embeddings.

* **Reduces compute cost**
  CNN/ViT forward passes are expensive. Caching embeddings avoids repeated GPU inference.

* **Creates a reusable feature store**
  Embeddings can power:

  * Scenic regression
  * Pairwise ranking
  * Clustering scenic regions
  * Similarity search ("find routes like this")

* **Supports versioning discipline**
  Embeddings must be stored with:

  * Encoder checkpoint hash
  * Dataset version
  * Preprocessing version

Design considerations:

* Storage format (Parquet / NumPy / Torch tensors)
* Memory-mapped loading for fast reads
* Batch retrieval for road-level aggregation
* Optional compression (float16)

Optional extension:

```
tile_id → fused_embedding (satellite + terrain_features)
```

This moves toward a unified multimodal representation.

---

# 4. Precomputed Feature Layer

## Terrain Feature Cache

Precompute:

* Elevation
* Slope
* Local variance
* Relief metrics

Store as structured feature vectors per tile.

Avoid recomputing terrain decoding from RGB at runtime.

---

# 4. Spatial Indexing & Road Segment Mapping

## Road Segment Index (Step 2)

Precompute:

```
road_segment_id → [tile_id_1, tile_id_2, ...]
```

This enables:

* Fast scenic aggregation per road
* Avoiding repeated spatial lookup
* Efficient route scoring

Design considerations:

* Spatial joins (tile bbox vs road polyline)
* R-tree or spatial indexing
* Cache invalidation when tile dataset changes

Optional extension:

```
road_segment_id → aggregated_scenic_score
```

Precompute static scenic values for fast routing.

---

# 5. Scenic Routing Engine

## Edge Weight Definition

```
edge_cost = α * travel_time - β * scenic_score
```

Where:

* travel_time is baseline cost
* scenic_score is aggregated from tiles
* α, β are tunable parameters

## Routing Algorithm

* Dijkstra (baseline)
* A* with heuristic (production candidate)

Considerations:

* Latency constraints (<500ms target)
* Precomputed scenic edges
* Caching frequent routes

---

# 6. Observability & Monitoring

## Metrics to Track

* Model prediction distribution
* Scenic score drift by region
* Routing latency
* Cache hit/miss ratio
* Tile download failures

## Logging

* Structured JSON logs
* Route computation breakdown (I/O vs compute)

---

# 7. Failure Modes & Tradeoffs

## Storage vs Compute

* Precompute everything → more storage
* Compute on demand → higher latency

## Model Drift

* Region bias (West Coast vs Northeast)
* Seasonal variation

## Scalability

* Increasing tile density increases embedding storage
* Road mapping must remain O(log n) spatial query

---

# 8. Future Extensions

* Move to vector database for embeddings
* Serve model behind FastAPI endpoint
* Batch offline scenic graph generation
* Distributed tile processing
* Contrastive scenic ranking model

---

# Guiding Principle

This project is not just a model. It is:

* A data pipeline
* A feature store
* A spatial indexing system
* A graph optimization engine
* An ML scoring model

Design decisions should reflect production-level systems thinking from the start.
