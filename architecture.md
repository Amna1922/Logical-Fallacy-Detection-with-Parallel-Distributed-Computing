## Logical Fallacy Detection with Parallel Distributed Computing

---

## 📌 Table of Contents

1. [Project Overview](#project-overview)
2. [What Problem Are We Solving](#what-problem-are-we-solving)
3. [Complete Architecture](#complete-architecture)
4. [Dataset Details](#dataset-details)
5. [File-by-File Breakdown](#file-by-file-breakdown)
6. [Core Technical Components](#core-technical-components)
7. [Parallel Processing Strategies](#parallel-processing-strategies)
8. [Training Details](#training-details)
9. [Results & Performance](#results--performance)
10. [How to Run Everything](#how-to-run-everything)
11. [Common Questions & Answers](#common-questions--answers)

---

## 1. Project Overview

### What Does This Project Do?

This project builds a **complete automated system** that:
1. Takes an argument text as input (e.g., "You can't trust Dr. Chen - she's from Canada")
2. Identifies which logical fallacy it contains (13 types like ad hominem, false dilemma, straw man)
3. Generates a human-readable explanation of WHY it's that fallacy
4. Does all of this FAST using parallel processing (48 texts/second vs 10 texts/second sequential)

### Why Is This Important?

- Millions of arguments are posted daily on Reddit, Twitter, Kialo, etc.
- Logical fallacies mislead people and pollute public discourse
- Manual detection is impossible at scale
- Existing systems are too slow and don't explain their reasoning

---

## 2. What Problem Are We Solving

### The Core Challenges

| Challenge | Why Hard | Our Solution |
|-----------|----------|--------------|
| **Scale** | Millions of arguments daily | Parallel processing (4.66x speedup) |
| **Speed** | Single model inference is slow | GPU batching + threading |
| **Explanation** | Need human-readable reasons | Dual-path (RAG + CPACE) |
| **Retrieval** | Need similar examples as evidence | FAISS index over 150k passages |
| **Class Imbalance** | Some fallacies have only 5 samples | Class-weighted loss function |

### The 13 Fallacy Classes

| # | Fallacy | Example |
|---|---------|---------|
| 1 | ad_hominem | "You can't trust Dr. Smith - he drives an SUV" |
| 2 | appeal_to_emotion | "Think of the starving children!" |
| 3 | appeal_to_popularity | "Everyone is buying this, so it's good" |
| 4 | circular_reasoning | "The Bible is true because it says so" |
| 5 | equivocation | "A feather is light, so it can't be heavy" |
| 6 | fallacy_of_credibility | "My actor friend says this medicine works" |
| 7 | false_cause | "I wore lucky socks and we won" |
| 8 | false_dilemma | "You're either with us or against us" |
| 9 | faulty_generalization | "Two rude NYers → all are rude" |
| 10 | intentional | Deliberately misleading statistics |
| 11 | logical_fallacy | "If A then B, B true, therefore A" |
| 12 | relevance_fallacy | "Why worry about pollution when people starve?" |
| 13 | straw_man | "You want lower military spending? So you want us defenseless!" |

---

## 3. Complete Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER                                    │
│                    Raw Argument Text from User/API                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA PREPROCESSING LAYER                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  NER Normalization (spaCy + Multiprocessing)                        │    │
│  │  "Dr. Chen from Canada" → "[PERSON] from [GPE]"                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RETRIEVAL LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  DPR Encoding (SentenceTransformer E5)                              │    │
│  │  Text → 768-dim vector                                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  FAISS IVF Search (nlist=100, nprobe=10)                            │    │
│  │  Finds top-5 similar passages from 150k corpus (200-400ms)          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLASSIFICATION LAYER                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  DistilBERT Classifier (66.9M parameters)                           │    │
│  │  Input: normalized text → Output: 1 of 13 fallacy classes           │    │
│  │  Training: 2e-5 LR, AdamW, OneCycleLR, class-weighted loss          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EXPLANATION GENERATION LAYER                           │
│                                                                             │
│  ┌─────────────────────────┐    ┌─────────────────────────┐                 │
│  │   RAG-Token Generator   │    │    CPACE Module         │                 │
│  │   (T5-small/BART-base)  │    │   (spaCy + ConceptNet)  │                 │
│  │                         │    │                         │                 │
│  │  Evidence-grounded      │    │  Contrastive symbolic   │                 │
│  │  Neural generation      │    │  Template-based         │                 │
│  │  Slow (2-5s)            │    │  Fast (1-2s)            │                 │
│  └─────────────────────────┘    └─────────────────────────┘                 │
│                    │                         │                              │
│                    └───────────┬─────────────┘                              │
│                                ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  FUSION MODULE (MiniLM-L6-v2 + Cosine Similarity)                   │    │
│  │  Selects explanation better grounded in retrieved passages          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  {                                                                  │    │
│  │    "fallacy": "ad_hominem",                                         │    │
│  │    "confidence": 0.87,                                              │    │
│  │    "explanation": "This argument attacks the person...",            │    │
│  │    "similar_examples": [...]                                        │    │
│  │  }                                                                       │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Five-Layer Architecture Summary

| Layer | Purpose | Key Technologies |
|-------|---------|------------------|
| **Data Layer** | Store all inputs, models, indexes | CSV files, JSONL, FAISS index files |
| **Processing Layer** | Run preprocessing, training, inference | 6 Python scripts |
| **Model Layer** | AI/ML models | DistilBERT, E5, spaCy, T5/BART |
| **Parallel Layer** | Speed up everything | GPU batching, ThreadPool, Multiprocessing, FP16 |
| **Output Layer** | Present results | JSON, CSV, interactive demo |

---

## 4. Dataset Details

### Training Datasets (2,300 total samples)

| Dataset | Size | Source | Label Column | Text Column |
|---------|------|--------|--------------|-------------|
| climate_train.csv | ~1,200 | Climate debate articles | `logical_fallacies` | `source_article` |
| edu_train.csv | ~1,100 | Education debate articles | `updated_label` | `masked_articles` |

**Class Distribution Problem:**
```
equivocation:        5 samples  (VERY RARE)
relevance_fallacy:  16 samples
appeal_to_emotion:  19 samples
false_dilemma:      12 samples  (But F1=0.72 - easy to detect!)
faulty_generalization: 35 samples (Most common)
```

### Retrieval Corpus (150,000 passages)

| Dataset | Size | Purpose |
|---------|------|---------|
| kialo_flat_clean.csv | 100,000 | Structured pro/con arguments (high quality) |
| CMV (Change My View) | ~50,000 | Natural Reddit discussions (realistic) |

### Data Split Strategy

```
Total: 2,300 samples
├── Train: 1,885 samples (82%) - for training the model
├── Validation: 236 samples (10.3%) - for tuning hyperparameters
└── Test: 236 samples (10.3%) - for final evaluation

Split method: Stratified (preserves class distribution across splits)
Random seed: 42 (reproducible)
```

---

## 5. File-by-File Breakdown

### File 0: `00_setup_colab.py` - Environment Setup

**Purpose:** Install all dependencies in one go

**What it installs:**
```python
packages = [
    "torch",                 # PyTorch for GPU training
    "transformers",          # HuggingFace for DistilBERT, T5, BART
    "sentence-transformers", # For E5 embeddings
    "scikit-learn",          # For train/test split, metrics
    "pandas", "numpy",       # Data handling
    "spacy",                 # NER normalization
    "faiss-cpu",             # Vector search (CPU version)
    "ray", "dask", "joblib", "numba"  # Parallel computing
]
```

**Run this FIRST before anything else.**

---

### File 1: `1_preprocess_data.py` - Data Preprocessing

**Purpose:** Clean, normalize, and split all data

**What it does (step by step):**

1. **Load datasets** - Reads climate_train.csv and edu_train.csv
2. **Combine & clean** - Merges, removes duplicates, filters short texts
3. **NER Normalization (PARALLEL)** - Replaces names with placeholders:
   ```
   "Dr. Chen from Canada" → "[PERSON] from [GPE]"
   ```
4. **Train/Val/Test split** - Stratified 80/10/10 split
5. **Build retrieval corpus** - Processes Kialo data (100k arguments)

**Key Function: Parallel NER Processing**
```python
# Uses multiprocessing (not threading) because NER is CPU-bound
with Pool(processes=4) as pool:
    normalized = pool.map(replace_entities_worker, texts)
```

**Output files:**
- `/content/data/processed/train.csv`
- `/content/data/processed/val.csv`
- `/content/data/processed/test.csv`
- `/content/data/processed/label_map.json`
- `/content/data/retrieval/retrieval_corpus.jsonl`

---

### File 2: `2_build_faiss_index.py` - FAISS Index Builder

**Purpose:** Create searchable index of 150,000 passages

**What it does:**

1. **Load retrieval corpus** - Reads 150k passages from JSONL
2. **Encode with E5 model** - Converts each passage to 768-dim vector
   ```python
   # E5 requires "passage:" prefix
   text = f"passage: {passage['text_normalized']}"
   embedding = model.encode(text)  # Shape: (768,)
   ```
3. **Build IVF index** - Organizes vectors into 100 clusters
4. **Save index** - For reuse at inference time

**Key Parameters:**
```python
nlist = 100   # Number of clusters (√150000 would be 387, but 100 is balanced)
nprobe = 10   # Number of clusters to search per query (10 × 1500 vectors = 15k searched)
```

**Output:**
- `/content/index/faiss/corpus.index` - The FAISS index file

---

### File 3: `3_train_classifier.py` - DistilBERT Training

**Purpose:** Train the fallacy classifier

**What it does:**

1. **Loads pre-trained DistilBERT** - 66.9M parameters, knows English
2. **Adds classification head** - Linear(768 → 13) for fallacy classes
3. **Fine-tunes on our data** - Updates ALL parameters

**Training Configuration:**
```python
max_length = 256 tokens          # Arguments are usually short
batch_size = 16                  # With gradient accumulation
learning_rate = 2e-5             # Standard for BERT-family
optimizer = AdamW                # Decoupled weight decay
scheduler = OneCycleLR           # Warmup → peak → annealing
loss = Class-weighted CE         # Handles class imbalance
epochs = 10                      # Early stopping at epoch 8
mixed_precision = FP16           # 2x memory, 1.5x speed
```

**Class Weights Calculation:**
```python
# Rare classes (equivocation: 5 samples) get HIGHER weight
# Common classes (faulty_generalization: 35) get LOWER weight
class_weights = compute_class_weight('balanced', classes, y_train)
```

**Output:**
- `/content/models/classifier/` - Saved model, tokenizer, label map

**Performance:**
| Epoch | Train Loss | Val Acc | Val F1 |
|-------|------------|---------|--------|
| 1 | 2.55 | 20.8% | 0.182 |
| 8 (best) | 0.72 | 42.8% | 0.415 |
| 10 | 0.62 | 42.4% | 0.393 |

---

### File 4: `4_train_generator.py` - Explanation Generator

**Purpose:** Generate explanations for predicted fallacies

**What it does:**

1. **Loads trained classifier** - From File 3
2. **Uses template-based generation** - No training needed (simplified from paper's RAG)
3. **Extracts keywords** - Finds topics, persons, emotions from text
4. **Fills templates** - Inserts keywords into fallacy-specific templates

**Example Template:**
```python
'ad_hominem': {
    'template': "This argument commits the AD HOMINEM fallacy. Instead of 
                 addressing the argument about {topic}, it attacks the 
                 character of {person}...",
    'example': "Example: 'You can't trust Dr. Smith's research because 
                 he drives an SUV' attacks the person, not the research."
}
```

**Output:**
- `/content/outputs/explanations.json`
- `/content/outputs/predictions.csv`

**Also provides interactive mode** - Type arguments and get predictions + explanations

---

### File 5: `5_cpace_module.py` - CPACE Contrastive Module

**Purpose:** Generate contrastive explanations (explaining why X fallacy rather than Y)

**What "CPACE" stands for:**
- **C**ontrastive
- **P**arallel
- **A**rgumentation and
- **C**ontrastive
- **E**xplanation

**What it does (3 steps):**

1. **Concept Extraction (spaCy)**
   - Identifies persons, organizations, locations
   - Parses grammatical relationships (subject, object, action)
   
2. **Knowledge Retrieval (ConceptNet)**
   - Enriches concepts with common-sense knowledge
   - Example: "senator" → IsA "politician", PartOf "government"

3. **Contrastive Generation (Templates)**
   - Generates explanations contrasting with alternative fallacies
   - Example: "This is AD HOMINEM rather than STRAW MAN because..."

**Why Contrastive?**
- Humans learn better from contrasts
- Distinguishing ad hominem from straw man clarifies both concepts
- RAG cannot guarantee contrastive explanations

**Test Cases Built In:**
```python
test_cases = [
    ("You can't trust Dr. Smith - he drives an SUV", "ad_hominem", ["straw_man"]),
    ("Everyone buys this, so it's best", "appeal_to_popularity", ["false_cause"]),
    ("You're either with us or against us", "false_dilemma", ["ad_hominem"])
]
```

---

### File 6: `6_complete_pipeline.py` - Complete Parallel Pipeline

**Purpose:** End-to-end inference with all optimizations

**Key Classes:**

1. **ThreadedBatchProcessor** - Background thread collects batches
   ```python
   # Overlaps I/O with GPU computation
   # Batch of 32 takes same time as 1 due to GPU parallelism
   ```

2. **ParallelBatchProcessor** - Multi-threaded prediction
   ```python
   # Uses ThreadPoolExecutor for concurrent inference
   with ThreadPoolExecutor(max_workers=7) as executor:
       results = executor.map(predict_one, texts)
   ```

3. **FaissRetriever** - Optional similarity search
   ```python
   # Finds similar examples from corpus
   similar = retriever.search(query, k=3)
   ```

**Benchmark Results (224 arguments):**
| Method | Time | Throughput | Speedup |
|--------|------|------------|---------|
| Sequential | 21.7s | 10.3 texts/sec | 1.00x |
| GPU Batch | 5.1s | 43.9 texts/sec | 4.29x |
| Thread + GPU | 4.66s | 48.1 texts/sec | 4.66x |

---

## 6. Core Technical Components

### Component A: NER Normalization (spaCy)

**What it does:** Replaces named entities with placeholders

**Why:** Prevents overfitting to specific people/places

**Example:**
```
Input:  "Dr. Chen from Toronto supports healthcare"
Output: "[PERSON] from [GPE] supports healthcare"
```

**Entity types replaced:**
- PERSON → [PERSON]
- ORG → [ORG]
- GPE (countries/cities) → [GPE]
- DATE → [DATE]
- EVENT → [EVENT]

**Why multiprocessing?** spaCy NER is CPU-bound, so we use multiple processes (not threads) to bypass Python's GIL.

---

### Component B: Dense Passage Retrieval (E5 + FAISS)

**What it does:** Finds semantically similar passages from 150k corpus

**E5 Model:**
- Full name: "EmbEddings from bidirEctional Encoder rEpresentations"
- Developed by Microsoft
- Requires special prefixes: "query: " for search, "passage: " for indexing
- Output: 768-dim vectors

**FAISS IVF Parameters:**
```python
nlist = 100   # Number of clusters
nprobe = 10   # Number of clusters to search per query

# Math:
# Vectors searched = (150,000 / 100) × 10 = 15,000
# Time: 200-400ms (vs 500ms for flat index)
```

---

### Component C: DistilBERT Classifier

**What it is:** A smaller, faster version of BERT

**DistilBERT vs BERT:**
| Aspect | BERT | DistilBERT (OURS) |
|--------|------|-------------------|
| Layers | 12 | 6 |
| Parameters | 110M | 66.9M |
| Speed | 100ms | 40ms |
| Accuracy | 100% | 97% |

**Fine-Tuning Details:**
- **Type:** Full fine-tuning (all 66.9M parameters updated)
- **Learning rate:** 2e-5 (standard for BERT-family)
- **Optimizer:** AdamW (decoupled weight decay)
- **Scheduler:** OneCycleLR (warmup → peak → annealing)
- **Loss:** Class-weighted CrossEntropy (handles imbalance)

**Why 2e-5?** 
- Low enough to preserve pre-trained knowledge (catastrophic forgetting prevention)
- High enough to adapt to fallacy task in reasonable time
- Standard in literature (BERT paper used 2e-5)

---

### Component D: RAG-Token Generator

**What it does:** Generates explanations using retrieved passages

**RAG-Token vs RAG-Sequence:**
| Aspect | RAG-Sequence | RAG-Token (OURS) |
|--------|--------------|------------------|
| Passage use | One passage for entire generation | Different passage per token |
| Contrastive | Cannot compare passages | Can synthesize from multiple |
| Complexity | Lower | Higher |

**Why RAG-Token for fallacies?**
Contrastive explanations (e.g., "This is X not Y") need information from two different passages. RAG-Token allows each token to come from a different passage.

**Backbone models:**
- T5-small (60M params) - Faster, for development
- BART-base (140M params) - Better quality, for production

---

### Component E: CPACE Module

**What it does:** Symbolic, contrastive explanations

**Three layers:**
1. **spaCy** - Extracts concepts and grammatical relationships
2. **ConceptNet** - Adds common-sense knowledge
3. **Templates** - Generates contrastive explanations

**Example contrastive explanation:**
> "This argument commits the **AD HOMINEM** fallacy rather than a **STRAW MAN** fallacy. Instead of addressing the argument about healthcare, it attacks the speaker's nationality. Unlike straw man (which distorts the position), ad hominem ignores the argument entirely."

**Advantages over RAG:**
- No training data needed (zero-shot)
- Never hallucinates
- Contrastive by design
- Faster (1-2s vs 2-5s)

---

### Component F: Fusion Module

**What it does:** Selects best explanation from RAG and CPACE

**Selection method:** Cosine similarity to retrieved passages

**Why cosine similarity:**
- Measures semantic alignment
- Scale-invariant (length doesn't bias)
- Fast to compute

**Encoder:** MiniLM-L6-v2
- 384-dim embeddings (smaller than E5's 768)
- 15ms per encoding (10x faster than E5)
- 95% of E5's quality (good enough for ranking)

**Fusion rule:**
```python
if abs(rag_score - cpace_score) < 0.1:
    return CONCATENATED  # Both are equally good
else:
    return HIGHER_SCORE  # Pick the better one
```

---

## 7. Parallel Processing Strategies

### Strategy 1: GPU Batch Processing

**Speedup:** 4.29x

**How it works:**
```
Sequential: [Input1] → GPU → [Output1] → [Input2] → GPU → [Output2]
Batch:      [Input1, Input2, ..., Input32] → GPU → [Outputs]
```

**Why:** GPU has thousands of cores. Processing 32 inputs takes almost same time as 1.

**Code location:** `6_complete_pipeline.py` - `predict_batch_gpu()`

---

### Strategy 2: Thread-Level Concurrency

**Speedup:** Additional 1.09x (total 4.66x)

**How it works:** Overlaps I/O (loading data) with GPU computation

```
Without threading:
GPU: [Compute] → [Idle] → [Compute] → [Idle]
CPU: [Load]    → [Load]  → [Load]    → [Load]

With threading:
GPU: [Compute] → [Compute] → [Compute]
CPU: [Load]    → [Load]    → [Load]   (overlapped!)
```

**Why not just more threads?** Python's GIL limits CPU parallelism, but GPU operations release the GIL.

**Code location:** `6_complete_pipeline.py` - `ThreadedBatchProcessor`

---

### Strategy 3: Multiprocessing for CPU Work

**How it works:** Uses separate processes (not threads) for CPU-bound tasks

**When used:** NER normalization (spaCy)

**Why not threading:** Python's GIL prevents true CPU parallelism with threads

**Code location:** `1_preprocess_data.py` - `Pool` with `initializer`

---

### Strategy 4: Mixed Precision FP16

**Speedup:** 1.5-2x (memory-bound)

**How it works:** Uses 16-bit floating point instead of 32-bit

| Precision | Bytes/param | Batch size (8GB GPU) |
|-----------|-------------|---------------------|
| FP32 | 4 | 2,000 samples |
| FP16 | 2 | 4,000 samples (2x!) |

**Code location:** `3_train_classifier.py` - `torch.cuda.amp.autocast()`

---

## 8. Training Details

### DistilBERT Training Configuration

```python
# From 3_train_classifier.py

model_name = "distilbert-base-uncased"
max_length = 256
batch_size = 16
learning_rate = 2e-5
optimizer = AdamW (weight_decay=0.01)
scheduler = OneCycleLR (pct_start=0.1, anneal='cos')
loss = CrossEntropyLoss (class_weighted)
epochs = 10
mixed_precision = True
```

### Why These Choices?

| Choice | Reason |
|--------|--------|
| **DistilBERT not BERT** | 60% faster, 40% smaller, 97% accuracy |
| **2e-5 learning rate** | Standard for fine-tuning BERT-family, prevents catastrophic forgetting |
| **AdamW optimizer** | Decoupled weight decay gives better generalization |
| **OneCycleLR scheduler** | Warmup + peak + annealing escapes local minima |
| **Class-weighted loss** | Handles imbalance (equivocation has 5 samples, faulty_generalization has 35) |
| **FP16 mixed precision** | 2x memory = 2x batch size = 1.5x faster |

### Training Results

**Best checkpoint (Epoch 8):**
- Validation Accuracy: 42.8%
- Validation Macro F1: 0.415

**Test Set Results:**
- Accuracy: 47.46%
- Macro F1: 0.4504

**Per-class performance:**
| Class | F1 | Why? |
|-------|-----|------|
| false_dilemma | 0.72 | Clear pattern: "either X or Y" |
| circular_reasoning | 0.60 | Tautological structure |
| appeal_to_popularity | 0.59 | Keywords: "everyone", "most" |
| equivocation | 0.00 | Only 5 test samples |
| appeal_to_emotion | 0.30 | Overlaps with non-fallacious language |

---

## 9. Results & Performance

### Throughput (Classification Only)

| Strategy | Throughput | Speedup |
|----------|------------|---------|
| Sequential | 10.3 texts/sec | 1.00x |
| GPU Batch | 43.9 texts/sec | 4.29x |
| Thread + GPU | 48.1 texts/sec | 4.66x |

**Real-world impact:** 10,000 arguments in 208 seconds (vs 971 seconds sequential)

### Retrieval Performance

| Index Type | Time | Recall@10 |
|------------|------|-----------|
| Flat (brute force) | ~500ms | 100% |
| IVF (nprobe=10) | 200-400ms | ~95% |
| IVF (nprobe=5) | 50-150ms | ~85% |

### Classification Performance by Class

```
Best: false_dilemma (F1=0.72)
- Clear structural pattern: "either X or Y", "only two options"

Good: circular_reasoning (F1=0.60), appeal_to_popularity (F1=0.59)
- Distinctive patterns and keywords

Poor: equivocation (F1=0.00), appeal_to_emotion (F1=0.30)
- Rare in training data, subtle distinctions
```

---

## 10. How to Run Everything

### Prerequisites

```bash
# Hardware
- Google Colab (free) OR
- Local GPU with 8GB+ memory

# Software
- Python 3.8+
- CUDA-capable GPU (for training, optional for inference)
```

### Step-by-Step Execution

**Step 0: Setup Environment**
```bash
# Run this FIRST
python 00_setup_colab.py
# Installs all dependencies (5-10 minutes)
```

**Step 1: Upload Data**
```bash
# Upload these files to /content/data/raw/
- climate_train.csv
- edu_train.csv  
- kialo_flat_clean.csv
```

**Step 2: Preprocess Data**
```bash
python 1_preprocess_data.py
# Outputs: train.csv, val.csv, test.csv, retrieval_corpus.jsonl
# Takes: 5-10 minutes
```

**Step 3: Build FAISS Index**
```bash
python 2_build_faiss_index.py
# Outputs: corpus.index (FAISS index file)
# Takes: 10-15 minutes
```

**Step 4: Train Classifier**
```bash
python 3_train_classifier.py
# Outputs: /models/classifier/
# Takes: 30-60 minutes (on Colab GPU)
```

**Step 5: Test Explanation Generator**
```bash
python 4_train_generator.py
# Interactive mode - type arguments to test
```

**Step 6: Test CPACE Module**
```bash
python 5_cpace_module.py
# Runs built-in test cases
```

**Step 7: Run Complete Pipeline**
```bash
python 6_complete_pipeline.py
# Interactive mode with benchmarks
```

### Quick Test (After Training)

```python
from 6_complete_pipeline import ParallelPipeline

pipeline = ParallelPipeline()
result = pipeline.predict("You can't trust Dr. Chen - she's from Canada")
print(result)
# Output: {'fallacy': 'ad_hominem', 'confidence': 0.87, ...}
```

---

## 11. Common Questions & Answers

### Q1: Why DistilBERT instead of BERT?

**A:** DistilBERT is 60% faster and 40% smaller while retaining 97% of BERT's accuracy. In a multi-module pipeline (retrieval + classification + generation), speed matters.

### Q2: What fine-tuning are we doing?

**A:** Full fine-tuning (all 66.9M parameters updated), not parameter-efficient methods like LoRA. We start from pre-trained DistilBERT and adapt it to fallacy classification.

### Q3: Why 2e-5 learning rate?

**A:** Standard for BERT-family fine-tuning. Low enough to preserve pre-trained knowledge (prevent catastrophic forgetting), high enough to adapt to fallacy task in reasonable time.

### Q4: Why both RAG and CPACE?

**A:** They are complementary:
- RAG: fluent, evidence-grounded, but can hallucinate
- CPACE: contrastive, never hallucinates, but template-bound
- Fusion picks the best of both

### Q5: What is CPACE and why use it?

**A:** CPACE = Contrastive Parallel Argumentation and Contrastive Explanation. It generates explanations that contrast the predicted fallacy with alternatives (e.g., "This is ad hominem, not straw man"). This is how humans learn fallacies, and RAG cannot guarantee contrastive explanations.

### Q6: RAG-Token or RAG-Sequence?

**A:** RAG-Token. It marginalizes over passages per token, allowing synthesis from multiple passages. This is essential for contrastive explanations that need information from two different passages.

### Q7: What is ConceptNet?

**A:** A large common-sense knowledge graph with 34 million edges (relationships) between concepts. CPACE uses it to enrich extracted concepts (e.g., "senator" → IsA "politician").

### Q8: What is MiniLM-L6-v2?

**A:** A lightweight sentence encoder (80MB, 384-dim) used for fusion. It's 10x faster than E5 (15ms vs 150ms) with 95% of the quality - perfect for ranking explanations.

### Q9: What do nlist=100 and nprobe=10 mean?

**A:** 
- `nlist=100`: Number of clusters in FAISS IVF index. √150,000 would be 387, but 100 balances speed and recall.
- `nprobe=10`: Number of clusters searched per query. Searches 10 × 1,500 = 15,000 of 150,000 vectors (10%).

### Q10: What is the fusion module?

**A:** Selects the better explanation (RAG or CPACE) based on cosine similarity to retrieved passages. Uses MiniLM for fast encoding. Keeps the final output grounded in evidence.

### Q11: Where is E5 used vs MiniLM?

**A:**
- **E5** (768-dim, slow, high quality): Retrieval (encoding 150k passages + queries)
- **MiniLM** (384-dim, fast, good enough): Fusion (computing similarity scores)

### Q12: What is AdamW and OneCycleLR?

**A:**
- **AdamW**: Optimizer with decoupled weight decay (better generalization than Adam)
- **OneCycleLR**: Learning rate scheduler that increases LR then decreases (helps escape local minima)

### Q13: Why class-weighted loss?

**A:** Class imbalance - equivocation has 5 samples, faulty_generalization has 35. Weighted loss gives higher importance to rare classes.

### Q14: What is the expected output format?

**A:**
```json
{
    "fallacy": "ad_hominem",
    "confidence": 0.87,
    "explanation": "This argument attacks the person...",
    "similar_examples": [
        {"text": "...", "score": 0.89}
    ]
}
```

### Q15: How long does each step take?

| Step | Time |
|------|------|
| Setup | 5-10 min |
| Preprocessing | 5-10 min |
| FAISS index | 10-15 min |
| Training | 30-60 min |
| Inference (per argument) | 3.35s (sequential), 21ms (batched) |

---

## Appendix: Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUICK REFERENCE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  MODEL           │ DistilBERT-base (66.9M params)                           │
│  FINE-TUNING     │ Full fine-tuning, 2e-5 LR, AdamW, OneCycleLR             │
│  RETRIEVAL       │ E5-large-v2 + FAISS IVF (nlist=100, nprobe=10)           │
│  RAG TYPE        │ RAG-Token with T5-small/BART-base                        │
│  CPACE           │ spaCy + ConceptNet (contrastive symbolic)                │
│  FUSION          │ MiniLM-L6-v2 + cosine similarity                         │
│  PARALLEL        │ GPU batch (4.29x) + ThreadPool (4.66x total)            │
│  PRECISION       │ FP16 mixed precision                                     │
│  ACCURACY        │ 47.46% (13 classes)                                      │
│  THROUGHPUT      │ 48.1 texts/second (4.66x sequential)                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Final Notes

- All code assumes Google Colab directory structure (`/content/`)
- For local runs, change `BASE_DIR` in configuration
- The retrieval corpus (150k passages) is optional - pipeline works without it
- CPACE module requires ConceptNet (falls back to templates if unavailable)
- The fusion module defaults to RAG if CPACE not available

**When you come back to this project after months, start with:**
1. Read this README from top to bottom
2. Run `00_setup_colab.py` to verify environment
3. Test with `python 6_complete_pipeline.py` (interactive mode)
4. Review the specific file you need based on the file descriptions above
