> Eduard-Stefanut Iordache, April 2026
# Veridion Search Engine: Intent Qualification Pipeline

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [System Architecture (The 4-Phase Pipeline)](#2-system-architecture-the-4-phase-pipeline)
3. [Key Engineering Decisions & Trade-offs](#3-key-engineering-decisions--trade-offs)
4. [Error Analysis & Edge Case Handling](#4-error-analysis--edge-case-handling)
5. [Scalability & Production Readiness](#5-scalability--production-readiness)
6. [Usage & Reproducibility](#6-usage--reproducibility)
---

## 1. Executive Summary

The goal of this project was to build a robust, scalable pipeline for a search
engine for parsing complex, natural-language business queries, and accurately
returning the companies matching the user's intent from a dataset.

To achieve a balance between execution speed, computational cost, and
high-accuracy intent matching, I designed a  **Cascaded Retrieval-Augmented
Generation (RAG) Pipeline**. Rather than relying solely on slow, token-heavy
LLM evaluations or rigid keyword matching, this system progressively funnels
the dataset through 4 distinct phases:
1. **LLM Schema Extraction**
2. **Deterministic Data Filtering**
3. **Dense Vector Semantic Search**
4. **Concurrent LLM Intent Validation**

This architecture ensures that obvious mismatches are dropped instantly with
$O(1)$ operations, while more complex qualifications are handled by language
models.

---

## 2. System Architecture (The 4-Phase Pipeline)

The system is orchestrated by the `SearchEngine` class, which redirects the data
through 4 phases:

### Phase 1: Query Parsing (`QueryParser`)
The user's raw natural language query is passed to an LLM, configured for strict
JSON output, to extract specific constraints.
* **Hard Filters:** Numerical or geographic constraints, that are hard to
interpret (e.g., `min_revenue`, `max_employees`, `location`, `exclude_location`).
* **Soft Filters:** Semantic context (e.g., `industry_or_vertical`,
`business_model`).

The LLM is prompted with strict deterministic rules, such as mapping natural
language countries to ISO codes and evaluating negative constraints ("Not from
Spain" -> `"exclude_location": "es"`). This ensures edge-cases are handled
correctly, and the LLM is instructed to default to `NULL` for anything
unconfident.

After the extraction of the JSON, I clean it of any artifacts the LLM generated
(using the `clean_json()` method), and sanitize it (using `sanitize_json()`),
checking for logic errors and correct data types of each field.

### Phase 2: Deterministic Filtering (`CompaniesFilter`)
Using `pandas`, the system applies the extracted `hard_filters` to the dataset.
This phase is designed to be highly defensive against messy, real-world data:
* **Missing Data Tolerance:** Missing values (`NaN`) are safely retained using
bitwise logic to ensure potential candidates aren't dropped due to incomplete
profiles.
* **Geographic Rationalism:** Location filtering inspects nested dictionary
values (like `country_code` and `town`) to ensure exact matches, preventing
false positives from substrings.

### Phase 3: Semantic Search (`Searcher`)
The companies that pass through the hard filters are ranked by their semantic
relevance to the user's raw query.
* Using the `sentence-transformers` library (`all-MiniLM-L6-v2`), the system
calculates the **Cosine Similarity** between the embedded user query and the
company profiles.
* To achieve as low as possible latency, the system utilizes **pre-computed
embeddings**. It loads a `.npy` matrix into memory and uses Pandas index
mapping to slice out only the vectors of the remaining valid companies. If
the matrix is not existent in the directory, then it just calculates the
embeddings directly, only for the required companies.

### Phase 4: Intent Validation (`IntentValidator`)
Because a simple cosine similarity ranking can misinterpret the intent of the
user or the purpose of a company by keyword density (e.g., a packaging company
using the word "cosmetics" heavily), the first k semantic matches are sent back
to the LLM for a final evaluation.
* The LLM evaluates the company and returns a JSON containing a status (MATCH
or REJECTED), a confidence level (high, medium, low), and a brief reason
justifying its decision.
* To bypass Python I/O bottlenecks, this phase utilizes `concurrent.futures.
ThreadPoolExecutor` to process all companies simultaneously in parallel.

---

## 3. Key Engineering Decisions & Trade-offs

The most important part of building a production-ready pipeline is balancing
speed, cost, and accuracy. Here are the specific trade-offs and decisions I made
to optimize the system:

### Feature Selection for Embeddings
By throwing every available piece of information into the embedding model,
I would have diluted the semantic density of each company's profile, and
that would result in less accurate cosine similarity search. I intentionally
omitted low-signal metadata fields, like exact coordinates, phone numbers,
and social media profiles. By concatenating only the core fields (`description`,
`core_offerings`, and `primary_naics`), I optimized both accuracy (**signal-to
noise ratio**), and the overall token-precessing speed.

### Pre-computed Caching vs. Live Embeddings
Embedding hundreds of text strings live during Phase 3 takes several seconds,
affecting the overall speed of the project. While acceptable for a prototype,
this doesn't scale.
* **The Decision:** I wrote a standalone `precompute.py` script which needs to
be executed separately to embed the entire dataset and save it on the disk as a
Numpy matrix (`.npy`).
* **The Trade-off:** Cached embeddings are incredibly fast but risk becoming
obsolete if the database changes. To mitigate this, I built the `Searcher` class
to be defensive: it uses Pandas index-mapping to instantly slice the
pre-computed matrix if the file exists, but falls back to dynamic live-embedding
if the `.npy` file is missing.

### Raw Query Embedding vs. Soft Filter Weighting
In Phase 1, the LLM extracts `soft_filters` (like industry or business model).
Initially, I considered using these to artificially boost the cosine similarity
scores in Phase 3 (e.g., adding +0.1 to the score if the industry matched).
* **The Decision:** I chose not to apply arbitrary scalar weight boosts, because
doing so without a labeled tuning dataset introduces a magic number bias, where
a terrible company with one matching keyword artificially outranks a
great company with high semantic similarity.
* **The Alternative:** Instead, I embed the user's raw query to capture the pure
semantic essence and leave the strict qualification of those soft constraints to
the rigorous Phase 4 LLM Intent Validation.

### The Inference Toggle
A major bottleneck I encountered was during the Phase 4 concurrent validation.
Sending 10+ simultaneous requests to a local LLM (via Ollama) caused the local
hardware to serialize the requests to prevent out-of-memory crashes, resulting
in execution times of over 4 minutes.
* **The Solution:** I implemented a `--mode` CLI argument with an underlying
`BaseLLM` parent class.
* **The Trade-off:** Users can run `--mode local` to ensure 100% data privacy
and zero API costs, while sacrifing speed, or they can run `--mode cloud` to
use Groq's high-speed API. Cloud mode executes the asynchronous Phase 4
validation in a true parallel fashion, dropping the 4-minute wait time down to
roughly 3 seconds. This in turn uses high costs for the API and token limits,
if there is need for a better model to be used.

---

## 4. Error Analysis & Edge Case Handling

### Negative Location Bug
* **The Problem:** LLMs can struggle with negative constraints. When parsing
queries like "Companies not from Spain", smaller models would hallucinate
invalid JSON structures, like `{"!ES": null}`, or mistakenly assign "Spain" to
the positive location filter. Furthermore, using a simple Pandas substring
filter like `~df['address'].str.contains("br")` (for Brazil) would accidentally
drop companies that contain "br" in their address.
* **The Fix:** I made the JSON to include an explicit `exclude_location` key
and instructed the LLM to map all countries to their strict 2-letter ISO codes.
In the Pandas filter, I implemented a dictionary check (`address.get
('country_code') == target`) rather than a blind substring search, making false
positives mathematically impossible.

### Broad Geographic Regions
* **The Problem:** I designed the hard filters for deterministic, exact matches,
but when the system was faced with the query: "Renewable energy companies in
Scandinavia", the LLM extracted `"location": "scandinavia"`. The Pandas filter
then dropped the entire dataset because company addresses contain specific
countries, not broader regions.
* **The Fix:** I updated the LLM system prompt: if a user mentions a specific
country, it maps to the ISO code, and if they mention a continent or region
(e.g., "Europe", "Scandinavia"), the LLM outputs `null` for the location.
This safely bypasses the strict Pandas filter and allows semantic search to
naturally rank the relevant companies based on vector similarity.

### Semantic Density Trap
* **The Problem:** When querying "Cosmetics companies", the Phase 3 semantic
search returned several cosmetics packaging companies. Smaller embedding models
are manipulated by keyword frequency. If a packaging company says the word
"cosmetics" a lot of times in its description, its vector similarity
artificially spikes.
* **The Fix:** I engineered the `IntentValidator` system prompt to aggressively
check supply chain boundaries. The LLM is instructed to differentiate between a
company's core product and the industry it serves.

### API Token Limits & Context Window Crashes
* **The Problem:** During automated testing, the cloud API threw `400 Bad
Request` errors due to token limits. This happened because a few companiesin the
dataset had big text blocks in their `description` field, causing the LLM to run
out of output tokens before it could finish.
* **The Fix:** I implemented safe string truncation, slicing company
descriptions to a maximum of 2,000 characters before sending them to the Intent
Validator, ensuring the prompt doesn't exceed context window limits.

### Dirty Data & Redundant Compute
* **The Problem:** I noticed in my logs that the system was evaluating identical
companies (like "windainergy") multiple times per query, wasting both API tokens
and compute time.
* **The Fix:** I added a `.drop_duplicates(subset=['operational_name'])` step
during the `SearchEngine` initialization to instantly clean the dataset before
any processing occurs.

### Current System Limitations & Misclassifications
While the pipeline handles explicit constraints well, it currently struggles in
two specific areas:

**1. Ambiguous User Intent vs. LLM Strictness**
When testing the query *"Public software companies with more than 1000
employees"*, the Phase 4 `IntentValidator` rejected **EPAM** and **Capgemini**.
The LLM's reasoning was: *"They are IT consulting/service providers, not
software product companies."

Technically, the LLM is correct. They are service firms, not SaaS product
vendors, but in human business contexts, many users querying "software
companies" may want to see enterprise software engineering firms like
**EPAM**. Because the LLM acts as a strict, literal judge, it can create false
negatives when the user's natural language intent is highly subjective or broad.

**2. The "Thin Profile" Penalty**
The Semantic Search relies entirely on the density of the specific fields
concatenated together, like the `description` and `core_offerings` fields.

If a massive, highly relevant company only has a one-sentence description in the
database (e.g., "We build things."), its dense vector will contain very little
semantic meaning and can be outranked by a much smaller, less relevant company
that has a description rich in industry keywords (SEO-optimized).

---

## 5. Scalability & Production Readiness

The current architecture is optimized for a local prototype, running entirely
in-memory using Python data structures (Pandas and Numpy). To scale this
pipeline to handle Veridion's production volume of 100+ million companies and
high-concurrency API traffic, some architectural upgrades would be required.

### 1. Moving from RAM to a Database
Loading a 100M-row JSONL file into a Pandas DataFrame would result in an
Out-Of-Memory (OOM) crash on standard servers.
* **The Upgrade:** Phase 2 (Deterministic Filtering) should move out of Python
memory and into a structured database (like PostgreSQL or MongoDB). By indexing
fields like `min_revenue` and `location`, the database can perform the
hard-filtering step in a much faster time. Phase 1 would be updated so the LLM
outputs a structured database query (like SQL) rather than a JSON object for
Pandas.

### 2. Replacing Numpy with a Vector Database
Currently, Phase 3 performs an exact K-Nearest Neighbors (KNN) search by
computing the cosine similarity between the query and every single pre-computed
vector in the `.npy` matrix. This is an $O(N)$ operation that becomes too slow
with scaling.
* **The Upgrade:** The pre-computed embeddings would be migrated to a dedicated
Vector Database (such as pgvector or Pinecone). Instead of comparing the query
against every row, vector databases use specialized indexing to cluster similar
data points together, allowing the system to retrieve the closest matches almost
instantly.

### 3. Asynchronous Microservices & Queueing
Currently, Phase 4 uses `ThreadPoolExecutor` to handle concurrent LLM calls.
While effective, it relies on a hardcoded `sleep()` function in the
`run_queries.sh` script to avoid API rate limits, which is not viable in
production.
* **The Upgrade:** The system would be split into an API backend and a
background worker queue (using a tool like Celery or Redis). Instead of making
the user wait synchronously, validation tasks would be pushed to the queue. If
the LLM API throws a "Rate Limit" error, the queue can safely pause and retry
the request later using Exponential Backoff without crashing the main
application.

---

## 6. Usage & Reproducibility

### Prerequisites
Ensure you have Python installed, then install the required dependencies:
```
pip install -r requirements.txt
```

### Pre-computation:
Before running the search engine, you can generate the local vector cache. This
script reads `companies.jsonl`, calculates the dense vectors using
`sentence-transformers`, and outputs the `companies_embeddings.npy` file used for
fast semantic retrieval.
```
python3 precompute.py
```

### CLI Usage:
You can query the engine directly from the command line using `solution.py`.

#### Available CLI Parameters:
* **`--query`** *(Required, String)*: The natural language search query to be
evaluated (e.g., `"Logistics companies in Romania"`). Must be wrapped in quotes.
* **`--mode`** *(Optional, String)*: The environment to use.
    * `local` (Default): Uses Ollama for 100% private, local inference.
    * `cloud`: Uses the Groq API for high-speed, parallel processing.
* **`--model`** *(Optional, String)*: The specific LLM model to route the
request to.
    * Default is `llama3` for local mode.
    * For cloud mode, `llama-3.3-70b-versatile` or
    `meta-llama/llama-4-scout-17b-16e-instruct` are recommended.
* **`--top_k`** *(Optional, Integer)*: The number of top semantic matches from
Phase 3 to pass into Phase 4 for LLM intent validation. Default is 10.

#### Execution Examples:
**Example 1: Cloud Mode (Recommended for high-speed evaluation)**
```
python solution.py --query "Logistics companies in Romania" --mode cloud --model meta-llama/llama-4-scout-17b-16e-instruct --top_k 15
```
**Example 2: Local Mode (Requires Ollama running locally)**
```
python solution.py --query "Logistics companies in Romania" --mode local --model llama3 --top_k 10
```

### Automated Testing:
To test the system against the 12 example queries, execute the included bash
script. This script automatically handles Cloud API rate limits
with a hardcoded `sleep 30` command and outputs a clean log of the results in
the `results.txt` file. The bash script takes as input all the queries in the
`queries.txt` file, line by line.
```
chmod +x run_queries.sh
./run_queries.sh
```
