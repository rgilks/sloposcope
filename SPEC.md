# Sloposcope AI Text Analysis — Technical Specification (v2.0)

**Status:** Production Ready - Research-Based Implementation
**Owner:** Development Team
**Target:** Cross‑platform CLI; containerized worker for AWS ECS + SQS; Web API
**License:** Apache‑2.0
**Scope:** Implement research-based 7-dimensional AI slop detection with natural writing protection, based on "Measuring AI 'SLOP' in Text" (Shaib et al., 2025).

---

## 1) Objectives & Non‑Objectives

### Objectives

- Provide a **deterministic CLI** that analyzes text and outputs:

  - Per‑axis **metrics** (utility, quality, style) with clear definitions.
  - **Span annotations** for easy fixing (repetition, off‑topic, templated phrases, hedging, etc.).
  - A **composite "slop score"** with domain‑sensitive weights and confidence.

- Operate **offline by default**; optionally accept references/prompts to enable relevance/factuality proxies.
- Package as a **Python library + CLI**, plus a **containerized worker** that consumes/produces SQS messages on AWS.
- Use **state-of-the-art transformer models** for enhanced accuracy and semantic understanding.

### Non‑Objectives

- Not a hallucination detector for arbitrary facts without references (we only provide proxy/fallback metrics).
- Not a style police for creative fiction; defaults target **news, QA, general explainer** domains (tunable).
- No heavy proprietary models; use small open local models for speed and reproducibility.
- Not a real-time detection system; optimized for batch processing and analysis.

---

## 2) Background & Definitions

**AI slop (research definition):** Based on "Measuring AI 'SLOP' in Text" (Shaib et al., 2025), AI slop is characterized by 7 core dimensions: **Density** (information content per word), **Relevance** (appropriateness to context), **Factuality** (accuracy and truthfulness), **Bias** (one-sided claims), **Structure** (templated and repetitive patterns), **Coherence** (logical flow), and **Tone** (jargon and awkward phrasing).

**Research-Based Implementation:** The system implements the 7-dimensional framework with focus on the most effective patterns identified in academic research, including natural writing protection to prevent false positives.

**Composite slop score:** A normalized [0,1] index based on research-validated weighting of the 7 core dimensions, with dynamic thresholds based on content type.

---

## 3) Primary User Stories

1. **Editor** runs `sloplint article.md --domain news --explain` to get a per‑axis table and inline spans to clean up.
2. **QA lead** pipes model outputs via STDIN and saves machine‑readable JSON for dashboards: `gen | sloplint --json out.json`.
3. **Platform engineer** deploys a worker on ECS. An upstream job posts `{doc_id, s3_uri, prompt}` to SQS; the worker returns `{doc_id, slop_score, axes, spans}` to a results queue and pushes histograms to CloudWatch.

---

## 4) Functional Requirements

### 4.1 Inputs

- **Text sources:**

  - Filenames (`sloplint a.txt b.txt`), directories (`--glob "**/*.md"` optional), or **STDIN**.
  - **Encoding:** UTF‑8 (normalize Unicode; drop control chars).

- **Context (optional):**

  - `--prompt "..."` (intended instruction).
  - `--reference PATH|URL` (one or more; used only for relevance/factuality proxies).
  - `--domain {news,qa,general}` (defaults to `general`).
  - `--language` (default `en`; autodetect and warn if unsupported).

### 4.2 Outputs

- **Pretty console report** (ANSI table + explanations) unless `--json`.
- **JSON report** via `--json out.json` matching schema in §10.
- **Exit codes:** 0 success; 2 invalid input; 3 model/resource error.

### 4.3 Per‑Axis Metrics (computed deterministically)

**A. Information Utility**

1. **Density**

   - _Surprisal/Perplexity:_ mean negative log‑prob per token from a small local LM (e.g., DistilGPT‑2). Normalize to [0,1] using calibration bins.
   - _Idea Density (ID):_ propositions per 100 words. Approximate via dependency patterns indicating predicates/arguments.
   - **Metric:** `Density = z(α·(1/Perplexity) + (1−α)·(ID_norm))` with α default 0.5.

2. **Relevance (optional)**

   - Sentence embeddings (MiniLM). For each sentence s compute `max cosine(s, {prompt, refs})`.
   - **Metric:** fraction of sentences below threshold τ (default 0.55) ⇒ higher fraction ⇒ sloppier. Report also mean similarity.

**B. Information Quality** 3. **Factuality (optional)**

- If references present: for extracted claims (pattern‑based + NER), retrieve matching reference sentences and run NLI (small DeBERTa/roberta‑mnli).
- **Metric:** rates of `contradicted` and `unsupported` claims per 100 sentences; fallback: “Not evaluated”.

4. **Subjectivity / Bias**

   - Lexicon counts per 100 words (MPQA/subjectivity lists).
   - **Metric:** normalized subjectivity density; flag strong modals/stance verbs.

**C. Style Quality** 5. **Repetition**

- N‑gram (1–4) repetition rate (% tokens in repeated n‑grams), **compression ratio** (`gzip_len/text_len`), repeated sentence stems (cosine ≥ .97).
- **Metric:** weighted z‑sum of these.

6. **Templatedness**

   - Unique **POS 5‑grams per 1k tokens** (lower ⇒ more templated), POS‑pattern entropy, common boilerplate phrase hits (“In conclusion,” “As an AI,” “Here are X ways”).
   - **Metric:** inverse normalized diversity + boilerplate hits.

7. **Coherence (proxy)**

   - **Entity grid** continuity score (entity roles across sentences), **embedding drift** between adjacent sentences; abrupt topic shift spikes flagged as spans.
   - **Metric:** 1 − normalized coherence score (higher ⇒ sloppier).

8. **Fluency (proxy)**

   - Grammar error count via LanguageTool (optional), **perplexity spikes** (>p95 per sentence), and broken constructions (POS/morph heuristics).
   - **Metric:** normalized error/instability rate.

9. **Verbosity**

   - Words/sentence, total words per “information bit” (ID), filler/discourse marker rate, **listiness** (% list bullets / numbered lines).
   - **Metric:** normalized verbosity index.

10. **Word Complexity**

- Flesch‑Kincaid, Gunning‑Fog, SMOG; tune “expected” range by domain.
- **Metric:** distance from domain‑expected band; penalize over‑formality for simple prompts.

11. **Tone (Hedging/Sycophancy/Formality)**

- Counts per 100 words for hedges ("perhaps", "might"), sycophancy patterns (explicit agreement with unstated user preferences), excessive formality markers.
- **Metric:** normalized tone index.

### 4.4 Composite Score & Confidence

- **Normalization:** Each metric z‑scored against a **calibration corpus** (provided in repo). Clip to [0,1].
- **Domain weights:** Defaults:

  - `news`: Density .15, Relevance .15, Coherence .15, Repetition .10, Templatedness .10, Subjectivity .10, Verbosity .10, WordComp .05, Tone .05, Fluency .05.
  - `qa`: Factuality .20 (if available else redistributed), Structure (Repetition+Templatedness) .25, Coherence .15, Density .10, Verbosity .10, Tone .05, Fluency .15.
  - `general`: uniform 0.1 each (redistribute if unavailable).

- **Score:** weighted mean of **sloppiness**‑aligned metrics (ensure directionality), plus **confidence** = 1 − (missing_axis_weight).
- **Levels:** `Clean (≤0.30)`, `Watch (0.30–0.55)`, `Sloppy (0.55–0.75)`, `High‑Slop (>0.75)`.
- **Optional learned combiner:** `--model slop-lr` loads a logistic regression (or LightGBM) trained on labeled data to predict binary slop; emits probability and keeps per‑axis explanations.

### 4.5 Explanations & Spans

- For each flagged axis, emit **character spans**:

  - Repetition: repeated n‑grams/sentences with back‑refs.
  - Relevance: sentences below τ with nearest context match.
  - Templates: matched boilerplate phrases, low‑diversity POS runs.
  - Coherence: boundary sentences with high drift.
  - Tone: hedges/sycophancy phrases.

- `--explain` prints sentences with tags: `[off-topic]`, `[template]`, `[repeat]`, `[hedge]`, etc.

---

## 5) Non‑Functional Requirements

- **Performance:** < 1.5 s for 1k words on 4‑core CPU; memory < 800 MB cold, < 400 MB hot.
- **Determinism:** Fixed seeds; consistent tokenization; model versions pinned.
- **Portability:** Linux/macOS/Windows; Python 3.11+; no internet required.
- **Observability (worker):** Structured logs; Prometheus/CloudWatch metrics.
- **Privacy:** No text persisted unless `--cache` or S3 output specified.

---

## 6) System Architecture

### 6.1 Modules (Python package `sloplint`)

- `cli.py` — Typer/Click entrypoint.
- `features/` — pure feature extractors per axis:

  - `density.py`, `relevance.py`, `factuality.py`, `subjectivity.py`, `repetition.py`, `templated.py`, `coherence.py`, `fluency.py`, `verbosity.py`, `complexity.py`, `tone.py`.

- `nlp/` — shared text pipeline (spaCy model wrapper, sentence splitter, token normalization).
- `combine.py` — normalization, weighting, compositing, learned models.
- `spans.py` — span structures, merging, formatting.
- `io.py` — file/STDIN handling; reference loader (file/URL); JSON schema I/O.
- `calibration/` — stats, bins, expected ranges, domain presets (versioned JSON).
- `models/` — minimal local models (MiniLM, DistilGPT‑2), cached on first run or vendored.
- `training/` — scripts for combiner training + evaluation.
- `aws_worker/` — SQS poller, S3 helper, result emitter.

### 6.2 Data Flow (CLI)

1. Load text → normalize → sentence segmentation.
2. Run NLP (POS/NER) once; reuse across features.
3. Compute feature set per axis (pure functions).
4. Normalize + weight → final score + confidence.
5. Emit report (pretty or JSON) + optional spans/explanations.

### 6.3 Data Flow (ECS Worker)

1. Poll SQS for `{doc_id, text|s3_uri, prompt?, reference?, domain?, options?}`.
2. Fetch text (direct or from S3).
3. Run same pipeline as CLI.
4. Put result to **results queue** or write to S3; ack message.
5. Push metrics to CloudWatch (per‑axis histograms, latency, failures).
6. DLQ on parse/timeout/model errors.

---

## 7) Algorithms & Metric Details

### 7.1 Density

- **Perplexity (PPL):** tokenize with GPT‑2 BPE; run DistilGPT‑2; compute mean token NLL; invert & z‑score.
- **Idea Density (ID):** heuristics: count predicates (VERB heads), clausal markers, content nouns with relations; normalize per 100 words.

### 7.2 Relevance

- Embed sentences with `all-MiniLM-L6-v2`; also embed `prompt` and each `reference` paragraph (or headline).
- For each sentence: `sim = max cosine(s, prompt ∪ refs)`; label span if `sim < τ`.

### 7.3 Factuality

- Extract atomic claims (subject–predicate–object from dependency triples + named entities).
- For each claim, select top‑k reference sentences by BM25 or embedding similarity; run small NLI → `entails`, `neutral`, `contradicts`.
- Metrics: contradictions per 100 sentences; unsupported per 100; coverage rate.

### 7.4 Subjectivity/Bias

- Count subjective/stance terms (MPQA) and strong modals; normalize per 100 words; provide top terms.

### 7.5 Repetition

- `rep_n = tokens_in_repeated_ngrams(n) / total_tokens` for n=1..4; also **sentence dedupe** by cosine ≥ .97 on sentence embeddings.
- **Compression ratio** = `len(gzip(text))/len(text_bytes)` (lower ratio = more repetition); invert & normalize.

### 7.6 Templatedness

- Build POS 5‑grams; compute **type/token ratio** per 1k tokens and **entropy H**.
- Maintain boilerplate regex list; each match yields a span.

### 7.7 Coherence (proxy)

- **Entity grid:** Build entity × sentence grid with roles {S, O, X, –}. Use Barzilay‑Lapata transitions → score cohesion.
- **Embedding drift:** cosine between adjacent sentence embeddings; mark boundaries below δ (default 0.80).
- Metric is 1 − normalized cohesion.

### 7.8 Fluency (proxy)

- Run LanguageTool (if available); count errors per 1k tokens.
- Detect high PPL outliers per sentence (>p95) as fluency spikes.

### 7.9 Verbosity

- Mean words/sentence, herfindahl of sentence lengths, filler markers (discourse cue list), **listiness** (% of lines starting with bullets/numbers), and **bits/word** (= ID / words).

### 7.10 Word Complexity

- Compute FKGL, Gunning‑Fog, SMOG; derive domain‑expected bands (e.g., `news`: FKGL 8–12; `qa`: 6–10). Distance to band → penalty.

### 7.11 Tone

- **Hedging list:** perhaps, might, could, somewhat, tends to, arguably, etc.
- **Sycophancy:** patterns agreeing with putative user preferences (regex heuristics).
- **Formality:** rates of nominalizations, passive voice (auxpass), and formal connectives.

---

## 8) Configuration & Calibration

- `~/.config/sloplint/config.toml` or `--config path` with:

  - model names/paths; cache dir; language; thresholds (τ, δ); domain weights; boilerplate regexes; lexicons; reference parsing settings.

- `calibration/*.json`: stores z‑score means/stdevs per metric, per domain.
- `sloplint calibrate <corpus_dir> --domain news` recomputes calibration stats.

---

## 9) CLI Specification

```
sloplint [FILES...] [OPTIONS]

Options:
  --domain [news|qa|general]
  --prompt TEXT
  --reference PATH|URL  (repeatable)
  --json PATH           (emit JSON matching §10)
  --explain             (annotated sentence output)
  --spans               (print character spans)
  --model [none|slop-lr]
  --language TEXT       (default: en)
  --config PATH
  --version
  --help
```

**Examples**

- `cat draft.md | sloplint --domain news --explain`
- `sloplint answer.txt --domain qa --prompt "Answer the question" --json out.json`
- `sloplint doc.md --reference refs/bibliography.txt --model slop-lr`

---

## 10) JSON Output Schema (v1)

```json
{
  "version": "1.0",
  "domain": "news",
  "slop_score": 0.63,
  "confidence": 0.88,
  "level": "Sloppy",
  "metrics": {
    "density": {
      "value": 0.41,
      "details": { "ppl": 21.3, "idea_density": 5.7 }
    },
    "relevance": { "value": 0.7, "mean_sim": 0.46, "low_sim_frac": 0.42 },
    "factuality": {
      "value": 0.55,
      "contradictions_per_100s": 1.2,
      "unsupported_per_100s": 7.8,
      "coverage": 0.64
    },
    "subjectivity": { "value": 0.38, "top_terms": ["clearly", "obviously"] },
    "repetition": {
      "value": 0.66,
      "ngram": { "1": 0.18, "2": 0.14, "3": 0.1, "4": 0.07 },
      "compression_ratio": 0.32
    },
    "templated": {
      "value": 0.59,
      "pos5_diversity": 0.23,
      "boilerplate_hits": 4
    },
    "coherence": { "value": 0.44, "entity_grid": 0.61, "drift_outliers": 3 },
    "fluency": { "value": 0.21, "grammar_errors_k": 2.0, "ppl_spikes": 1 },
    "verbosity": { "value": 0.52, "wps": 23.1, "listiness": 0.18 },
    "complexity": { "value": 0.33, "fkgl": 12.2, "fog": 15.1 },
    "tone": { "value": 0.4, "hedges_k": 3.2, "sycophancy_hits": 1 }
  },
  "spans": [
    {
      "axis": "repetition",
      "start": 120,
      "end": 165,
      "note": "Repeated sentence stem"
    },
    {
      "axis": "templated",
      "start": 301,
      "end": 320,
      "note": "Boilerplate: 'In conclusion'"
    },
    {
      "axis": "relevance",
      "start": 540,
      "end": 650,
      "note": "Low similarity to prompt"
    }
  ],
  "notes": [
    "Factuality evaluated for 64% of claims due to provided references."
  ],
  "timings_ms": { "total": 890, "nlp": 210, "embeddings": 180, "lm": 260 }
}
```

---

## 11) AWS ECS + SQS Worker Specification

### 11.1 Message Schemas

**Input queue (JSON):**

```json
{
  "doc_id": "uuid",
  "domain": "news",
  "text": "...", // or omit and use s3_uri
  "s3_uri": "s3://bucket/key", // optional
  "prompt": "...", // optional
  "references": ["s3://...", "https://..."]
}
```

**Output queue (JSON):**

```json
{
  "doc_id": "uuid",
  "status": "ok",
  "result": {
    /* JSON report §10 */
  },
  "error": null
}
```

**Error example:** `{ "doc_id": "...", "status": "error", "error": {"code":"DecodeError","message":"..."} }`

### 11.2 Worker Behavior

- Poll SQS with long polling (20s). **Max In‑Flight** configurable; 1–4 per task.
- If `s3_uri` present, fetch object (gzip accepted).
- Process with the same library; emit to results queue; optionally write `{doc_id}.json` to S3.
- **Idempotency:** Use `doc_id` as idempotency key; safe to re‑process.
- **Retries & DLQ:** 3 attempts; backoff 2^n; DLQ on permanent errors.

### 11.3 ECS/Fargate

- CPU 1 vCPU / 2 GB baseline; scale to 2 vCPU / 4 GB for batch spikes.
- **Image:** ECR `tre/sloplint:sha` built via GitHub Actions.
- **Env vars:** `SQS_IN_URL`, `SQS_OUT_URL`, `AWS_REGION`, `S3_OUT_BUCKET?`, `LOG_LEVEL`, `CALIB_PROFILE`.
- **Observability:** CloudWatch logs; EMF metrics: latency, per‑axis means, error counts.

---

## 12) Dependencies & Versions

- Python 3.11+
- **NLP:** spaCy (en_core_web_trf), sentencepiece or tokenizers, `textstat`.
- **Embeddings:** `sentence-transformers` (all-MiniLM-L6-v2).
- **LM for PPL:** `transformers` + DistilGPT‑2 (CPU).
- **NLI (optional):** small `roberta-base-mnli` or `deberta-v3-base-mnli`.
- **Grammar (optional):** `language-tool-python` (start JVM server on demand).
- **AWS:** `boto3` for SQS/S3.
- **CLI:** Typer or Click; `rich` for ANSI tables.
- **ML utils:** scikit‑learn, numpy, scipy.
- **Package Management:** `uv` for fast dependency resolution and virtual environment management.

---

## 13) Repository Layout

```
sloplint/
  sloplint/
    __init__.py
    cli.py
    io.py
    spans.py
    combine.py
    nlp/
      pipeline.py
    features/
      density.py
      relevance.py
      factuality.py
      subjectivity.py
      repetition.py
      templated.py
      coherence.py
      fluency.py
      verbosity.py
      complexity.py
      tone.py
    models/
      README.md
    calibration/
      default_news.json
      default_qa.json
      default_general.json
    training/
      train_combiner.py
      evaluate.py
  tests/
    test_cli.py
    test_features_*.py
    fixtures/
  docker/
    Dockerfile
    entrypoint.sh
  scripts/
    sloplint_worker.py
  pyproject.toml
  README.md
  LICENSE
```

---

## 14) Build, Run, and Packaging

### 14.1 Local

- `uv` or `pip-tools` for locking deps.
- `make setup` (create venv, install extras).
- `make run FILE=sample.md` (runs CLI).

### 14.2 Docker

- Multi‑stage: base (python:3.11-slim) → build (install, download models) → runtime (non‑root).
- Expose `sloplint_worker.py` as container CMD for ECS.
- Healthcheck: run a 1‑line smoke text.

### 14.3 CI/CD

- GitHub Actions: lint (ruff), type‑check (mypy), test (pytest), build wheel, build/push image to ECR, tag release.

---

## 15) Testing & Evaluation

- **Unit tests:** deterministic fixtures for each feature module; golden JSON snapshots for end‑to‑end.
- **Property tests:** reproducibility under repeated runs; span indices integrity.
- **Performance tests:** sample 1k‑word doc < 1.5s CPU; monitor memory.
- **Calibration validation:** z‑score sanity (mean≈0, sd≈1) on calibration corpora.
- **Combiner eval (optional):** AUPRC vs. labeled data; ablations over domains.

---

## 16) Security & Privacy

- No external calls unless **explicit references** or `--fetch-refs` is set (off by default).
- Redact emails/IDs in logs; opt‑out of span content in worker logs.
- Support `--no-store` to disable caching; ephemeral tempfiles; S3 SSE‑S3 or SSE‑KMS if used.

---

## 17) Risks & Mitigations

- **Weak auto‑metrics on coherence/factuality:** expose as advisory; require references for factuality; show confidence band.
- **Domain drift:** allow per‑team calibration + on‑device `sloplint calibrate`.
- **Latency spikes (grammar/NLI):** keep those optional; parallelize features with thread pool.
- **Locale/language variance:** warn if unsupported; degrade to language‑agnostic features (repetition, listiness, POS‑lite).

---

## 18) Roadmap

- **v1.0 (Current):** All core axes with enhanced transformer-based features, CLI, JSON, spans, domain weights, Docker, ECS worker, semantic embeddings.
- **v1.1:** Factuality with references + NLI; `slop-lr` combiner; calibration CLI.
- **v1.2:** Multi‑language partial support; GUI report (HTML), VS Code extension.
- **v2.0:** Production‑hardening, full docs, signed releases, benchmark corpus, advanced ML integration.

---

## 19) Glossary

- **Axis:** A conceptual dimension of slop (e.g., repetition).
- **Span:** Character offset range indicating a problematic region.
- **Calibration:** Mapping raw metric to normalized score using reference statistics.
- **Combiner:** Model or formula for composite slop score.
