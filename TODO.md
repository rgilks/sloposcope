# AI Slop CLI - Implementation TODO

**Project:** AI Slop CLI - Cross-platform CLI for detecting "AI slop" in text  
**Status:** Planning Phase  
**Target:** v0.1 MVP Implementation  
**Based on:** SPEC.md + "Measuring AI Slop in Text" (arXiv:2509.19163v1)

---

## 🎯 Project Overview

This project implements a deterministic CLI tool that analyzes text and outputs per-axis metrics (utility, quality, style) with clear definitions, span annotations for easy fixing, and a composite "slop score" with domain-sensitive weights and confidence.

### Key Features

- **Per-axis metrics**: Information Utility, Information Quality, Style Quality
- **Span annotations**: Character-level spans for problematic regions
- **Composite scoring**: Domain-sensitive weighted combination
- **Multiple interfaces**: CLI, Python library, AWS ECS worker
- **Offline operation**: No internet required by default

---

## 📋 Implementation Phases

### Phase 1: Project Setup & Foundation

- [ ] **1.1** Initialize Python project structure

  - [ ] Create `pyproject.toml` with dependencies
  - [ ] Set up package structure (`sloplint/` directory)
  - [ ] Configure development environment (venv, uv/pip-tools)
  - [ ] Add pre-commit hooks (ruff, mypy, black)

- [ ] **1.2** Repository structure setup

  - [ ] Create directory structure per SPEC §13
  - [ ] Add `LICENSE` (Apache-2.0)
  - [ ] Create `README.md` with basic usage
  - [ ] Set up `.gitignore` and `.dockerignore`

- [ ] **1.3** Development tooling
  - [ ] Configure `Makefile` with common commands
  - [ ] Set up pytest configuration
  - [ ] Add type checking with mypy
  - [ ] Configure linting with ruff

### Phase 2: Core Infrastructure

- [ ] **2.1** Base package structure

  - [ ] Implement `sloplint/__init__.py`
  - [ ] Create `sloplint/io.py` for file/STDIN handling
  - [ ] Implement `sloplint/spans.py` for span structures
  - [ ] Create `sloplint/combine.py` for score combination

- [ ] **2.2** NLP pipeline foundation

  - [ ] Implement `sloplint/nlp/pipeline.py`
  - [ ] Add spaCy model wrapper (`en_core_web_md`)
  - [ ] Create sentence splitter and token normalization
  - [ ] Add text preprocessing utilities

- [ ] **2.3** Configuration system
  - [ ] Implement config file handling (`~/.config/sloplint/config.toml`)
  - [ ] Add domain-specific presets
  - [ ] Create calibration data structure
  - [ ] Implement threshold management

### Phase 3: Feature Extractors (Per-Axis Metrics)

#### 3.1 Information Utility

- [ ] **3.1.1** Density metrics (`sloplint/features/density.py`)

  - [ ] Implement perplexity calculation with DistilGPT-2
  - [ ] Add idea density heuristics (propositions per 100 words)
  - [ ] Create density combination formula
  - [ ] Add calibration and normalization

- [ ] **3.1.2** Relevance metrics (`sloplint/features/relevance.py`)
  - [ ] Implement sentence embeddings with MiniLM
  - [ ] Add cosine similarity calculations
  - [ ] Create relevance scoring with thresholds
  - [ ] Add span annotation for low-relevance sentences

#### 3.2 Information Quality

- [ ] **3.2.1** Factuality metrics (`sloplint/features/factuality.py`)

  - [ ] Implement claim extraction (dependency triples + NER)
  - [ ] Add NLI model integration (DeBERTa/roberta-mnli)
  - [ ] Create factuality scoring with references
  - [ ] Add contradiction/unsupported claim detection

- [ ] **3.2.2** Subjectivity/Bias (`sloplint/features/subjectivity.py`)
  - [ ] Implement MPQA lexicon integration
  - [ ] Add subjectivity density calculation
  - [ ] Create bias detection heuristics
  - [ ] Add modal/stance verb detection

#### 3.3 Style Quality

- [ ] **3.3.1** Repetition detection (`sloplint/features/repetition.py`)

  - [ ] Implement n-gram repetition analysis (1-4 grams)
  - [ ] Add compression ratio calculation
  - [ ] Create sentence deduplication (cosine similarity)
  - [ ] Add repetition span annotation

- [ ] **3.3.2** Templatedness (`sloplint/features/templated.py`)

  - [ ] Implement POS 5-gram diversity analysis
  - [ ] Add boilerplate phrase detection
  - [ ] Create templatedness scoring
  - [ ] Add template span annotation

- [ ] **3.3.3** Coherence proxy (`sloplint/features/coherence.py`)

  - [ ] Implement entity grid continuity scoring
  - [ ] Add embedding drift detection
  - [ ] Create coherence scoring
  - [ ] Add topic shift span detection

- [ ] **3.3.4** Fluency proxy (`sloplint/features/fluency.py`)

  - [ ] Integrate LanguageTool for grammar checking
  - [ ] Add perplexity spike detection
  - [ ] Create fluency scoring
  - [ ] Add broken construction detection

- [ ] **3.3.5** Verbosity (`sloplint/features/verbosity.py`)

  - [ ] Implement words-per-sentence analysis
  - [ ] Add filler/discourse marker detection
  - [ ] Create listiness calculation
  - [ ] Add verbosity scoring

- [ ] **3.3.6** Word complexity (`sloplint/features/complexity.py`)

  - [ ] Implement readability metrics (Flesch-Kincaid, Gunning-Fog, SMOG)
  - [ ] Add domain-expected range calculation
  - [ ] Create complexity scoring
  - [ ] Add domain-specific tuning

- [ ] **3.3.7** Tone analysis (`sloplint/features/tone.py`)
  - [ ] Implement hedging detection
  - [ ] Add sycophancy pattern detection
  - [ ] Create formality analysis
  - [ ] Add tone scoring

### Phase 4: Score Combination & Calibration

- [ ] **4.1** Score normalization

  - [ ] Implement z-score normalization
  - [ ] Add calibration corpus integration
  - [ ] Create domain-specific calibration
  - [ ] Add confidence band calculation

- [ ] **4.2** Composite scoring

  - [ ] Implement domain-specific weighting
  - [ ] Add composite slop score calculation
  - [ ] Create confidence scoring
  - [ ] Add slop level classification

- [ ] **4.3** Calibration system
  - [ ] Create calibration data files
  - [ ] Implement `sloplint calibrate` command
  - [ ] Add calibration validation
  - [ ] Create domain preset management

### Phase 5: CLI Interface

- [ ] **5.1** CLI implementation (`sloplint/cli.py`)

  - [ ] Implement Typer/Click CLI framework
  - [ ] Add command-line argument parsing
  - [ ] Create help system and documentation
  - [ ] Add version and configuration management

- [ ] **5.2** Output formatting

  - [ ] Implement pretty console reports (ANSI tables)
  - [ ] Add JSON output formatting
  - [ ] Create span annotation display
  - [ ] Add explanation mode (`--explain`)

- [ ] **5.3** Input handling
  - [ ] Add file input support
  - [ ] Implement STDIN processing
  - [ ] Add directory/glob pattern support
  - [ ] Create reference loading (file/URL)

### Phase 6: AWS Worker Implementation

- [ ] **6.1** Worker infrastructure (`sloplint/aws_worker/`)

  - [ ] Implement SQS message polling
  - [ ] Add S3 integration for text fetching
  - [ ] Create result queue publishing
  - [ ] Add CloudWatch metrics integration

- [ ] **6.2** Worker logic (`scripts/sloplint_worker.py`)

  - [ ] Implement message processing pipeline
  - [ ] Add error handling and DLQ support
  - [ ] Create idempotency handling
  - [ ] Add retry logic with backoff

- [ ] **6.3** Containerization
  - [ ] Create multi-stage Dockerfile
  - [ ] Add health check implementation
  - [ ] Create ECS task definition
  - [ ] Add container optimization

### Phase 7: Testing & Quality Assurance

- [ ] **7.1** Unit testing

  - [ ] Create test fixtures for each feature module
  - [ ] Add deterministic test cases
  - [ ] Implement golden JSON snapshots
  - [ ] Add property-based testing

- [ ] **7.2** Integration testing

  - [ ] Create end-to-end CLI tests
  - [ ] Add AWS worker integration tests
  - [ ] Implement performance benchmarks
  - [ ] Add calibration validation tests

- [ ] **7.3** Quality metrics
  - [ ] Add code coverage reporting
  - [ ] Implement performance monitoring
  - [ ] Create reproducibility tests
  - [ ] Add memory usage validation

### Phase 8: Documentation & Examples

- [ ] **8.1** User documentation

  - [ ] Create comprehensive README
  - [ ] Add CLI usage examples
  - [ ] Create API documentation
  - [ ] Add troubleshooting guide

- [ ] **8.2** Developer documentation

  - [ ] Document architecture decisions
  - [ ] Add contribution guidelines
  - [ ] Create development setup guide
  - [ ] Add testing documentation

- [ ] **8.3** Examples and tutorials
  - [ ] Create sample text files
  - [ ] Add domain-specific examples
  - [ ] Create AWS deployment guide
  - [ ] Add integration examples

### Phase 9: CI/CD & Deployment

- [ ] **9.1** GitHub Actions setup

  - [ ] Configure linting (ruff)
  - [ ] Add type checking (mypy)
  - [ ] Implement testing (pytest)
  - [ ] Add wheel building

- [ ] **9.2** Container registry

  - [ ] Set up ECR repository
  - [ ] Configure Docker image building
  - [ ] Add image pushing to ECR
  - [ ] Implement release tagging

- [ ] **9.3** Release management
  - [ ] Add semantic versioning
  - [ ] Create release automation
  - [ ] Add changelog generation
  - [ ] Implement signed releases

### Phase 10: Performance & Optimization

- [ ] **10.1** Performance optimization

  - [ ] Optimize feature extraction speed
  - [ ] Add parallel processing where possible
  - [ ] Implement model caching
  - [ ] Add memory usage optimization

- [ ] **10.2** Scalability
  - [ ] Test with large documents
  - [ ] Optimize AWS worker scaling
  - [ ] Add batch processing support
  - [ ] Implement resource monitoring

---

## 🔧 Technical Dependencies

### Core Dependencies

- **Python 3.11+**
- **NLP**: spaCy (`en_core_web_md`), `sentence-transformers`, `transformers`
- **ML**: scikit-learn, numpy, scipy
- **CLI**: Typer/Click, `rich` for ANSI tables
- **AWS**: `boto3` for SQS/S3
- **Grammar**: `language-tool-python` (optional)

### Model Dependencies

- **Embeddings**: `all-MiniLM-L6-v2`
- **LM**: DistilGPT-2 for perplexity
- **NLI**: `roberta-base-mnli` or `deberta-v3-base-mnli`

---

## 📊 Success Metrics

### Performance Targets

- **Speed**: < 1.5s for 1k words on 4-core CPU
- **Memory**: < 800 MB cold, < 400 MB hot
- **Determinism**: Fixed seeds, consistent tokenization
- **Portability**: Linux/macOS/Windows support

### Quality Targets

- **Accuracy**: Validated against calibration corpus
- **Reproducibility**: Consistent results across runs
- **Coverage**: All specified axes implemented
- **Documentation**: Complete API and usage docs

---

## 🚀 MVP Scope (v0.1)

**Core Features for MVP:**

- [ ] All 11 per-axis metrics (except factuality/fluency optional)
- [ ] CLI with pretty and JSON output
- [ ] Span annotations and explanations
- [ ] Domain-specific weighting
- [ ] Docker containerization
- [ ] Basic AWS ECS worker

**Post-MVP (v0.2+):**

- [ ] Full factuality with NLI
- [ ] `slop-lr` combiner model
- [ ] Multi-language support
- [ ] GUI report (HTML)
- [ ] VS Code extension

---

## 📝 Notes

- **Determinism**: Use fixed seeds and consistent tokenization
- **Privacy**: No text persistence unless explicitly configured
- **Offline**: No internet required by default
- **Domain-aware**: Default weights for news/qa/general domains
- **Extensible**: Clean architecture for adding new metrics

---

## 🎯 Next Steps

1. **Start with Phase 1**: Set up project structure and development environment
2. **Focus on MVP**: Implement core features for v0.1 release
3. **Iterate quickly**: Build and test incrementally
4. **Document decisions**: Keep architecture decisions documented
5. **Test early**: Implement testing from the beginning

This TODO provides a comprehensive roadmap for implementing the AI Slop CLI project based on the detailed specification and research foundation.
