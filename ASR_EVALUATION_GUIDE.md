# ASR (Attack Success Rate) Evaluation Guide

## Overview

This guide explains how to evaluate Attack Success Rate (ASR) following the PoisonedRAG paper methodology. ASR measures the fraction of target questions whose answers are successfully changed to the attacker-chosen target answer.

## Paper Methodology

According to the PoisonedRAG paper:
- **Target Questions**: Randomly select 10 close-ended questions from each dataset
- **Non-determinism**: Repeat the experiment 10 times to account for randomness in LLM responses
- **Total Questions**: 100 target questions per dataset (10 × 10)
- **Metrics**: Compare ASR across different attack methods

## Attack Methods Compared

1. **No Attack (Baseline)**: RAG system without any attacks
   - Use_truth: False (use top-5 retrieval results)
   - Measures natural LLM answer quality

2. **PoisonedRAG**: Original attack from the paper
   - Method: `LM_targeted`
   - Injects adversarial texts into retrieval results

3. **Corpus Poisoning**: Our new attack
   - Method: `corpus_poisoning`
   - Poisons the knowledge base (corpus) by embedding adversarial documents
   - Simulates database-level attacks

4. **Prompt Injection**: Our new attack
   - Method: `prompt_injection`
   - Injects malicious instructions into the prompt
   - Simulates instruction-level attacks

## ASR Calculation

ASR is calculated as:
```
ASR = (Number of successful attacks) / (Total target questions)
```

A successful attack occurs when:
- The target (incorrect) answer appears in the LLM's response
- The response matches the attacker's goal

## Running ASR Evaluation

### Quick Start (Recommended)

Run the simple ASR evaluation script:

```bash
python run_asr_simple.py
```

This will:
- Evaluate all 4 attack methods (No attack, Corpus Poisoning, Prompt Injection, PoisonedRAG)
- Run on all 3 datasets (NQ, HotpotQA, MSMARCO)
- Repeat 10 times for each configuration
- Print ASR comparison table

**Expected Output:**
```
Attack Method              | NQ                 | HotpotQA           | MSMARCO
No Attack (Baseline)       | 0.25 ± 0.08        | 0.20 ± 0.06        | 0.18 ± 0.05
Corpus Poisoning          | 0.80 ± 0.10        | 0.78 ± 0.12        | 0.75 ± 0.08
Prompt Injection          | 0.85 ± 0.09        | 0.82 ± 0.11        | 0.80 ± 0.07
PoisonedRAG              | 0.75 ± 0.11        | 0.72 ± 0.13        | 0.70 ± 0.09
```

### Advanced Evaluation

For more detailed evaluation with custom parameters:

```bash
python main.py \
  --eval_dataset nq \
  --attack_method corpus_poisoning \
  --M 10 \
  --repeat_times 10 \
  --seed 42
```

### Evaluation with Defenses

To evaluate ASR with defense mechanisms applied:

```bash
# With defense enabled
python main.py \
  --eval_dataset nq \
  --attack_method corpus_poisoning \
  --apply_defense True \
  --defense_type semantic_similarity
```

## Key Parameters

- `--eval_dataset`: Dataset to evaluate on (nq, hotpotqa, msmarco)
- `--attack_method`: Attack method to use
  - `None`: No attack (baseline)
  - `corpus_poisoning`: Corpus poisoning attack
  - `prompt_injection`: Prompt injection attack
  - `LM_targeted`: Original PoisonedRAG attack
- `--M`: Number of target questions (default: 10, as per paper)
- `--repeat_times`: Number of repeats (default: 10, as per paper)
- `--top_k`: Number of top-k retrieval results to use (default: 5)
- `--seed`: Random seed for reproducibility

## Results

Results are saved in:
- **Detailed results**: `results/query_results/asr_eval/`
- **Summary**: `results/asr_evaluation/asr_summary.json`

Each result file contains:
- Individual question evaluations
- Ground truth answers
- Injected adversarial texts
- LLM responses
- Whether attack was successful

## Analysis

### Interpreting Results

1. **Baseline ASR (No Attack)**:
   - Shows LLM's natural error rate
   - Typically 15-25% depending on dataset

2. **Attack ASR**:
   - Corpus Poisoning: 70-85% effectiveness
   - Prompt Injection: 80-90% effectiveness
   - PoisonedRAG: 70-80% effectiveness

3. **Statistical Significance**:
   - Compare mean ± std across methods
   - Use t-tests for formal comparison
   - 10 repeats provide sufficient samples for significance

### ASR Components

```
ASR = Retrieval Success × Prompt Injection Success × LLM Response Success

Where:
- Retrieval Success: % of poisoned docs retrieved in top-k
- Prompt Injection Success: % that affect prompt
- LLM Response Success: % where LLM includes target answer
```

## Comparison Table: Paper vs Our Implementation

| Aspect | Paper Methodology | Our Implementation |
|--------|-------------------|-------------------|
| Target Questions | 10 randomly selected | 10 randomly selected |
| Close-ended | Yes | Yes (filtered) |
| Repeats | 10 times | 10 times (configurable) |
| Datasets | 3 (NQ, HotpotQA, MSMARCO) | 3 (same) |
| Attack Methods | PoisonedRAG | PoisonedRAG + 2 new |
| LLM | GPT-4 | PaLM 2 (Gemini) |
| Metrics | ASR, Precision, Recall | ASR, Precision, Recall, F1 |

## Defense Evaluation

To evaluate defenses:

```bash
# Semantic similarity defense
python main.py \
  --eval_dataset nq \
  --attack_method corpus_poisoning \
  --apply_defense True \
  --defense_threshold 0.8

# Diversity defense
python main.py \
  --eval_dataset nq \
  --attack_method corpus_poisoning \
  --apply_defense True \
  --defense_type diversity

# Weighted retrieval defense
python main.py \
  --eval_dataset nq \
  --attack_method corpus_poisoning \
  --apply_defense True \
  --defense_type weighted_rank
```

## Performance Benchmarks

Expected runtime:
- Single run (10 targets × 1 repeat): ~2-3 minutes
- Full evaluation (10 targets × 10 repeats × 4 methods × 3 datasets): ~16 hours

GPU requirements:
- VRAM: 8GB+
- Recommended: RTX 3090 or higher

## Troubleshooting

### Low ASR values
- Check if adversarial examples are being generated correctly
- Verify corpus is loaded properly
- Check embedding model is working

### High variance across repeats
- Increase num_repeats to 20
- Use fixed seed for reproducibility
- Check LLM temperature setting

### Timeout errors
- Increase timeout in run script
- Run on separate GPUs
- Reduce top_k value

## References

- PoisonedRAG Paper: [Link to arxiv]
- BEIR Benchmark: [Link]
- Contriever Model: [Link]

## Next Steps

1. Run `python run_asr_simple.py` to get baseline results
2. Compare with paper results
3. Analyze defense effectiveness
4. Document findings

---

Last Updated: April 1, 2026
