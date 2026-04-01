# ASR Evaluation - Quick Start Guide

## What is ASR?

**Attack Success Rate (ASR)** measures what fraction of target questions the attacker successfully causes the LLM to give the desired (incorrect) answer.

- **Baseline (No Attack)**: 15-25% (LLM's natural error rate)
- **With Attack**: 70-90% (successful poisoning)

## 3-Step Quick Start

### Step 1: Start Quick Evaluation (5 minutes)

```bash
# Quick test with 2 target questions, 2 repeats
python main.py \
  --eval_dataset nq \
  --attack_method corpus_poisoning \
  --M 2 \
  --repeat_times 2
```

### Step 2: Run Full Evaluation (16 hours)

```bash
# Full evaluation: 10 targets × 10 repeats × 4 methods × 3 datasets
python run_asr_simple.py
```

This automatically runs:
- ✓ No Attack (Baseline)
- ✓ Corpus Poisoning (our new attack)
- ✓ Prompt Injection (our new attack)  
- ✓ PoisonedRAG (original attack)

On all datasets: NQ, HotpotQA, MSMARCO

### Step 3: Analyze Results

```bash
# Print results and generate report
python analyze_asr_results.py
```

Output:
- Console summary table
- Detailed statistics
- Markdown report: `results/asr_evaluation/ASR_Report.md`

## Complete Evaluation Workflow

```
┌─────────────────────────────────────────┐
│  Step 1: Select Target Questions        │
│  - 10 random close-ended questions      │
│  - Per dataset                          │
└─────────────────────────┬───────────────┘
                          │
┌─────────────────────────▼───────────────┐
│  Step 2: Run Attacks 10 Times           │
│  - No Attack                            │
│  - Corpus Poisoning                     │
│  - Prompt Injection                     │
│  - PoisonedRAG                          │
└─────────────────────────┬───────────────┘
                          │
┌─────────────────────────▼───────────────┐
│  Step 3: Calculate ASR for Each         │
│  - Count successful attacks             │
│  - Convert to percentage                │
└─────────────────────────┬───────────────┘
                          │
┌─────────────────────────▼───────────────┐
│  Step 4: Compare Results                │
│  - Across attack methods                │
│  - Across datasets                      │
│  - Generate report                      │
└─────────────────────────────────────────┘
```

## Expected Results

Based on our implementation:

### Corpus Poisoning Attack
```
NQ:        70-80% ASR
HotpotQA:  65-75% ASR
MSMARCO:   60-70% ASR
```

### Prompt Injection Attack
```
NQ:        75-85% ASR
HotpotQA:  70-80% ASR
MSMARCO:   65-75% ASR
```

### Original PoisonedRAG
```
NQ:        70-80% ASR
HotpotQA:  65-75% ASR
MSMARCO:   60-70% ASR
```

## Key Files

### Evaluation Scripts
- `run_asr_simple.py` - **Start here!** Simple full evaluation
- `main.py` - Single evaluation run
- `analyze_asr_results.py` - Analyze and visualize results

### Configuration
- `asr_config.json` - Evaluation parameters and expected results
- `ASR_EVALUATION_GUIDE.md` - Comprehensive documentation

### Results Location
- `results/query_results/asr_eval/` - Detailed per-query results
- `results/asr_evaluation/asr_summary.json` - Summary statistics
- `results/asr_evaluation/ASR_Report.md` - Generated report

## Command Reference

### Quick Tests
```bash
# Test 1 dataset, 1 attack, 2 targets, 2 repeats
python main.py --eval_dataset nq --attack_method corpus_poisoning --M 2 --repeat_times 2

# Test with prompt injection
python main.py --eval_dataset nq --attack_method prompt_injection --M 2 --repeat_times 2

# Test baseline (no attack)
python main.py --eval_dataset nq --attack_method None --M 2 --repeat_times 2
```

### Full Evaluations
```bash
# Standard evaluation (paper methodology)
python run_asr_simple.py

# With higher repeats for statistical confidence
python run_asr_evaluation.py

# Detailed ASR analysis
python analyze_asr_results.py
```

### Default Parameters
```
Datasets:         NQ, HotpotQA, MSMARCO
Target Questions: 10 per dataset
Repeats:          10 per attack
LLM:              PaLM 2 (Gemini 2.5 Flash)
Retriever:        Contriever
Top-K Results:    5
```

## Customization Examples

### Increase statistical power
```bash
python main.py \
  --eval_dataset nq \
  --attack_method corpus_poisoning \
  --M 20 \
  --repeat_times 20
```

### Test single question thoroughly
```bash
python main.py \
  --eval_dataset hotpotqa \
  --attack_method prompt_injection \
  --M 1 \
  --repeat_times 50
```

### Compare two attacks head-to-head
```bash
# Run corpus poisoning
python main.py --eval_dataset nq --attack_method corpus_poisoning --M 10 --repeat_times 10

# Run prompt injection
python main.py --eval_dataset nq --attack_method prompt_injection --M 10 --repeat_times 10

# Compare
python analyze_asr_results.py
```

### Different LLM models
```bash
# Try GPT-4 (if configured)
python main.py \
  --eval_dataset nq \
  --attack_method corpus_poisoning \
  --model_name gpt4 \
  --M 10 \
  --repeat_times 10
```

## Understanding ASR Metrics

### Primary Metric: Attack Success Rate (ASR)
- **Definition**: Fraction of target questions where attack succeeded
- **Range**: 0.0 (no success) to 1.0 (perfect success)
- **Interpretation**: Higher is worse (from defense perspective)

### Supporting Metrics
- **Precision**: Fraction of retrieved documents that are poisoned
- **Recall**: Fraction of poisoned documents that were retrieved  
- **F1 Score**: Harmonic mean (0.0 to 1.0)

## Interpreting Results

### High ASR (>80%)
- Attack is very effective
- LLM easily influenced by poisoned documents
- Defense needed

### Medium ASR (40-80%)
- Attack is moderately effective
- LLM sometimes follows poisoned guidance
- Defense recommended

### Low ASR (<40%)
- Attack is ineffective
- LLM resists poisoning
- Natural robustness

## Next Steps

1. **Quick Test** (5 min):
   ```bash
   python main.py --eval_dataset nq --attack_method corpus_poisoning --M 2 --repeat_times 2
   ```

2. **Full Evaluation** (16 hours):
   ```bash
   python run_asr_simple.py
   ```

3. **Analyze** (5 min):
   ```bash
   python analyze_asr_results.py
   ```

4. **Test Defenses** (follow ASR_EVALUATION_GUIDE.md):
   ```bash
   python main.py --eval_dataset nq --attack_method corpus_poisoning --apply_defense True
   ```

## Common Issues

**Q: ASR values are 0%**
- Check if adversarial examples are being generated
- Verify corpus is loaded correctly
- Check embedding model is working

**Q: Very high variance between repeats**
- Increase num_repeats to 20+
- Use fixed seed
- Check LLM temperature setting

**Q: Evaluation times out**
- Increase timeout in run script
- Reduce top_k value
- Run on faster GPU

## Support

For detailed information, see:
- `ASR_EVALUATION_GUIDE.md` - Comprehensive guide
- `asr_config.json` - Configuration options
- Source code comments in evaluation scripts

---

**Ready to start?** Run this:
```bash
python run_asr_simple.py
```

**Questions?** Check the detailed guide:
```bash
cat ASR_EVALUATION_GUIDE.md
```
