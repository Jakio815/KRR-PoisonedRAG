"""
Simple ASR Evaluation Runner
Follows PoisonedRAG paper methodology:
- 10 randomly selected close-ended questions per dataset
- 10 repeats (accounting for non-determinism)
- Compare ASR: No attack vs Corpus Poisoning vs Prompt Injection
"""

import os
import sys
import json
import subprocess
from collections import defaultdict
import numpy as np


def run_evaluation(dataset, attack_method, num_questions=10, num_repeats=10):
    """
    Run evaluation and collect ASR for all repeats
    
    Attack methods:
    - None: No attack (baseline)
    - 'corpus_poisoning': Corpus poisoning attack
    - 'prompt_injection': Prompt injection attack
    - 'LM_targeted': Original PoisonedRAG attack
    """
    
    print(f"\n{'='*80}")
    print(f"Evaluating {dataset} with {attack_method if attack_method else 'NO ATTACK'}")
    print(f"{'='*80}")
    
    asr_list = []
    
    for repeat_idx in range(num_repeats):
        print(f"\nRepeat {repeat_idx + 1}/{num_repeats}...")
        
        # Build command for main.py
        attack_arg = f"--attack_method {attack_method}" if attack_method else "--attack_method None"
        
        cmd = (
            f"python main.py "
            f"--eval_model_code contriever "
            f"--eval_dataset {dataset} "
            f"--split test "
            f"--query_results_dir asr_eval "
            f"--model_name palm2 "
            f"--use_truth False "
            f"--top_k 5 "
            f"--gpu_id 0 "
            f"{attack_arg} "
            f"--adv_per_query 5 "
            f"--score_function dot "
            f"--repeat_times 1 "
            f"--M {num_questions} "
            f"--seed {42 + repeat_idx} "
            f"--name asr_{dataset}_{attack_method or 'no_attack'}_repeat{repeat_idx}"
        )
        
        print(f"Running: {cmd}")
        
        # Execute
        result = os.system(cmd)
        
        if result == 0:
            print(f"✓ Repeat {repeat_idx + 1} completed")
        else:
            print(f"✗ Repeat {repeat_idx + 1} failed")
    
    return asr_list


def main():
    """Run ASR evaluation following paper methodology"""
    
    datasets = ['nq', 'hotpotqa', 'msmarco']
    
    # Attack methods to compare
    attack_configs = [
        ('No Attack (Baseline)', None),
        ('PoisonedRAG', 'LM_targeted'),
        ('Corpus Poisoning', 'corpus_poisoning'),
        ('Prompt Injection', 'prompt_injection'),
    ]
    
    results = defaultdict(lambda: defaultdict(list))
    
    print("\n" + "="*80)
    print("ASR EVALUATION - Following PoisonedRAG Paper")
    print("="*80)
    print("Configuration:")
    print(f"  - Datasets: {', '.join(datasets)}")
    print(f"  - Attack Methods: {', '.join([name for name, _ in attack_configs])}")
    print(f"  - Target Questions: 10 per dataset")
    print(f"  - Repeats: 10 (for non-determinism)")
    print("="*80 + "\n")
    
    for dataset in datasets:
        print(f"\n{'#'*80}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'#'*80}")
        
        for attack_name, attack_method in attack_configs:
            print(f"\n{'-'*80}")
            print(f"Attack: {attack_name}")
            print(f"{'-'*80}")
            
            # Run evaluation (10 repeats)
            asr_list = run_evaluation(
                dataset=dataset,
                attack_method=attack_method,
                num_questions=10,
                num_repeats=10
            )
            
            results[dataset][attack_name] = asr_list
    
    print("\n\n" + "="*80)
    print("FINAL ASR RESULTS")
    print("="*80 + "\n")
    
    # Print comparison table
    print(f"{'Attack Method':<30} | {'NQ':<20} | {'HotpotQA':<20} | {'MSMARCO':<20}")
    print("-" * 95)
    
    for attack_name, _ in attack_configs:
        row = f"{attack_name:<30} | "
        for dataset in datasets:
            if attack_name in results[dataset]:
                asr_vals = results[dataset][attack_name]
                if asr_vals:
                    mean = np.mean(asr_vals)
                    std = np.std(asr_vals)
                    row += f"{mean:.2%} ± {std:.2%}         | "
                else:
                    row += f"{'N/A':<20} | "
            else:
                row += f"{'N/A':<20} | "
        print(row)
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("Results saved in: results/query_results/asr_eval/")
    print("="*80)


if __name__ == '__main__':
    main()
