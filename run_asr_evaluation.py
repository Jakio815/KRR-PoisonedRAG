"""
ASR Evaluation Run Script
Evaluates Attack Success Rate following PoisonedRAG paper methodology
- 10 target questions per dataset
- 10 repeats each (for non-determinism)
- Compares: No attack vs PoisonedRAG vs Corpus Poisoning vs Prompt Injection
"""

import os
import json
import random
import numpy as np
import logging
from collections import defaultdict
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ASRRunner:
    def __init__(self):
        self.output_dir = "results/asr_evaluation"
        os.makedirs(self.output_dir, exist_ok=True)
        self.results_summary = defaultdict(lambda: defaultdict(list))
        
    def run_asr_evaluation(self):
        """Run ASR evaluation for all attack methods and datasets"""
        
        datasets = ['nq', 'hotpotqa', 'msmarco']
        attack_methods = [
            ('no_attack', None),
            ('poisoned_rag', 'LM_targeted'),
            ('corpus_poisoning', 'corpus_poisoning'),
            ('prompt_injection', 'prompt_injection'),
        ]
        
        num_target_questions = 10  # Per paper
        num_repeats = 10  # Per paper
        
        logger.info("=" * 100)
        logger.info("ASR EVALUATION - Following PoisonedRAG Paper Methodology")
        logger.info("=" * 100)
        logger.info(f"Datasets: {datasets}")
        logger.info(f"Attack Methods: {[m[0] for m in attack_methods]}")
        logger.info(f"Target Questions per Dataset: {num_target_questions}")
        logger.info(f"Repeats per Method: {num_repeats}")
        logger.info("=" * 100 + "\n")
        
        for dataset in datasets:
            logger.info(f"\n{'='*100}")
            logger.info(f"DATASET: {dataset.upper()}")
            logger.info(f"{'='*100}\n")
            
            for method_name, attack_code in attack_methods:
                logger.info(f"\n{'-'*100}")
                logger.info(f"ATTACK METHOD: {method_name.upper()}")
                logger.info(f"{'-'*100}\n")
                
                asr_values = []
                
                for repeat_idx in range(num_repeats):
                    logger.info(f"Running repeat {repeat_idx + 1}/{num_repeats}...")
                    
                    # Run evaluation for this attack method on this dataset
                    asr = self._run_single_evaluation(
                        dataset=dataset,
                        attack_method=attack_code,
                        method_name=method_name,
                        repeat_idx=repeat_idx,
                        num_target_questions=num_target_questions
                    )
                    
                    asr_values.append(asr)
                    logger.info(f"  ASR for repeat {repeat_idx + 1}: {asr:.2%}\n")
                
                # Calculate statistics
                asr_mean = np.mean(asr_values)
                asr_std = np.std(asr_values)
                asr_min = np.min(asr_values)
                asr_max = np.max(asr_values)
                
                self.results_summary[dataset][method_name] = {
                    'values': asr_values,
                    'mean': asr_mean,
                    'std': asr_std,
                    'min': asr_min,
                    'max': asr_max,
                }
                
                logger.info(f"\nASR Summary for {method_name} on {dataset}:")
                logger.info(f"  Mean ± Std: {asr_mean:.2%} ± {asr_std:.2%}")
                logger.info(f"  Range: [{asr_min:.2%}, {asr_max:.2%}]")
                logger.info(f"  All values: {[f'{v:.2%}' for v in asr_values]}\n")
        
        # Print final comparison table
        self._print_final_comparison()
        self._save_summary()
    
    def _run_single_evaluation(self, dataset, attack_method, method_name, repeat_idx, num_target_questions):
        """
        Run a single evaluation iteration
        Returns: ASR value (0.0 to 1.0)
        """
        
        # Prepare parameters
        params = {
            'eval_model_code': 'contriever',
            'eval_dataset': dataset,
            'split': 'test',
            'query_results_dir': 'asr_eval',
            'model_name': 'palm2',
            'use_truth': 'False',
            'top_k': 5,
            'gpu_id': 0,
            'attack_method': attack_method,
            'adv_per_query': 5,
            'score_function': 'dot',
            'repeat_times': 1,
            'M': num_target_questions,
            'seed': 42 + repeat_idx,  # Different seed for each repeat
            'note': f"ASR_{method_name}_{dataset}_repeat{repeat_idx}"
        }
        
        # Build command
        cmd = self._build_command(params)
        
        logger.debug(f"Running command: {cmd}")
        
        # Run evaluation
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.warning(f"Evaluation failed: {result.stderr}")
                return 0.0
            
            # Parse output to extract ASR
            asr = self._parse_asr_from_output(result.stdout)
            
            return asr
            
        except subprocess.TimeoutExpired:
            logger.warning("Evaluation timed out")
            return 0.0
        except Exception as e:
            logger.warning(f"Error during evaluation: {e}")
            return 0.0
    
    def _build_command(self, params):
        """Build command to run main.py with parameters"""
        cmd = "python main.py "
        for key, value in params.items():
            cmd += f"--{key} {value} "
        return cmd
    
    def _parse_asr_from_output(self, output):
        """Parse ASR value from main.py output"""
        try:
            # Look for "ASR Mean: X.XX" in output
            for line in output.split('\n'):
                if 'ASR Mean:' in line:
                    asr_str = line.split('ASR Mean:')[1].strip().split()[0]
                    return float(asr_str)
        except:
            pass
        
        # Fallback - return 0 if parsing fails
        return 0.0
    
    def _print_final_comparison(self):
        """Print comparison table of ASR results"""
        logger.info("\n" + "=" * 120)
        logger.info("FINAL ASR COMPARISON TABLE")
        logger.info("=" * 120)
        
        # Header
        datasets = list(self.results_summary.keys())
        methods = list(self.results_summary[datasets[0]].keys()) if datasets else []
        
        header = f"{'Attack Method':<25} | "
        for dataset in datasets:
            header += f"{dataset.upper():<30} | "
        logger.info(header)
        logger.info("-" * 120)
        
        # Results
        for method in methods:
            row = f"{method:<25} | "
            for dataset in datasets:
                if dataset in self.results_summary and method in self.results_summary[dataset]:
                    stats = self.results_summary[dataset][method]
                    row += f"{stats['mean']:.2%} ± {stats['std']:.2%}         | "
                else:
                    row += f"{'N/A':<30} | "
            logger.info(row)
        
        logger.info("=" * 120)
        
        # Print detailed stats
        logger.info("\nDETAILED STATISTICS:")
        for dataset in datasets:
            logger.info(f"\n{dataset.upper()}:")
            for method in methods:
                if dataset in self.results_summary and method in self.results_summary[dataset]:
                    stats = self.results_summary[dataset][method]
                    logger.info(f"  {method}:")
                    logger.info(f"    Mean: {stats['mean']:.4f}")
                    logger.info(f"    Std:  {stats['std']:.4f}")
                    logger.info(f"    Min:  {stats['min']:.4f}")
                    logger.info(f"    Max:  {stats['max']:.4f}")
    
    def _save_summary(self):
        """Save summary to JSON file"""
        output_file = os.path.join(self.output_dir, "asr_summary.json")
        
        summary_dict = {}
        for dataset in self.results_summary:
            summary_dict[dataset] = {}
            for method in self.results_summary[dataset]:
                stats = self.results_summary[dataset][method]
                summary_dict[dataset][method] = {
                    'values': [float(v) for v in stats['values']],
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'min': float(stats['min']),
                    'max': float(stats['max']),
                }
        
        with open(output_file, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        
        logger.info(f"\nSummary saved to {output_file}")


def main():
    runner = ASRRunner()
    runner.run_asr_evaluation()


if __name__ == '__main__':
    main()
