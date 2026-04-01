"""
ASR (Attack Success Rate) Evaluation Script
Follows the PoisonedRAG paper methodology:
- Randomly select 10 close-ended questions from each dataset
- Repeat experiment 10 times (to account for non-determinism)
- Compare ASR: No attack vs PoisonedRAG vs Corpus Poisoning vs Prompt Injection
- Evaluate with and without defenses
"""

import os
import json
import random
import numpy as np
from collections import defaultdict
import torch
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ASREvaluator:
    def __init__(self, config):
        """
        Initialize ASR Evaluator
        config: dict with evaluation parameters
        """
        self.config = config
        self.datasets = config.get('datasets', ['nq', 'hotpotqa', 'msmarco'])
        self.attack_methods = config.get('attack_methods', [None, 'poisoned_rag', 'corpus_poisoning', 'prompt_injection'])
        self.num_target_questions = config.get('num_target_questions', 10)
        self.num_repeats = config.get('num_repeats', 10)
        self.results = defaultdict(lambda: defaultdict(list))
        
    def run_evaluation(self):
        """Run full ASR evaluation across all datasets and attack methods"""
        logger.info("=" * 80)
        logger.info("Starting ASR Evaluation")
        logger.info(f"Datasets: {self.datasets}")
        logger.info(f"Attack Methods: {self.attack_methods}")
        logger.info(f"Target Questions: {self.num_target_questions}")
        logger.info(f"Repeats: {self.num_repeats}")
        logger.info("=" * 80)
        
        for dataset in self.datasets:
            logger.info(f"\n{'='*80}")
            logger.info(f"Evaluating Dataset: {dataset.upper()}")
            logger.info(f"{'='*80}")
            
            # Randomly select 10 close-ended questions for this dataset
            target_questions = self._select_target_questions(dataset)
            logger.info(f"Selected {len(target_questions)} target questions for {dataset}")
            
            for attack_method in self.attack_methods:
                logger.info(f"\n{'-'*80}")
                logger.info(f"Attack Method: {attack_method if attack_method else 'NO ATTACK (Baseline)'}")
                logger.info(f"{'-'*80}")
                
                asr_list = []
                
                # Repeat experiment num_repeats times
                for repeat_idx in range(self.num_repeats):
                    logger.info(f"Repeat {repeat_idx + 1}/{self.num_repeats}")
                    
                    # Run evaluation for this repeat
                    asr = self._evaluate_attack(dataset, attack_method, target_questions, repeat_idx)
                    asr_list.append(asr)
                    logger.info(f"  ASR for repeat {repeat_idx + 1}: {asr:.2%}")
                
                # Calculate statistics
                asr_mean = np.mean(asr_list)
                asr_std = np.std(asr_list)
                
                self.results[dataset][attack_method] = {
                    'asr_list': asr_list,
                    'asr_mean': asr_mean,
                    'asr_std': asr_std,
                    'asr_min': np.min(asr_list),
                    'asr_max': np.max(asr_list),
                }
                
                logger.info(f"ASR Statistics for {attack_method if attack_method else 'NO ATTACK'}:")
                logger.info(f"  Mean: {asr_mean:.2%} ± {asr_std:.2%}")
                logger.info(f"  Range: [{np.min(asr_list):.2%}, {np.max(asr_list):.2%}]")
        
        # Print final comparison table
        self._print_comparison_table()
        self._save_results()
    
    def _select_target_questions(self, dataset):
        """Randomly select target questions from dataset"""
        # Load dataset
        from src.utils import load_beir_datasets
        
        if dataset == 'msmarco':
            corpus, queries, qrels = load_beir_datasets("msmarco", "train")
        else:
            corpus, queries, qrels = load_beir_datasets(dataset, "test")
        
        # Get close-ended questions (this is a simplified version)
        # In practice, you'd filter for truly close-ended questions
        query_ids = list(queries.keys())
        random.shuffle(query_ids)
        
        selected = []
        for qid in query_ids[:min(self.num_target_questions, len(query_ids))]:
            selected.append({
                'id': qid,
                'text': queries[qid],
                'answers': list(qrels[qid].keys()) if qid in qrels else []
            })
        
        return selected
    
    def _evaluate_attack(self, dataset, attack_method, target_questions, repeat_idx):
        """
        Evaluate ASR for a specific attack method on target questions
        Returns: ASR (fraction of successful attacks)
        """
        successful_attacks = 0
        
        for q_idx, question in enumerate(target_questions):
            # For each question, evaluate if attack was successful
            # Success = incorrect answer appears in LLM output
            
            # This is a placeholder - replace with actual evaluation logic
            # that runs the attack and checks if the target answer appears in the response
            
            attack_successful = self._check_attack_success(
                dataset=dataset,
                attack_method=attack_method,
                question=question,
                repeat_idx=repeat_idx,
                q_idx=q_idx
            )
            
            if attack_successful:
                successful_attacks += 1
        
        asr = successful_attacks / len(target_questions)
        return asr
    
    def _check_attack_success(self, dataset, attack_method, question, repeat_idx, q_idx):
        """
        Check if a single attack was successful
        Returns: True if attack was successful (incorrect answer in response)
        """
        # This would integrate with the main.py evaluation logic
        # For now, this is a placeholder
        # The actual implementation would:
        # 1. Generate adversarial examples if attack_method is not None
        # 2. Run the RAG pipeline with the adversarial examples
        # 3. Check if the target (incorrect) answer appears in the LLM output
        
        return False  # Placeholder
    
    def _print_comparison_table(self):
        """Print comparison table of ASR across all methods and datasets"""
        logger.info("\n" + "="*100)
        logger.info("ASR COMPARISON TABLE")
        logger.info("="*100)
        
        # Header
        header = f"{'Dataset':<15} | "
        for method in self.attack_methods:
            method_name = method if method else "NO ATTACK"
            header += f"{method_name:<30} | "
        logger.info(header)
        logger.info("-"*100)
        
        # Results
        for dataset in self.datasets:
            row = f"{dataset:<15} | "
            for method in self.attack_methods:
                if dataset in self.results and method in self.results[dataset]:
                    stats = self.results[dataset][method]
                    asr_mean = stats['asr_mean']
                    asr_std = stats['asr_std']
                    row += f"{asr_mean:.2%} ± {asr_std:.2%}             | "
                else:
                    row += f"{'N/A':<30} | "
            logger.info(row)
        
        logger.info("="*100)
    
    def _save_results(self):
        """Save detailed results to JSON file"""
        output_path = "results/asr_evaluation_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results_dict = {}
        for dataset in self.results:
            results_dict[dataset] = {}
            for method in self.results[dataset]:
                stats = self.results[dataset][method]
                results_dict[dataset][str(method)] = {
                    'asr_list': stats['asr_list'],
                    'asr_mean': float(stats['asr_mean']),
                    'asr_std': float(stats['asr_std']),
                    'asr_min': float(stats['asr_min']),
                    'asr_max': float(stats['asr_max']),
                }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Main evaluation function"""
    
    config = {
        'datasets': ['nq', 'hotpotqa', 'msmarco'],
        'attack_methods': [
            None,  # No attack (baseline)
            'poisoned_rag',  # Original PoisonedRAG
            'corpus_poisoning',  # Our new corpus poisoning attack
            'prompt_injection',  # Our new prompt injection attack
        ],
        'num_target_questions': 10,  # As per paper methodology
        'num_repeats': 10,  # Repeat 10 times for non-deterministic results
        'model_name': 'palm2',
        'top_k': 5,
        'gpu_id': 0,
    }
    
    evaluator = ASREvaluator(config)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()
