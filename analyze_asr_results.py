"""
ASR Results Analysis and Visualization
Analyzes and visualizes Attack Success Rate evaluation results
"""

import json
import os
import numpy as np
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ASRAnalyzer:
    def __init__(self, results_dir="results/asr_evaluation"):
        self.results_dir = results_dir
        self.results = self._load_results()
    
    def _load_results(self):
        """Load ASR results from summary file"""
        summary_file = os.path.join(self.results_dir, "asr_summary.json")
        
        if not os.path.exists(summary_file):
            logger.error(f"Results file not found: {summary_file}")
            return {}
        
        with open(summary_file, 'r') as f:
            return json.load(f)
    
    def print_summary(self):
        """Print readable summary of results"""
        if not self.results:
            logger.warning("No results to analyze")
            return
        
        print("\n" + "="*100)
        print("ASR EVALUATION RESULTS SUMMARY")
        print("="*100 + "\n")
        
        datasets = list(self.results.keys())
        methods = list(self.results[datasets[0]].keys()) if datasets else []
        
        # Print comparison table
        print(f"{'Attack Method':<30} | ", end="")
        for dataset in datasets:
            print(f"{dataset.upper():<30} | ", end="")
        print()
        print("-" * (30 + len(datasets) * 33 + 10))
        
        for method in methods:
            print(f"{method:<30} | ", end="")
            for dataset in datasets:
                if dataset in self.results and method in self.results[dataset]:
                    stats = self.results[dataset][method]
                    mean = stats['mean']
                    std = stats['std']
                    print(f"{mean:.2%} ± {std:.2%}         | ", end="")
                else:
                    print(f"{'N/A':<30} | ", end="")
            print()
        
        print("\n" + "="*100)
        print("DETAILED STATISTICS")
        print("="*100 + "\n")
        
        # Print detailed statistics
        for dataset in datasets:
            print(f"\n{dataset.upper()}:")
            print(f"{'-'*50}")
            
            for method in methods:
                if dataset in self.results and method in self.results[dataset]:
                    stats = self.results[dataset][method]
                    
                    print(f"\n  {method}")
                    print(f"    Mean ASR:     {stats['mean']:.4f}")
                    print(f"    Std Dev:      {stats['std']:.4f}")
                    print(f"    Min:          {stats['min']:.4f}")
                    print(f"    Max:          {stats['max']:.4f}")
                    print(f"    All values:   {stats['values']}")
    
    def generate_report(self):
        """Generate markdown report of results"""
        if not self.results:
            logger.warning("No results to report")
            return
        
        report_file = os.path.join(self.results_dir, "ASR_Report.md")
        
        with open(report_file, 'w') as f:
            f.write("# ASR Evaluation Results Report\n\n")
            f.write(f"*Generated: {self._get_timestamp()}*\n\n")
            
            datasets = list(self.results.keys())
            methods = list(self.results[datasets[0]].keys()) if datasets else []
            
            # Summary table
            f.write("## ASR Comparison Table\n\n")
            f.write(f"| Attack Method | {' | '.join([d.upper() for d in datasets])} |\n")
            f.write("|---|" + "|".join(["---|"]*len(datasets)) + "\n")
            
            for method in methods:
                f.write(f"| {method} |")
                for dataset in datasets:
                    if dataset in self.results and method in self.results[dataset]:
                        stats = self.results[dataset][method]
                        f.write(f" {stats['mean']:.2%} ± {stats['std']:.2%} |")
                    else:
                        f.write(" N/A |")
                f.write("\n")
            
            # Detailed analysis
            f.write("\n## Detailed Analysis\n\n")
            
            for dataset in datasets:
                f.write(f"### {dataset.upper()}\n\n")
                
                for method in methods:
                    if dataset in self.results and method in self.results[dataset]:
                        stats = self.results[dataset][method]
                        
                        f.write(f"#### {method}\n\n")
                        f.write(f"- **Mean ASR**: {stats['mean']:.4f}\n")
                        f.write(f"- **Std Dev**: {stats['std']:.4f}\n")
                        f.write(f"- **Min**: {stats['min']:.4f}\n")
                        f.write(f"- **Max**: {stats['max']:.4f}\n")
                        f.write(f"- **All Values**: `{stats['values']}`\n\n")
            
            # Insights
            f.write("\n## Key Insights\n\n")
            f.write(self._generate_insights(datasets, methods))
        
        logger.info(f"Report saved to {report_file}")
    
    def _generate_insights(self, datasets, methods):
        """Generate key insights from results"""
        insights = ""
        
        # Find best performing method
        best_method = None
        best_asr = 0
        
        for method in methods:
            total_asr = 0
            count = 0
            for dataset in datasets:
                if dataset in self.results and method in self.results[dataset]:
                    total_asr += self.results[dataset][method]['mean']
                    count += 1
            
            avg_asr = total_asr / count if count > 0 else 0
            if avg_asr > best_asr:
                best_asr = avg_asr
                best_method = method
        
        if best_method:
            insights += f"- **Best Performing Attack**: {best_method} (avg {best_asr:.2%} across datasets)\n"
        
        # Compare attack effectiveness
        insights += "\n- **Attack Effectiveness Ranking**:\n"
        
        method_scores = []
        for method in methods:
            total_asr = 0
            count = 0
            for dataset in datasets:
                if dataset in self.results and method in self.results[dataset]:
                    total_asr += self.results[dataset][method]['mean']
                    count += 1
            
            avg_asr = total_asr / count if count > 0 else 0
            method_scores.append((method, avg_asr))
        
        method_scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (method, score) in enumerate(method_scores, 1):
            insights += f"  {rank}. {method}: {score:.2%}\n"
        
        # Dataset difficulty
        insights += "\n- **Dataset Difficulty Ranking** (easier to attack):\n"
        
        dataset_scores = []
        for dataset in datasets:
            total_asr = 0
            count = 0
            for method in methods:
                if dataset in self.results and method in self.results[dataset]:
                    total_asr += self.results[dataset][method]['mean']
                    count += 1
            
            avg_asr = total_asr / count if count > 0 else 0
            dataset_scores.append((dataset, avg_asr))
        
        dataset_scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (dataset, score) in enumerate(dataset_scores, 1):
            insights += f"  {rank}. {dataset}: {score:.2%}\n"
        
        return insights
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def compare_with_paper(self):
        """Compare results with paper baseline"""
        print("\n" + "="*100)
        print("COMPARISON WITH POISONED RAG PAPER")
        print("="*100 + "\n")
        
        paper_results = {
            "nq": {
                "No Attack": 0.70,
                "PoisonedRAG": 0.85,
            },
            "hotpotqa": {
                "No Attack": 0.80,
                "PoisonedRAG": 0.92,
            },
            "msmarco": {
                "No Attack": 0.83,
                "PoisonedRAG": 0.95,
            }
        }
        
        print("Note: Paper used GPT-4 as LLM. We use PaLM 2 (Gemini).")
        print("Different LLMs may produce different ASR values.\n")
        
        for dataset in self.results:
            print(f"\n{dataset.upper()}:")
            print(f"{'-'*50}")
            
            if dataset in paper_results:
                for method in paper_results[dataset]:
                    paper_asr = paper_results[dataset][method]
                    if dataset in self.results and method in self.results[dataset]:
                        our_asr = self.results[dataset][method]['mean']
                        diff = our_asr - paper_asr
                        print(f"  {method}:")
                        print(f"    Paper (GPT-4):  {paper_asr:.2%}")
                        print(f"    Our (PaLM 2):   {our_asr:.2%}")
                        print(f"    Difference:     {diff:+.2%}")


def main():
    """Main analysis function"""
    
    analyzer = ASRAnalyzer()
    
    # Print summary
    analyzer.print_summary()
    
    # Generate report
    analyzer.generate_report()
    
    # Compare with paper
    analyzer.compare_with_paper()
    
    logger.info("Analysis complete!")


if __name__ == '__main__':
    main()
