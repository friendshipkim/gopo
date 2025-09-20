#!/usr/bin/env python3
"""
Comparison script for analyzing two evaluation files with flipped model order.
Applies logic to compare win results between two JSON evaluation files.
"""

import json
import argparse
from typing import Dict, List, Any, Tuple


def load_evaluation_file(file_path: str) -> Dict[str, Any]:
    """Load and validate an evaluation JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Basic validation
    required_keys = ['results', 'analysis']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")
    
    return data


def compare_evaluations(file1_path: str, file2_path: str, ignore_ties: bool = True) -> Dict[str, Any]:
    """
    Compare two evaluation files and apply the comparison logic.
    
    Args:
        file1_path: Path to first evaluation file
        file2_path: Path to second evaluation file  
        ignore_ties: If True, ignore prompts where either file has a tie
        
    Returns:
        Dictionary with comparison results
    """
    # Load both files
    file1_data = load_evaluation_file(file1_path)
    file2_data = load_evaluation_file(file2_path)
    
    results1 = file1_data['results']
    results2 = file2_data['results']
    
    if len(results1) != len(results2):
        raise ValueError(f"Files have different number of results: {len(results1)} vs {len(results2)}")
    
    # Apply comparison logic
    comparison_results = []
    model1_wins = 0
    model2_wins = 0
    ties = 0
    ignored_due_to_ties = 0
    
    for i, (result1, result2) in enumerate(zip(results1, results2)):
        winner1 = result1['majority_winner']
        winner2 = result2['majority_winner']
        
        # Skip if either has a tie and ignore_ties is True
        if ignore_ties and (winner1 == 'tie' or winner2 == 'tie'):
            ignored_due_to_ties += 1
            continue
        
        # Apply comparison logic
        if winner1 == winner2:
            if winner1 == 'model1':
                final_result = 'tie (both model1)'
                ties += 1
            elif winner1 == 'model2':
                final_result = 'tie (both model2)'
                ties += 1
            else:  # both are 'tie'
                final_result = 'tie (both tie)'
                ties += 1
        elif winner1 == 'model1' and winner2 == 'model2':
            final_result = 'model1 wins'
            model1_wins += 1
        elif winner1 == 'model2' and winner2 == 'model1':
            final_result = 'model2 wins'
            model2_wins += 1
        else:
            # Handle cases where one is 'tie' and other is not (when ignore_ties=False)
            if winner1 == 'tie':
                final_result = f'model2 wins (file1: tie, file2: {winner2})'
                model2_wins += 1
            elif winner2 == 'tie':
                final_result = f'model1 wins (file1: {winner1}, file2: tie)'
                model1_wins += 1
            else:
                final_result = f'unknown: {winner1} vs {winner2}'
        
        comparison_results.append({
            'prompt_index': i + 1,
            'file1_winner': winner1,
            'file2_winner': winner2,
            'final_result': final_result,
            'prompt': result1['prompt'][:100] + '...' if len(result1['prompt']) > 100 else result1['prompt']
        })
    
    # Calculate statistics
    total_analyzed = len(comparison_results)
    model1_win_rate = model1_wins / total_analyzed if total_analyzed > 0 else 0
    model2_win_rate = model2_wins / total_analyzed if total_analyzed > 0 else 0
    tie_rate = ties / total_analyzed if total_analyzed > 0 else 0
    
    # Tie breakdown
    tie_breakdown = {}
    for result in comparison_results:
        if 'tie' in result['final_result']:
            tie_type = result['final_result']
            tie_breakdown[tie_type] = tie_breakdown.get(tie_type, 0) + 1
    
    return {
        'total_prompts': len(results1),
        'total_analyzed': total_analyzed,
        'ignored_due_to_ties': ignored_due_to_ties,
        'model1_wins': model1_wins,
        'model2_wins': model2_wins,
        'ties': ties,
        'model1_win_rate': model1_win_rate,
        'model2_win_rate': model2_win_rate,
        'tie_rate': tie_rate,
        'tie_breakdown': tie_breakdown,
        'comparison_results': comparison_results,
        'file1_stats': file1_data['analysis'],
        'file2_stats': file2_data['analysis']
    }


def print_summary(results: Dict[str, Any], show_examples: int = 10):
    """Print a formatted summary of the comparison results."""
    print("=" * 80)
    print("EVALUATION COMPARISON SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal prompts: {results['total_prompts']}")
    print(f"Prompts analyzed: {results['total_analyzed']}")
    print(f"Ignored due to ties: {results['ignored_due_to_ties']}")
    
    print(f"\nComparison Results:")
    print(f"  Model1 wins: {results['model1_wins']} ({results['model1_win_rate']*100:.1f}%)")
    print(f"  Model2 wins: {results['model2_wins']} ({results['model2_win_rate']*100:.1f}%)")
    print(f"  Ties: {results['ties']} ({results['tie_rate']*100:.1f}%)")
    
    if results['tie_breakdown']:
        print(f"\nTie breakdown:")
        for tie_type, count in results['tie_breakdown'].items():
            print(f"  {tie_type}: {count}")
    
    print(f"\nOriginal file statistics:")
    print(f"File 1:")
    print(f"  Model1 wins: {results['file1_stats']['model1_wins']} ({results['file1_stats']['model1_win_rate']*100:.1f}%)")
    print(f"  Model2 wins: {results['file1_stats']['model2_wins']} ({results['file1_stats']['model2_win_rate']*100:.1f}%)")
    print(f"  Ties: {results['file1_stats']['ties']} ({results['file1_stats']['tie_rate']*100:.1f}%)")
    
    print(f"File 2:")
    print(f"  Model1 wins: {results['file2_stats']['model1_wins']} ({results['file2_stats']['model1_win_rate']*100:.1f}%)")
    print(f"  Model2 wins: {results['file2_stats']['model2_wins']} ({results['file2_stats']['model2_win_rate']*100:.1f}%)")
    print(f"  Ties: {results['file2_stats']['ties']} ({results['file2_stats']['tie_rate']*100:.1f}%)")
    
    if show_examples > 0:
        print(f"\nFirst {show_examples} examples:")
        print("-" * 50)
        for i, result in enumerate(results['comparison_results'][:show_examples]):
            print(f"Prompt {result['prompt_index']:2d}: File1={result['file1_winner']:7s} vs File2={result['file2_winner']:7s} -> {result['final_result']}")
            print(f"  Prompt: {result['prompt']}")
            print()


def save_results(results: Dict[str, Any], output_path: str):
    """Save comparison results to a JSON file."""
    # Remove the detailed comparison_results for cleaner output
    output_data = {k: v for k, v in results.items() if k != 'comparison_results'}
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare two evaluation files with flipped model order")
    parser.add_argument("file1", help="Path to first evaluation file")
    parser.add_argument("file2", help="Path to second evaluation file")
    parser.add_argument("--ignore-ties", action="store_true", default=True,
                       help="Ignore prompts where either file has a tie (default: True)")
    parser.add_argument("--show-examples", type=int, default=10,
                       help="Number of examples to show (default: 10)")
    parser.add_argument("--output", help="Output file to save results")
    
    args = parser.parse_args()
    
    try:
        results = compare_evaluations(args.file1, args.file2, args.ignore_ties)
        print_summary(results, args.show_examples)
        
        if args.output:
            save_results(results, args.output)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())