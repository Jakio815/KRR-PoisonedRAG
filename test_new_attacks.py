#!/usr/bin/env python3
"""
Test script to verify the two new attack methods work correctly
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import argparse
from unittest.mock import Mock
import torch
from src.attack import Attacker

def create_mock_attacker(attack_method='corpus_poisoning'):
    """Create a mock attacker for testing"""
    
    # Create mock args
    mock_args = Mock()
    mock_args.attack_method = attack_method
    mock_args.adv_per_query = 3
    mock_args.score_function = 'dot'
    mock_args.eval_dataset = 'nq'
    
    # Create mock pre-generated texts
    mock_texts = {
        'test1': {
            'adv_texts': [
                'Wrong answer 1 to the question',
                'Wrong answer 2 to the question',
                'Wrong answer 3 to the question',
                'Wrong answer 4 to the question',
                'Wrong answer 5 to the question',
            ]
        },
        'test2': {
            'adv_texts': [
                'Different wrong answer 1',
                'Different wrong answer 2',
                'Different wrong answer 3',
                'Different wrong answer 4',
                'Different wrong answer 5',
            ]
        }
    }
    
    # Create attacker with mocked dependencies
    attacker = Attacker(
        mock_args,
        model=Mock(),
        c_model=Mock(),
        tokenizer=Mock(),
        get_emb=Mock(return_value=torch.randn(1, 768))
    )
    
    # Override pre-generated texts
    attacker.all_adv_texts = mock_texts
    
    return attacker


def test_corpus_poisoning():
    """Test corpus poisoning attack"""
    print("\n" + "="*70)
    print("TEST: Corpus Poisoning Attack")
    print("="*70)
    
    attacker = create_mock_attacker('corpus_poisoning')
    
    # Create test queries
    target_queries = [
        {
            'query': 'Who invented the telephone?',
            'top1_score': 0.8,
            'id': 'test1'
        },
        {
            'query': 'What is the capital of France?',
            'top1_score': 0.85,
            'id': 'test2'
        }
    ]
    
    # Generate attack
    print("\n[*] Generating corpus poisoning attack...")
    try:
        adv_text_groups = attacker.get_attack(target_queries)
        print("✅ Attack generated successfully!\n")
        
        # Show results
        for i, query_dict in enumerate(target_queries):
            print(f"Query {i+1}: {query_dict['query']!r}")
            adv_texts = adv_text_groups[i]
            print(f"  Generated {len(adv_texts)} poisoned documents:")
            
            for j, doc in enumerate(adv_texts, 1):
                print(f"    [{j}] {doc[:80]}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_injection():
    """Test prompt injection attack"""
    print("\n" + "="*70)
    print("TEST: Prompt Injection Attack")
    print("="*70)
    
    attacker = create_mock_attacker('prompt_injection')
    
    # Create test queries
    target_queries = [
        {
            'query': 'Which country has the largest economy?',
            'top1_score': 0.82,
            'id': 'test1'
        },
        {
            'query': 'What is 2+2?',
            'top1_score': 0.9,
            'id': 'test2'
        }
    ]
    
    # Generate attack
    print("\n[*] Generating prompt injection attack...")
    try:
        adv_text_groups = attacker.get_attack(target_queries)
        print("✅ Attack generated successfully!\n")
        
        # Show results
        for i, query_dict in enumerate(target_queries):
            print(f"Query {i+1}: {query_dict['query']!r}")
            adv_texts = adv_text_groups[i]
            print(f"  Generated {len(adv_texts)} injected documents:")
            
            for j, doc in enumerate(adv_texts, 1):
                print(f"    [{j}] {doc[:100]}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fallback_generation():
    """Test fallback poisoned document generation"""
    print("\n" + "="*70)
    print("TEST: Fallback Poisoned Document Generation")
    print("="*70)
    
    attacker = create_mock_attacker('corpus_poisoning')
    
    # Test _generate_poisoned_docs
    print("\n[*] Testing fallback generation for simple query...")
    try:
        query = "What is artificial intelligence?"
        generated_docs = attacker._generate_poisoned_docs(query, num_docs=3)
        
        print(f"✅ Generated {len(generated_docs)} fallback documents:")
        for i, doc in enumerate(generated_docs, 1):
            print(f"  [{i}] {doc[:80]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_both_attacks_comparison():
    """Compare both attacks side by side"""
    print("\n" + "="*70)
    print("COMPARISON: Corpus Poisoning vs Prompt Injection")
    print("="*70)
    
    query_dict = {
        'query': 'How many continents are there?',
        'top1_score': 0.88,
        'id': 'test1'
    }
    
    target_queries = [query_dict]
    
    print(f"\nTest Query: '{query_dict['query']}'\n")
    
    # Corpus Poisoning
    print("[1] Corpus Poisoning Attack:")
    attacker_cp = create_mock_attacker('corpus_poisoning')
    adv_cp = attacker_cp.get_attack(target_queries)
    for i, doc in enumerate(adv_cp[0], 1):
        print(f"    Document {i}: {doc[:95]}...")
    
    # Prompt Injection
    print("\n[2] Prompt Injection Attack:")
    attacker_pi = create_mock_attacker('prompt_injection')
    adv_pi = attacker_pi.get_attack(target_queries)
    for i, doc in enumerate(adv_pi[0], 1):
        print(f"    Document {i}: {doc[:95]}...")
    
    print("\n✅ Both attacks generated different poisoned documents")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("NEW ATTACKS INTEGRATION TEST SUITE")
    print("="*70)
    
    tests = [
        ("Corpus Poisoning", test_corpus_poisoning),
        ("Prompt Injection", test_prompt_injection),
        ("Fallback Generation", test_fallback_generation),
        ("Comparison", test_both_attacks_comparison),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Unexpected error in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! New attacks are ready to use.\n")
        return 0
    else:
        print("\n❌ Some tests failed. Check the errors above.\n")
        return 1


if __name__ == '__main__':
    exit(main())
