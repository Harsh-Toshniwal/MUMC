"""
VQA-RAD Dataset Preprocessing Script
This script converts the downloaded VQA-RAD dataset into the format required by MUMC model.
"""

import json
import os
from collections import Counter
import random

def preprocess_vqa_rad(
    input_json='ofs_dataset/VQA_RAD Dataset Public.json',
    output_dir='ofs_dataset',
    train_ratio=0.8,
    seed=42
):
    """
    Process VQA-RAD dataset and create train/test splits with answer list.
    
    Args:
        input_json: Path to the original VQA-RAD JSON file
        output_dir: Directory to save processed files
        train_ratio: Ratio of data to use for training (default: 0.8)
        seed: Random seed for reproducibility
    """
    
    print("=" * 60)
    print("VQA-RAD Dataset Preprocessing")
    print("=" * 60)
    
    # Load the original dataset
    print(f"\n1. Loading dataset from: {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   Total QA pairs: {len(data)}")
    
    # Format each entry for MUMC
    # MUMC expects: {'image_name': str, 'question': str, 'answer': str, 'qid': str}
    # Optional fields that may be useful: 'image_organ', 'question_type', 'answer_type'
    
    formatted_data = []
    all_answers = []
    
    for item in data:
        formatted_item = {
            'qid': item['qid'],
            'image_name': item['image_name'],
            'question': item['question'],
            'answer': str(item['answer']),  # Ensure answer is string
            'image_organ': item.get('image_organ', ''),
            'question_type': item.get('question_type', ''),
            'answer_type': item.get('answer_type', ''),
            'phrase_type': item.get('phrase_type', '')
        }
        formatted_data.append(formatted_item)
        all_answers.append(item['answer'])
    
    # Set random seed for reproducibility
    random.seed(seed)
    random.shuffle(formatted_data)
    
    # Split into train and test
    split_idx = int(len(formatted_data) * train_ratio)
    train_data = formatted_data[:split_idx]
    test_data = formatted_data[split_idx:]
    
    print(f"\n2. Splitting dataset:")
    print(f"   Train samples: {len(train_data)}")
    print(f"   Test samples: {len(test_data)}")
    
    # Create answer list from all unique answers
    # Count answer frequencies
    # Convert all answers to strings to handle mixed types
    all_answers_str = [str(ans) for ans in all_answers]
    answer_counter = Counter(all_answers_str)
    answer_list = sorted(answer_counter.keys())  # Alphabetically sorted
    
    print(f"\n3. Creating answer vocabulary:")
    print(f"   Total unique answers: {len(answer_list)}")
    print(f"   Most common answers:")
    for answer, count in answer_counter.most_common(10):
        print(f"      '{answer}': {count} times")
    
    # Save train set
    train_file = os.path.join(output_dir, 'trainset.json')
    print(f"\n4. Saving trainset to: {train_file}")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    # Save test set
    test_file = os.path.join(output_dir, 'testset.json')
    print(f"   Saving testset to: {test_file}")
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    # Save answer list
    answer_file = os.path.join(output_dir, 'answer_list.json')
    print(f"   Saving answer list to: {answer_file}")
    with open(answer_file, 'w', encoding='utf-8') as f:
        json.dump(answer_list, f, indent=2, ensure_ascii=False)
    
    # Also create answer_all_list.json (sometimes referenced in configs)
    answer_all_file = os.path.join(output_dir, 'answer_all_list.json')
    print(f"   Saving answer_all_list to: {answer_all_file}")
    with open(answer_all_file, 'w', encoding='utf-8') as f:
        json.dump(answer_list, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics:")
    print("=" * 60)
    
    # Question type distribution
    train_qtypes = Counter([item['question_type'] for item in train_data])
    test_qtypes = Counter([item['question_type'] for item in test_data])
    
    print("\nQuestion Types (Train):")
    for qtype, count in train_qtypes.most_common():
        print(f"  {qtype}: {count}")
    
    print("\nAnswer Types (Train):")
    train_atypes = Counter([item['answer_type'] for item in train_data])
    for atype, count in train_atypes.most_common():
        print(f"  {atype}: {count}")
    
    print("\n" + "=" * 60)
    print("✓ Preprocessing complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  - {train_file}")
    print(f"  - {test_file}")
    print(f"  - {answer_file}")
    print(f"  - {answer_all_file}")
    print("\nNext steps:")
    print("  1. Update configs/VQA.yaml with the correct paths")
    print("  2. Run training with: python train_vqa.py --dataset_use rad")
    print("=" * 60)
    
    return {
        'train_size': len(train_data),
        'test_size': len(test_data),
        'num_answers': len(answer_list)
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess VQA-RAD dataset for MUMC')
    parser.add_argument('--input_json', type=str, 
                        default='ofs_dataset/VQA_RAD Dataset Public.json',
                        help='Path to the original VQA-RAD JSON file')
    parser.add_argument('--output_dir', type=str, 
                        default='ofs_dataset',
                        help='Directory to save processed files')
    parser.add_argument('--train_ratio', type=float, 
                        default=0.8,
                        help='Ratio of data for training (default: 0.8)')
    parser.add_argument('--seed', type=int, 
                        default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Run preprocessing
    stats = preprocess_vqa_rad(
        input_json=args.input_json,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
