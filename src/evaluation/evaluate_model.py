
import torch
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge_score import rouge_scorer
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

from src.preprocessing.data_loader import create_data_loaders
from src.modeling.caption_model import RetailShelfCaptioner

def evaluate_model(model_path, data_dir):
    """Evaluate model using BLEU and ROUGE metrics"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = RetailShelfCaptioner(checkpoint['vocab_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load data
    _, _, test_loader, _, _, idx_to_word = create_data_loaders(data_dir, batch_size=1)

    # Initialize metrics
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    all_references = []
    all_hypotheses = []
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    print("Evaluating model...")

    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['image'].to(device)
            true_captions = batch['caption_text']

            # Generate caption
            generated_tokens = model(images)

            # Convert to text
            generated_words = []
            for token_idx in generated_tokens[0]:
                word = idx_to_word[token_idx.item()]
                if word == '<END>':
                    break
                if word not in ['<START>', '<PAD>', '<UNK>']:
                    generated_words.append(word)

            generated_caption = ' '.join(generated_words)
            true_caption = true_captions[0]

            # Store for BLEU calculation
            all_references.append([true_caption.split()])
            all_hypotheses.append(generated_caption.split())

            # Calculate ROUGE scores
            rouge_scores_batch = rouge_scorer_obj.score(true_caption, generated_caption)
            for key in rouge_scores:
                rouge_scores[key].append(rouge_scores_batch[key].fmeasure)

    # Calculate BLEU scores
    bleu1 = corpus_bleu(all_references, all_hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(all_references, all_hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(all_references, all_hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(all_references, all_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    # Calculate average ROUGE scores
    avg_rouge_scores = {key: np.mean(scores) for key, scores in rouge_scores.items()}

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
    print(f"ROUGE-1: {avg_rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {avg_rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {avg_rouge_scores['rougeL']:.4f}")

    return {
        'bleu1': bleu1, 'bleu2': bleu2, 'bleu3': bleu3, 'bleu4': bleu4,
        'rouge1': avg_rouge_scores['rouge1'],
        'rouge2': avg_rouge_scores['rouge2'],
        'rougeL': avg_rouge_scores['rougeL']
    }

def human_evaluation_examples():
    """Provide 5 examples where metrics might fail"""

    examples = [
        {
            "image": "shelf_sparse_snacks_001.jpg",
            "reference": "Snack shelf sparse, only few items",
            "generated": "Chip aisle low, minimal products available",
            "issue": "Synonyms: 'snack' vs 'chip', 'sparse' vs 'low' - semantically identical but BLEU penalizes"
        },
        {
            "image": "shelf_empty_beverages_002.jpg",
            "reference": "Beverage shelf completely empty, urgent restock",
            "generated": "Drink section vacant, needs immediate restocking",
            "issue": "Word order and synonyms: 'beverage'/'drink', 'empty'/'vacant', 'urgent'/'immediate'"
        },
        {
            "image": "shelf_full_cereal_003.jpg",
            "reference": "Cereal shelf completely stocked with boxes",
            "generated": "Breakfast aisle fully supplied with packages",
            "issue": "Category synonyms: 'cereal'/'breakfast', 'boxes'/'packages', 'stocked'/'supplied'"
        },
        {
            "image": "shelf_messy_mixed_004.jpg",
            "reference": "Shelf disorganized with scattered mixed products",
            "generated": "Aisle untidy, various items fallen over",
            "issue": "Different but accurate descriptions of same messy state"
        },
        {
            "image": "shelf_overstocked_dairy_005.jpg",
            "reference": "Dairy section overstocked, products stacked high",
            "generated": "Milk aisle packed, items piled up",
            "issue": "Specific vs general terms: 'dairy'/'milk', 'overstocked'/'packed', 'stacked'/'piled'"
        }
    ]

    print("\n" + "="*60)
    print("HUMAN EVALUATION EXAMPLES - Where Metrics May Fail")
    print("="*60)

    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}: {example['image']}")
        print(f"Reference:  {example['reference']}")
        print(f"Generated:  {example['generated']}")
        print(f"Issue:      {example['issue']}")

    return examples

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate retail shelf captioning model')
    parser.add_argument('--model_path', type=str, default='./models/checkpoints/best_model.pth')
    parser.add_argument('--data_dir', type=str, default='./data')

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_model(args.model_path, args.data_dir)

    # Show human evaluation examples
    human_evaluation_examples()
