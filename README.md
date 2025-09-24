

## Retailers want to monitor product availability and shelf health in stores. Your task is to build a computer vision + AI pipeline that takes in images of retail shelves and produces short, human-readable captions describing stock conditions.

Example Input → Output:
    • Input: image of a soda aisle with half the shelf empty.
    • Output: "Shelf half-empty, missing soda bottles"
The captions should be:
    • Concise (≤ 12 words)
    • Accurate (describe products, stock levels, obvious anomalies)
    • Useful for retail staff (e.g., alerts about missing or misplaced items)

## Requirements
1. Data & Preprocessing
    • Use any open dataset (MS COCO, Grocery shelves datasets, or mock product images).
    • Preprocess images (resize, normalize).
    • If dataset is small, you may use synthetic data augmentation (rotations, crops, brightness changes).
2. Modeling
    • Start from a pretrained visual encoder (CNN like ResNet, or ViT).
    • Connect it to a lightweight decoder (LSTM, GRU, or GPT-2 small).
    • Support beam search or greedy decoding for caption generation.
    • Ensure captions are capped at 12 words.
3. Retail-Specific Logic
    • Captions should reflect stock level (e.g., “shelf full,” “shelf empty,” “half empty”).
    • Captions should mention key products/categories (e.g., “soda bottles,” “snack packets”).
    • Must handle edge cases: blurry images, partial shelves.
4. Evaluation
    • Evaluate using BLEU and ROUGE against a small set of reference captions.
    • Additionally, provide 5 human-graded examples where metrics may fail (e.g., synonyms like “chips” vs. “crisps”).
    • Discuss strengths/limitations of your approach.

## Constraints

    • Must run on a single laptop (no need for large GPUs).
    • Model training can be shallow fine-tuning; no full retraining from scratch.
    • Captions must be generated within 2 seconds per image.
    • Code must be reproducible with a single command (python caption.py image.jpg).

## Deliverables

    1. Inference Script: CLI tool to generate captions for given shelf images.
        ◦ Example: python caption.py shelf1.jpg → "Shelf half-empty, missing soda bottles"
    2. Notebook/Report:
        ◦ Model architecture & training choices.
        ◦ Examples with input images and generated captions.
        ◦ Evaluation metrics (BLEU/ROUGE + human eval).
        ◦ Error analysis (where captions fail).
    3. README: setup instructions, dependencies, how to run.

