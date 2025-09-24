bash
# 1. Install dependencies
pip install torch torchvision transformers nltk rouge-score tensorboard pillow tqdm

# 2. Train the model
python src/train_model.py --epochs 30 --batch_size 16

# 3. Generate caption for an image
python caption.py data/raw/images/shelf_half_beverages_004.jpg

# 4. Evaluate the model
python src/evaluation/evaluate_model.py

# 5. Run with verbose output
python caption.py data/raw/images/shelf_empty_produce_010.jpg --verbose


cp -r /Users/macbookpro_2015/dev/python/retail_shelf_caption/* /path/to/clean-repo/
