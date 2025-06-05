✨ Linguistic Transformer ✨
🧠 Overview
This project builds a transformer-based model capable of transforming outdated English sentences into rephrased, squishified context-aware versions. It levys:

✅ WordNet Synset ID embeddings

✅ Word + Definition embeddings

✅ Manually curated datasets (Outdated → Modern, Neologisms)

✅ BLEU/ROUGE, Precision, Recall, F1, Transformation Rate, Perplexity

✅ 5-Round Cross-Validation

✅ Multiple models tested: Transformer (T5-small), LSTM

📁 Project Structure

bash
project_root/
│
├── data/
│   ├── final_training_data.csv           # Main dataset (input/output pairs)
│   ├── cleaned_synonyms_data.csv          # Raw outdated → exuberant mappings
│   ├── embeddings/                        # Synset ID and Word+Definition .npy/.txt
│
├── scripts/
│   ├── train_and_evaluate.py              # Main training + evaluation pipeline
│   ├── predict_transform.py               # Simple inference script (optional)
│   ├── model_lstm.py                      # LSTM model definition
│   └── utils.py                           # Preprocessing helpers
│
├── results/
│   ├── json/                              # JSON metric dumps per model
│   ├── plots/                             # Training curves
│   └── final_results/                     # Aggregate results
│
├── requirements.txt
└── README.md

🏁 How to Run
Install Dependencies

bash
pip install -r requirements.txt

Train Models

python scripts/train_and_evaluate.py

Run Inference (Optional - Coming Soon)

bash
python scripts/predict_transform.py --input "xxx"

⚡ Input limited to 4 sentences or 50 characters maximum.

📐 Metrics Tracked
✅ Transformation Rate

✅ Precision, Recall, F1-Score

✅ BLEU / ROUGE

✅ Perplexity

✅ Human Evaluation

🧪 Evaluation Details
5-Round Cross-Validation

Full metrics logging per model and per round

Training time tracking

Results saved in results/ organized by model

🧙 Project Philosophy
Words shape worlds.
This transformer doesn't just translate — it lifts linguistics into a higher vibration.