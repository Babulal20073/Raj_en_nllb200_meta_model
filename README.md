English–Rajasthani Machine Translation (Low-Resource)
🔍 Overview

This project focuses on building a machine translation system for English ↔ Rajasthani, a low-resource language with very limited parallel data. Since Rajasthani is not directly supported in most multilingual models, this project explores how far we can go using data augmentation, filtering, and efficient fine-tuning techniques.

The system is built using Meta’s NLLB-200 model, along with multiple preprocessing and training strategies to improve performance under constraints.

⚙️ Why this project?

Most machine translation systems rely on large datasets, but languages like Rajasthani lack such resources.

This project tries to answer:

Can we build a working translation system with limited data?
Can synthetic data compensate for missing real data?
How much improvement can we get using efficient fine-tuning?
📥 Input Data
Source Language: Rajasthani (Devanagari script)
Target Language: English
Initial Dataset Size: ~1800 parallel sentence pairs

Since Rajasthani is not supported in NLLB-200, we use:

hin_Deva (Hindi) as a proxy language

This allows the model to process Rajasthani text due to script and linguistic similarity.

🧹 Data Processing Pipeline
1. Data Cleaning
Removed empty and duplicate lines
Normalized whitespace and punctuation
Ensured proper sentence alignment
2. Zero-Shot Translation
Used NLLB-200 to generate initial translations
Created a baseline synthetic dataset
3. Back-Translation
Translated English → Rajasthani
Increased dataset diversity and size
4. LaBSE Filtering
Used sentence embeddings
Removed noisy or weakly aligned pairs
Ensured semantic consistency
5. Final Dataset
Combined:
original data
synthetic data
filtered pairs
Final size: ~3500–4000 sentence pairs
🧠 Model & Training
Base Model
facebook/nllb-200-distilled-600M
Fine-Tuning Method
LoRA (Low-Rank Adaptation)
Trained only ~0.38% of parameters
Reduced GPU memory usage
Enabled training on limited hardware (RTX 4050)
Key Parameters
Max sequence length: 160
Batch size: 4
Gradient accumulation: 4
Epochs: 5
Learning rate: 2e-4
📊 Evaluation
Metrics Used
BLEU → word-level accuracy
chrF → character-level similarity
Results
Model	BLEU	chrF
Zero-Shot NLLB	4.47	25.0
LoRA Fine-Tuned	7.93	31.0
📈 Interpretation of Results

The scores may appear low compared to high-resource translation systems, but this is expected because:

Very small dataset (~1.8k → ~3.5k)
Rajasthani is not directly supported
Synthetic data introduces noise
Limited computational resources

However, the important point is:

👉 There is a clear improvement from zero-shot to fine-tuned model
👉 The pipeline successfully adapts the model to a new low-resource language

🖥️ Demo (Gradio)

A simple interface is built using Gradio to test translations in real time.

Features:

Input Rajasthani sentence
Get English translation
Helps in qualitative evaluation
🚀 Key Learnings
Data quality matters more than model size in low-resource settings
Synthetic data + filtering can significantly help
LoRA is effective for training large models on small GPUs
Evaluation metrics must be interpreted carefully
⚠️ Limitations
Limited dataset size
Proxy language (Hindi) introduces bias
Not suitable for long or complex sentences
Performance can vary across dialects
🔮 Future Improvements
Increase dataset to 10k–20k+
Use better filtering thresholds
Try full fine-tuning with larger GPUs
Add human validation
Explore multilingual pivot (Hindi ↔ Rajasthani ↔ English)
