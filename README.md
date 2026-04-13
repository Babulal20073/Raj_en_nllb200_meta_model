# English–Rajasthani Machine Translation (Low-Resource)

## 🔍 Overview
This project focuses on building a machine translation system for **English ↔ Rajasthani**, a low-resource language with very limited parallel data. Since Rajasthani is not directly supported in most multilingual models, this project explores how far we can go using data augmentation, filtering, and efficient fine-tuning techniques.

The system is built using Meta’s **NLLB-200** model, along with multiple preprocessing and training strategies to improve performance under constraints.

---

## ⚙️ Why this project?
Most machine translation systems rely on large datasets, but languages like Rajasthani lack such resources. This project tries to answer:
* Can we build a working translation system with limited data?
* Can synthetic data compensate for missing real data?
* How much improvement can we get using efficient fine-tuning?

---

## 📥 Input Data
* **Source Language:** Rajasthani (Devanagari script)
* **Target Language:** English
* **Initial Dataset Size:** ~1800 parallel sentence pairs

Since Rajasthani is not supported in NLLB-200, we use **hin_Deva (Hindi)** as a proxy language. This allows the model to process Rajasthani text due to script and linguistic similarity.

---

## 🧹 Data Processing Pipeline
1. **Data Cleaning:** Removed empty/duplicate lines and normalized whitespace.
2. **Zero-Shot Translation:** Used NLLB-200 to generate initial translations for a baseline.
3. **Back-Translation:** Translated English → Rajasthani to increase dataset diversity.
4. **LaBSE Filtering:** Used sentence embeddings to remove noisy or weakly aligned pairs.
5. **Final Dataset:** Combined original and synthetic data for a final size of **~3500–4000 pairs**.

---

## 🧠 Model & Training
* **Base Model:** `facebook/nllb-200-distilled-600M`
* **Fine-Tuning Method:** **LoRA (Low-Rank Adaptation)**
    * Trained only ~0.38% of parameters.
    * Reduced GPU memory usage (Enabled training on RTX 4050).

### Key Parameters:
| Parameter | Value |
|---|---|
| Max sequence length | 160 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Epochs | 5 |
| Learning rate | 2e-4 |

---

## 📊 Evaluation
| Model | BLEU | chrF |
| :--- | :--- | :--- |
| Zero-Shot NLLB | 4.47 | 25.0 |
| **LoRA Fine-Tuned** | **7.93** | **31.0** |

### 📈 Interpretation of Results
The scores show a clear improvement from the zero-shot to the fine-tuned model. While the numbers are lower than high-resource systems, it proves the pipeline successfully adapts the model to a new low-resource language.

---

## 🖥️ Demo (Gradio)
A simple interface built using Gradio to test translations in real time.
* Input Rajasthani sentence -> Get English translation.
* Facilitates qualitative evaluation.

---

## 🚀 Key Learnings
* **Data quality** matters more than model size in low-resource settings.
* **Synthetic data + filtering** can significantly bridge the gap.
* **LoRA** is highly effective for training large models on consumer-grade GPUs.

---

## ⚠️ Limitations
* Limited dataset size.
* Proxy language (Hindi) may introduce linguistic bias.
* Not suitable for long/complex sentences or varied dialects.

## 🔮 Future Improvements
* Increase dataset to 10k–20k+ pairs.
* Use better filtering thresholds.
* Explore multilingual pivots (Hindi ↔ Rajasthani ↔ English).
