# 🇬🇭 Twi ↔ English Neural Machine Translation — Training Notebooks

> Training notebooks, experiments, and iteration logs for [`ninte/twi-en-marianmt`](https://huggingface.co/ninte/twi-en-marianmt) and its planned successors.  
> This repository documents the full journey from a MarianMT v1 baseline toward a production-quality Twi–English translation system.

---

## 📌 About the Project

This project aims to build and continuously improve a **bidirectional Twi ↔ English neural machine translation model**, addressing the severe lack of NLP tooling for Akan/Twi — one of the most widely spoken languages in Ghana and across the Ghanaian diaspora.

The current published model ([`ninte/twi-en-marianmt`](https://huggingface.co/ninte/twi-en-marianmt)) is a **v1 baseline** fine-tuned from `Helsinki-NLP/opus-mt-mul-en` on the [GhanaNLP TWI–ENGLISH Parallel Text](https://huggingface.co/datasets/Ghana-NLP/TWI_ENGLISH_PARALLEL_TEXT) dataset. This repo tracks all experiments and will house subsequent, significantly improved versions.

---

## 🗂️ Repository Structure

```
twi-en-nmt-notebooks/
│
├── v1_marianmt/
│   └── twi_en_marianmt_training.ipynb     # Full v1 training notebook (baseline)
│
├── v2_nllb/
│   └── twi_en_nllb_finetune.ipynb         # NLLB-200 fine-tuning (in progress)
│
├── data/
│   ├── cleaning/
│   │   └── data_cleaning.ipynb            # Dataset noise analysis & cleaning
│   └── augmentation/
│       └── back_translation.ipynb         # Back-translation augmentation
│
├── evaluation/
│   └── eval_metrics.ipynb                 # BLEU, chrF, TER evaluation suite
│
└── README.md
```

---

## 🧪 v1 Baseline — MarianMT

### Model Card
| Field | Value |
|---|---|
| 🤗 Model | [`ninte/twi-en-marianmt`](https://huggingface.co/ninte/twi-en-marianmt) |
| Base model | `Helsinki-NLP/opus-mt-mul-en` |
| Task | Bidirectional Twi ↔ English translation |
| Dataset | [`Ghana-NLP/TWI_ENGLISH_PARALLEL_TEXT`](https://huggingface.co/datasets/Ghana-NLP/TWI_ENGLISH_PARALLEL_TEXT) |
| License | CC-BY-NC-4.0 |

### Training Configuration
| Parameter | Value |
|---|---|
| Total pairs | 10,370 |
| Train / Val / Test | 80% / 10% / 10% |
| Epochs | 5 |
| Batch size | 16 |
| Learning rate | 5e-5 |
| Warmup steps | 200 |
| Weight decay | 0.01 |
| Max sequence length | 128 tokens |
| Precision | fp16 (mixed) |
| Optimizer | AdamW |
| Hardware | NVIDIA T4 (Google Colab) |
| Training time | ~25.6 min |

### Evaluation Results (Held-out Test Set, 1,037 pairs)

| Direction | BLEU | chrF | TER |
|---|---|---|---|
| Overall (mixed) | 8.26 | 30.36 | 89.90 |
| Twi → English | 10.50 | 32.30 | 87.64 |
| English → Twi | 6.32 | 28.74 | 91.55 |

> **Note:** BLEU scores in the 8–11 range are well-documented and expected for low-resource language pairs like Twi–English. The GhanaNLP Khaya model reports similar ranges on this dataset. chrF (28–32) is a more reliable metric for morphologically rich languages like Twi.

### Training Loss Curve

| Epoch | Train Loss | Val BLEU | chrF |
|---|---|---|---|
| 1 | 4.32 | 13.25 | 15.50 |
| 2 | 3.59 | 19.85 | 22.63 |
| 3 | 3.39 | 22.68 | 25.93 |
| 4 | 3.01 | 32.39 | 32.40 |
| 5 | 2.94 | 42.14 | 32.30 |

---

## 🗺️ Roadmap

The v1 model has clear, addressable limitations. The following improvements are planned and will each have dedicated notebooks in this repo.

### 1. 🔁 NLLB-200 Fine-Tune (`v2`)
Re-train on [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M), which natively supports Twi (`twi_Latn`). Expected to yield substantially higher BLEU and chrF.

### 2. 🔀 Direction Separation
Train dedicated `twi→en` and `en→twi` models using clean, direction-filtered data. Eliminates the mixed-direction noise causing the current model to occasionally output the wrong language (v1 English→Twi BLEU: 6.32 vs 10.50 for Twi→English).

### 3. 🧹 Data Cleaning
Remove pairs where source and target languages are swapped, deduplicate the corpus, and filter low-quality sentence pairs. The current training data contains noisy examples where English text appears in the Twi column and vice versa.

### 4. 📈 Back-Translation Augmentation
Use back-translation to expand the training corpus beyond ~10,370 pairs, particularly for underrepresented domains (medical, legal, agricultural).

### 5. ⏱️ Extended Training
The v1 model shows no clear plateau at epoch 5. Additional epochs at a reduced learning rate (`2e-5`) are expected to yield further chrF improvements.

---

## ⚠️ Known Limitations (v1)

- **Mixed-direction training** — no explicit direction tags (`>>en<<`) cause occasional wrong-language output
- **Low-resource constraints** — ~8,300 training pairs; model may hallucinate on complex or domain-specific inputs
- **Repetition artifacts** — use `no_repeat_ngram_size=3` in generation config to mitigate
- **Dialectal variation** — dataset skews toward Asante Twi; Akwapim Twi may produce degraded output
- **Special characters** — Twi characters `ɛ`, `ɔ`, `ŋ` are critical; inputs missing them degrade translation quality
- **Dataset noise** — directional mislabeling in the source dataset contributes to lower English→Twi scores

---

## 🚀 Quick Start

### Load the v1 Model

```python
from transformers import MarianMTModel, MarianTokenizer

model_name = "ninte/twi-en-marianmt"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model.generate(
        **inputs,
        num_beams=4,
        max_length=128,
        no_repeat_ngram_size=3,
        early_stopping=True,
        length_penalty=1.0,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Twi → English
print(translate("Ɛhɔ na ɔkɔɔ."))

# English → Twi
print(translate("The child is going to school."))
```

---

## 📦 Dataset

Training uses the [GhanaNLP TWI–ENGLISH Parallel Text](https://huggingface.co/datasets/Ghana-NLP/TWI_ENGLISH_PARALLEL_TEXT):
- **14,875** professionally translated sentence pairs
- Covers civic life, health, agriculture, education, and culture
- Sources include Wikipedia, novels, news outlets, and local linguists
- Dialectal diversity: Asante Twi, Akwapim Twi, and other Akan variants
- Funded by Google LLC; translated by independent paid professionals

---

## 📜 Citation

If you use this work in research or applications, please cite the underlying dataset:

```bibtex
@dataset{ghananlp_twi_english_2023,
  author    = {GhanaNLP},
  title     = {GhanaNLP TWI-ENGLISH Parallel Text},
  year      = {2023},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/Ghana-NLP/TWI_ENGLISH_PARALLEL_TEXT}
}
```

---

## 🙏 Acknowledgements

- Dataset funded by **Google LLC**
- Professional translations by independent paid translators in Ghana
- Trained on **Google Colab** infrastructure
- Built as part of an initiative to expand NLP tooling for Ghanaian and West African languages

---

## 📄 License

[Creative Commons Attribution Non-Commercial 4.0 (CC-BY-NC-4.0)](https://creativecommons.org/licenses/by-nc/4.0/)

Free for non-commercial use with attribution. Commercial use requires explicit permission from the author.