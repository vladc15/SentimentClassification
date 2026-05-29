# Sentiment Classification — Code

This repository contains the code for our CIL 2026 project on five-class sentiment classification of English and German product reviews. This README documents the codebase and how to reproduce the results.

---

## 1. Repository layout

| File / directory                      | What it does                                                                                                                                                |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset_analysis.ipynb`              | Class balance, language detection (EN/DE), length analysis, wordclouds, per-class bigrams/trigrams, noisy-sample inspection, preprocessing checks.          |
| `svm_linear_word2vec_cbow.ipynb`      | Baseline: CBOW Word2Vec + calibrated `LinearSVC`. Reports accuracy, score, classification report, confusion matrix.                                         |
| `svm_linear_word2vec_skip_gram.ipynb` | Same baseline pipeline with Skip-gram Word2Vec embeddings.                                                                                                  |
| `main.ipynb`                          | Fine-tunes the multilingual sentiment BERT (`nlptown/bert-base-multilingual-uncased-sentiment`); writes a checkpoint and a submission CSV.                  |
| `melectra_sep_title_text.ipynb`       | Title/body ablation: fine-tunes `AILabTUL/mELECTRA` with separated title and body inputs.                                                                   |
| `ensemble.ipynb`                      | Local version of the cross-model ensemble (BERT + XLM-RoBERTa).                                                                                             |
| `ensemble-kaggle.ipynb`               | Kaggle version of the ensemble: BERT + XLM-RoBERTa-large + ELECTRA + DistilBERT; averaging, weighted averaging, MLP stacking, expected-value thresholding.  |
| `final_ensemble.ipynb`                | Final 10-block (concat + sep) ensemble with weighted averaging, holdout MLP, K-fold MLP, multi-seed K-fold MLP. Produces the leaderboard submissions.       |
| `ensemble_level_analysis.ipynb`       | Ensemble error analysis: per-model F1/accuracy/score, confusion matrices, confidence calibration, disagreement, error magnitude, most-confused class pairs. |
| `roberta_confusion_matrix.ipynb`      | Standalone confusion-matrix + classification-report analysis for XLM-RoBERTa-large alone.                                                                   |
| `lime_roberta.ipynb`                  | LIME explanations of XLM-RoBERTa-large on correct/incorrect examples per class.                                                                             |
| `integrated_gradients_roberta.ipynb`  | Captum Integrated Gradients (tokenlevel attributions) for XLM-RoBERTa-large.                                                                                |

---

## 2. Running on Kaggle (primary path)

Most experiments were developed and executed on Kaggle so that the cached encodings, checkpoints, and probability arrays could be shared across notebooks without re-running expensive jobs.

To reproduce a notebook on Kaggle:

1. Upload the notebook to Kaggle and enable a GPU (2x T4 is enough for inference and the SVM/MLP work; the XLM-RoBERTa-large fine-tunes were run on P100/A100 or on a cluster).
2. Attach the following inputs (the slugs are the ones hardcoded in the notebooks — fork them if you need your own copy):
   - **Data:** `vladcvx/sentiment-classification-data-256` — train/test CSV plus cached BERT/ELECTRA/DistilBERT encodings.
   - **Cached RoBERTa encodings:** `vladcvx/sentiment-classification-data-256-roberta`.
   - **Cached val/test probability arrays:** `vladcvx/bert-roberta-val-test-probs` (used by `ensemble-kaggle.ipynb` and `final_ensemble.ipynb`).
   - **Title/body-separated probability arrays:** `vladcvx/cil-sep` (used by `final_ensemble.ipynb`).
   - **Fine-tuned model checkpoints (Kaggle model registry):** `vladcvx/cil-big-bert`, `vladcvx/cil-roberta-1-epoch`, `vladcvx/cil-roberta-5-epochs`, `vladcvx/cil-electra-5-epochs`, `vladcvx/cil-distilbert-3-epochs`.
3. Run **all cells**. Submissions are written under `/kaggle/working/` and can be submitted to the competition from there.

### Suggested execution order

```
dataset_analysis.ipynb
   ├── svm_linear_word2vec_cbow.ipynb
   ├── svm_linear_word2vec_skip_gram.ipynb
   ├── main.ipynb
   └── melectra_sep_title_text.ipynb

ensemble-kaggle.ipynb       # cross-model probability ensembles + stacking
final_ensemble.ipynb        # 10-block ensemble + leaderboard submission

ensemble_level_analysis.ipynb     # per-model + ensemble error analysis
roberta_confusion_matrix.ipynb    # single-model RoBERTa diagnostics
lime_roberta.ipynb                # LIME interpretation of RoBERTa
integrated_gradients_roberta.ipynb # IG interpretation of RoBERTa
```

Anything downstream of `final_ensemble.ipynb` only needs the cached probability arrays, so the heavy transformer fine-tunes do not have to be repeated to reproduce the ensemble or interpretability sections.

---

## 3. Running locally

You need the competition CSVs under `data/`, the fine-tuned checkpoints under `checkpoints/`, and the cached `.npy` probability arrays under `data/`. The local-friendly notebooks (`main.ipynb`, `ensemble.ipynb`) use these relative paths directly.

Minimum environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install \
  numpy pandas scikit-learn matplotlib seaborn \
  gensim wordcloud langdetect nltk \
  torch transformers sentencepiece \
  tqdm emoji captum lime
```

Then launch Jupyter and run the notebooks in the order shown above.

---

## 4. Running on a cluster

For long-running jobs (RoBERTa-large fine-tuning, multi-seed K-fold MLPs) we exported the relevant notebook to a Python script and submitted it as a Slurm job:

```bash
jupyter nbconvert --to script <notebook>.ipynb
python <notebook>.py
```

The notebooks keep configuration (paths, learning rate, batch size, epochs) at the top so that hyperparameters can be overridden by editing the generated `.py` or by patching environment variables before launch. The committed notebooks remain the source of truth.

---

## 6. Use of AI tools (disclosure)

We used AI coding assistants while developing this project. Every AI suggestion that ended up in the codebase was reviewed and edited by the authors before being committed. All design decisions (model selection, ablations, ensembling strategy, evaluation protocol, error analysis), all hyperparameter choices, and all reported numbers come from our own training, inference, and analysis runs. No ready-to-use external prediction API was used to produce any leaderboard submission.

- **Tool used:** Claude
  **Files affected:** `roberta_confusion_matrix.ipynb`
  **Purpose:** Scaffolded the standalone RoBERTa confusion-matrix notebook (data loading, tokenization, prediction loop, confusion-matrix and classification-report plots) by adapting patterns from `ensemble_level_analysis.ipynb` and `svm_linear_word2vec_cbow.ipynb`.

- **Tool used:** Claude
  **Files affected:** `ensemble_level_analysis.ipynb`, `integrated_gradients_roberta.ipynb`, `lime_roberta.ipynb`
  **Purpose:** Scaffolded the figure-plotting code in these analysis notebooks (per-model and ensemble confusion-matrix heatmaps, disagreement and error-magnitude plots, per-class IG attribution bar charts, and the LIME/Captum HTML visualisation wrappers). The underlying predictions, attributions, and qualitative interpretation were produced and validated by the authors.


- **Tool used:** Claude
  **Files affected:** `README.md`
  **Purpose:** Scaffolded the structure of this README.
