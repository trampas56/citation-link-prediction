# Citation link prediction

My project for the NLP course at the Department of Computer Science and Engineering, University of Ioannina (2025). It was a Kaggle competition: given two papers, predict the probability that one cites the other.

For every pair of papers I compute 14 features:

| Group | Features |
|---|---|
| Meaning of the text | similarity and distance of the abstract embeddings, from Sentence Transformers (all-MiniLM-L6-v2) and RoBERTa |
| Words | TF-IDF cosine similarity, Jaccard similarity of the top TF-IDF terms |
| Authors | Jaccard similarity of the author sets |
| Citation graph | common neighbours, Adamic-Adar, and the differences in degree, PageRank and clustering |
| Length | difference in the number of tokens |

For training, the citations in `edgelist.txt` are the positive pairs, and I sample as many random pairs as negatives. The prediction is the average of an XGBoost and a CatBoost classifier, and I checked it with the validation AUC.

## Running it

The notebook was made for Google Colab (CatBoost runs on the GPU) and reads the competition files (`abstracts.txt`, `authors.txt`, `edgelist.txt`, `test.txt`) from Google Drive, in `/content/drive/MyDrive/data_nlp`. The data isn't in the repo, it's on the competition's Kaggle page. The predictions are saved to `predictions.csv` in the same folder.

`citations_model_new.py` is the notebook exported as a script, so it also expects Colab (it mounts Google Drive and installs packages with `!pip`).

The report (in Greek) is in [report.pdf](report.pdf).
