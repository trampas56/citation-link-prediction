# Citation link prediction

My project for the NLP course at the Department of Computer Science and Engineering, University of Ioannina (2025). It was a Kaggle competition: given two papers, predict the probability that one cites the other.

For every pair of papers I build a feature vector from:

- the similarity of their abstracts, with Sentence Transformers embeddings (all-MiniLM-L6-v2)
- TF-IDF and Jaccard similarity of the text
- the citation graph: common neighbours, Adamic-Adar, and the differences in degree, PageRank and clustering
- Jaccard similarity of the authors
- the difference in text length

The negative pairs are sampled so there are as many as the positive ones. The prediction is the average of an XGBoost and a CatBoost classifier, and I checked it with the validation AUC.

## Running it

The notebook was made for Google Colab (CatBoost runs on the GPU) and reads the competition files (`abstracts.txt`, `authors.txt`, `edgelist.txt`, `test.txt`) from Google Drive, in `/content/drive/MyDrive/data_nlp`. The data isn't in the repo, it's on the competition's Kaggle page. The predictions are saved to `predictions.csv` in the same folder.

`citations_model_new.py` is the notebook exported as a script.

The report (in Greek) is in [report.pdf](report.pdf).
