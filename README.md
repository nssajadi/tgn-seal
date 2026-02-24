# A Hybrid TGN-SEAL Model for Dynamic Graph Link Prediction

## Introduction

This repository implements **TGN-SEAL**, a new approach for predicting links in networks that change over time.
Predicting links in such dynamic and often sparse networks is challenging, but it’s important for real-world
applications like social networks, telecommunications, and recommendation systems.
TGN-SEAL combines **Temporal Graph Networks (TGNs)** with **local structural information** by extracting enclosing
subgraphs around candidate links. By jointly leveraging temporal patterns and topological context, TGN-SEAL improves
predictive accuracy over standard TGNs, particularly under sparse conditions.

![TGN-SEAL Architecture](figures/architecture.png)

---

## Installation

### Requirements

* Python 3.9
* All dependencies are listed in `requirements.txt`.

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Dataset and Preprocessing

TGN-SEAL has been tested on several datasets, including **CDR (Reality Mining)**, **email-Eu-core temporal**, and *
*WikiTalk temporal networks**.

### Reality Mining CDR Dataset

1. Download the dataset from [MIT Reality Mining](http://realitycommons.media.mit.edu/realitymining.html).
2. Convert `.mat` files to CSV format:

```bash
python utils/convert_reality_mining_calls_format.py
```

3. Preprocess the CSV data:

```bash
python utils/preprocess_csv_data.py --data calls
```

---

### Email-Eu-core Temporal Network

1. Download the dataset from [SNAP: Email-Eu-core temporal](https://snap.stanford.edu/data/email-Eu-core-temporal.html).
2. Preprocess the TXT data:

```bash
python utils/preprocess_txt_data.py --data email
```

**Note:** Features are saved in dense `.npy` format. Missing edge or node features are replaced with zero vectors.

---

## Model Training

Train TGN-SEAL for self-supervised link prediction:

```bash
python train_self_supervised.py \
    -d {DATASET} \
    --use_memory \
    --prefix tgn-seal \
    --embedding_module {EMBEDDING_MODULE} \
    --n_runs {NUM_RUNS} \
    --n_epoch {NUM_EPOCHS}
```

Replace placeholders (`{DATASET}`, `{EMBEDDING_MODULE}`, `{NUM_RUNS}`, `{NUM_EPOCHS}`) with your desired values.

---

## Plotting Results

After training, visualize the results:

```bash
python -m plots.plot_tgn_seal_result
```

To compare against baseline models:

```bash
python -m plots.plot_results
```
