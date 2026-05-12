# PHIDA

(c) 2026 Naoki Masuyama

PHIDA is an online clustering method that first creates IDA-style nodes and then reads out final clusters from node-level 0-dimensional persistent homology.

## Requirements

- Python 3.10 or later
- numpy
- scikit-learn
- networkx
- joblib

Install the required packages with your preferred Python environment, for example:

```bash
pip install numpy scikit-learn networkx joblib
```

## Main Files

- `mainPHIDA.py`: example runner for stationary and nonstationary experiments.
- `phida.py`: core PHIDA model.
- `ph_view_builder_phida.py`: persistent-homology graph construction utilities.
- `continual_metrics.py`: average incremental and backward-transfer metrics.

## Data

`mainPHIDA.py` downloads datasets from OpenML through `sklearn.datasets.fetch_openml`. The default example uses the Iris dataset (`OpenML data_id=61`). OptDigits (`OpenML data_id=28`) is also listed in the script as a copy-paste dataset option.

To use another OpenML dataset, change `DATASET_NAME` in `mainPHIDA.py`. If the dataset has a known stable OpenML `data_id`, add it to `OPENML_DATASETS`; otherwise the runner tries to fetch by name. The example runner expects numeric features and classification labels.

## Run

```bash
python mainPHIDA.py
```

Set `SETTING = "stationary"` to randomly permute all samples and give them to PHIDA in one fit call. Set `SETTING = "nonstationary"` to split samples by class and give them class-by-class in random class order.

The script prints the number of nodes, number of clusters, runtime, ARI, and AMI. In the nonstationary setting it also prints average incremental scores and backward transfer.

## Citation

If you use this code, please cite:

N. Masuyama, Y. Nojima, S. Wermter, Y. Toda, H. Ishibuchi, and C. K. Loo,
"PHIDA: Persistence-Guided Node-to-Cluster Mapping for Online Clustering,"
arXiv preprintarXiv:2605.08673 [cs.LG], 2026.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
