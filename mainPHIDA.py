# Copyright (c) 2026 Naoki Masuyama
# SPDX-License-Identifier: MIT
#
# This file is part of PHIDA.
# Licensed under the MIT License; see LICENSE in the project root for details.

import hashlib
import random
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.datasets import fetch_openml
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

from continual_metrics import compute_continual_metrics
from phida import PHIDA


# Dataset examples:
# DATASET_NAME = "Iris"       # OpenML data_id=61
# DATASET_NAME = "OptDigits"  # OpenML data_id=28
DATASET_NAME = "Iris"
N_TRIALS = 30

# Experiment setting:
# - "stationary": all samples are randomly permuted and given to PHIDA in one fit call.
# - "nonstationary": samples are split by class and given class-by-class in random class order.
SETTING = "nonstationary"

OPENML_DATASETS = {
    "iris": {"data_id": 61},
    "optdigits": {"data_id": 28},
}


def _openml_fetch_kwargs(dataset_name: str):
    key = str(dataset_name).strip()
    if not key:
        raise ValueError("dataset_name must be a non-empty string.")

    spec = OPENML_DATASETS.get(key) or OPENML_DATASETS.get(key.lower())
    if spec is None:
        spec = {"name": key}

    fetch_kwargs = {"as_frame": False}
    fetch_kwargs.update(spec)
    return fetch_kwargs


def load_openml_dataset(dataset_name: str):
    dataset = fetch_openml(**_openml_fetch_kwargs(dataset_name))
    x = np.asarray(dataset.data)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    try:
        x = x.astype(np.float64)
    except ValueError as exc:
        raise ValueError(
            "PHIDA's runner expects numeric OpenML features. "
            "Choose a numeric dataset or add preprocessing before calling PHIDA."
        ) from exc

    if not np.all(np.isfinite(x)):
        raise ValueError("The selected OpenML dataset contains missing or non-finite feature values.")

    y_raw = np.asarray(dataset.target).reshape(-1)
    if y_raw.size != x.shape[0]:
        raise ValueError("OpenML target length does not match the number of samples.")
    if any(label is None or str(label).lower() == "nan" for label in y_raw):
        raise ValueError("The selected OpenML dataset contains missing target labels.")

    y = LabelEncoder().fit_transform(y_raw.astype(str)).astype(int)
    return x, y


def fmt_mean_std(values):
    v = np.asarray(values, dtype=float)
    return f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}"


def fmt_mean_std_1(values):
    v = np.asarray(values, dtype=float)
    return f"{np.mean(v):.1f} ± {np.std(v, ddof=1):.1f}"


def idat_protocol_class_split_seed(dataset_name: str) -> int:
    key = f"{dataset_name}::0".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def prepare_class_splits(x_train, y_train, seed=0):
    rng = np.random.RandomState(seed)
    classes = np.unique(y_train)

    class_data = []
    class_target = []
    for c in classes:
        idx = np.where(y_train == c)[0]
        idx = idx[rng.permutation(len(idx))]
        class_data.append(x_train[idx])
        class_target.append(y_train[idx])

    return classes, class_data, class_target


def get_num_nodes(model):
    flags = getattr(model, "is_weight_", None)
    if flags is not None:
        flags = np.asarray(flags, dtype=bool).reshape(-1)
        return int(np.sum(flags))
    return int(len(model.G_.nodes()))


def run_trial(
    trial_seed,
    x_train,
    y_train,
    setting="stationary",
    classes=None,
    class_data=None,
    class_target=None,
):
    np.random.seed(trial_seed)
    random.seed(trial_seed)

    if setting == "stationary":
        p = np.random.permutation(len(x_train))
        x = x_train[p]
        y = y_train[p]

        model = PHIDA()

        start = time.perf_counter()
        model.fit(x)
        elapsed = time.perf_counter() - start

        y_pred = model.predict(x)

        score_ari = adjusted_rand_score(y, y_pred)
        score_ami = adjusted_mutual_info_score(y, y_pred)

        n_nodes = get_num_nodes(model)
        n_clusters = int(model.n_clusters_)

        avg_inc_ari = None
        bwt_ari = None
        avg_inc_ami = None
        bwt_ami = None

        return (
            n_nodes,
            n_clusters,
            elapsed,
            score_ari,
            score_ami,
            avg_inc_ari,
            bwt_ari,
            avg_inc_ami,
            bwt_ami,
        )

    if setting != "nonstationary":
        raise ValueError(f"Unknown setting: {setting}. Use 'stationary' or 'nonstationary'.")

    if classes is None or class_data is None or class_target is None:
        raise ValueError("For nonstationary, classes/class_data/class_target must be provided.")

    num_classes = len(classes)
    random_class_order = np.random.permutation(num_classes)

    model = PHIDA()

    ari_steps = np.zeros(num_classes, dtype=float)
    ami_steps = np.zeros(num_classes, dtype=float)

    time_total = 0.0
    eval_data = None
    eval_target = None

    for step_idx in range(num_classes):
        cpos = random_class_order[step_idx]
        data_c = class_data[cpos]
        target_c = class_target[cpos]

        tic = time.perf_counter()
        model.fit(data_c)
        time_total += time.perf_counter() - tic

        if step_idx == 0:
            eval_data = data_c
            eval_target = target_c
            ari_steps[step_idx] = 1.0
            ami_steps[step_idx] = 1.0
            continue

        eval_data = np.vstack((eval_data, data_c))
        eval_target = np.concatenate((eval_target, target_c), axis=0)

        predicted_inc = model.predict(eval_data)

        ari_steps[step_idx] = adjusted_rand_score(eval_target, predicted_inc)
        ami_steps[step_idx] = adjusted_mutual_info_score(eval_target, predicted_inc)

    final_ari = float(ari_steps[-1])
    final_ami = float(ami_steps[-1])

    final_num_nodes = get_num_nodes(model)
    final_num_clusters = int(model.n_clusters_)

    inc_ari, bwt_ari = compute_continual_metrics(ari_steps)
    inc_ami, bwt_ami = compute_continual_metrics(ami_steps)

    return (
        final_num_nodes,
        final_num_clusters,
        time_total,
        final_ari,
        final_ami,
        inc_ari,
        bwt_ari,
        inc_ami,
        bwt_ami,
    )


def run_trials(
    data_name,
    x_train,
    y_train,
    setting,
    classes,
    class_data,
    class_target,
    n_trials,
):
    trial_seeds = list(range(int(n_trials)))
    total_trials = int(len(trial_seeds))
    start_time = time.perf_counter()
    print(
        f"dataset-start setting={setting} dataset={data_name} "
        f"trials={total_trials}",
        flush=True,
    )

    results = Parallel()(
        delayed(run_trial)(
            trial_seed,
            x_train,
            y_train,
            setting,
            classes,
            class_data,
            class_target,
        )
        for trial_seed in trial_seeds
    )

    elapsed = time.perf_counter() - start_time
    print(
        f"dataset-finished setting={setting} dataset={data_name} "
        f"elapsed={elapsed:.2f}s",
        flush=True,
    )
    return results


def main():
    print("Start...")

    data_name = DATASET_NAME
    x_train, y_train = load_openml_dataset(data_name)

    class_split_seed = idat_protocol_class_split_seed(data_name)
    classes, class_data, class_target = prepare_class_splits(
        x_train,
        y_train,
        seed=class_split_seed,
    )

    results = run_trials(
        data_name,
        x_train,
        y_train,
        SETTING,
        classes,
        class_data,
        class_target,
        N_TRIALS,
    )

    n_nodes_list = [r[0] for r in results]
    n_clusters_list = [r[1] for r in results]
    fit_time_list = [r[2] for r in results]
    ari_list = [r[3] for r in results]
    ami_list = [r[4] for r in results]

    avg_inc_ari_list = [r[5] for r in results if r[5] is not None]
    bwt_ari_list = [r[6] for r in results if r[6] is not None]
    avg_inc_ami_list = [r[7] for r in results if r[7] is not None]
    bwt_ami_list = [r[8] for r in results if r[8] is not None]

    print("      setting:", SETTING)
    print("      dataset:", data_name)
    print("   # of Nodes:", fmt_mean_std_1(n_nodes_list))
    print("# of Clusters:", fmt_mean_std_1(n_clusters_list))
    print("     fit time:", fmt_mean_std(fit_time_list))
    print("          ARI:", fmt_mean_std(ari_list))
    print("          AMI:", fmt_mean_std(ami_list))

    if SETTING == "nonstationary":
        print("   avgInc ARI:", fmt_mean_std(avg_inc_ari_list))
        print("   avgInc AMI:", fmt_mean_std(avg_inc_ami_list))
        print("      BWT ARI:", fmt_mean_std(bwt_ari_list))
        print("      BWT AMI:", fmt_mean_std(bwt_ami_list))

    print("Finished")


if __name__ == "__main__":
    main()
