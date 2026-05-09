# Copyright (c) 2026 Naoki Masuyama
# SPDX-License-Identifier: MIT
#
# This file is part of PHIDA.
# Licensed under the MIT License; see LICENSE in the project root for details.

import numpy as np


def compute_continual_metrics(per_metric):
    # Convert input list/array to a 1D float numpy array
    scores = np.asarray(per_metric, dtype=float).ravel()

    if scores.size == 0:
        raise ValueError("per_metric must contain at least one score.")

    # Number of phases
    C = scores.size

    # --- Average Incremental computation ---
    avg_inc = float(np.mean(scores))

    # --- Backward Transfer computation ---
    if C > 1:
        final_score = scores[-1]
        deltas = final_score - scores[:-1]
        bwt = float(np.mean(deltas))
    else:
        bwt = 0.0  # no previous phase to compare with

    return avg_inc, bwt
