# Copyright (c) 2026 Naoki Masuyama
# SPDX-License-Identifier: MIT
#
# This file is part of PHIDA.
# Licensed under the MIT License; see LICENSE in the project root for details.

import heapq
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import networkx as nx

from ph_view_builder_phida import (
    build_edges_by_density_ph_h0,
    compute_adaptive_feature_metric,
    transform_with_feature_metric,
    _auto_knn_edges_euclidean_pruned_symmetric,
    _select_persistence_threshold_largest_gap,
)

DEFAULT_FORCE_REFRESH_EACH_FIT_END = True
INITIAL_LAMBDA = 2
INITIAL_SIMILARITY_THRESHOLD = 0.0


def _inverse_distance_similarity(distance, alpha=1.0):
    # Convert Euclidean distance into a bounded similarity used for winner search.
    d = np.asarray(distance, dtype=float)
    a = np.asarray(alpha, dtype=float)
    return 1.0 / (1.0 + a * d)


def _safe_variance_from_welford(m2: np.ndarray, count: int, eps: float = 1.0e-12) -> np.ndarray:
    # Recover a strictly positive variance estimate from Welford statistics.
    if int(count) <= 1:
        return np.full_like(np.asarray(m2, dtype=float), 1.0, dtype=float)
    var = np.asarray(m2, dtype=float) / float(int(count) - 1)
    var = np.where(np.isfinite(var) & (var > 0.0), var, 1.0)
    return np.maximum(var, float(eps))


def _row_normalize_nonneg(w: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    # Project raw nonnegative feature scores onto row-wise probability vectors.
    ww = np.asarray(w, dtype=float)
    ww = np.where(np.isfinite(ww) & (ww >= 0.0), ww, 0.0)
    row_sum = np.sum(ww, axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0.0, row_sum, 1.0)
    ww = ww / row_sum
    ww = np.maximum(ww, float(eps))
    ww = ww / np.sum(ww, axis=1, keepdims=True)
    return ww


def _simpson_concentration_from_mass(mass: np.ndarray, clip_min: float | None = None) -> float:
    mm = np.asarray(mass, dtype=float).reshape(-1)
    mm = mm[np.isfinite(mm) & (mm > 0.0)]
    if mm.size == 0:
        conc = 1.0
    else:
        pp = mm / float(np.sum(mm))
        conc = float(np.sum(pp * pp))
    if clip_min is not None:
        conc = float(np.clip(conc, float(clip_min), 1.0))
    return conc


def _entropy_effective_count_from_prob(p: np.ndarray, fallback: float) -> float:
    pp = np.asarray(p, dtype=float).reshape(-1)
    pp = pp[np.isfinite(pp) & (pp > 0.0)]
    if pp.size == 0:
        return float(fallback)
    h = -float(np.sum(pp * np.log(pp)))
    if not np.isfinite(h):
        return float(fallback)
    return float(np.exp(h))


class PHIDA(BaseEstimator, ClusterMixin):
    # PHIDA combines online node construction with PH-constrained cluster
    # structure formation and the default final cut selection.

    def __init__(
        self,
        _force_refresh_each_fit_end=DEFAULT_FORCE_REFRESH_EACH_FIT_END,
    ):
        self._force_refresh_each_fit_end = bool(_force_refresh_each_fit_end)

        self.num_nodes_ = 0                 # Current number of cluster nodes.
        self.weight_ = None                 # Prototype vectors for each node.
        self.node_support_ = None           # Number of samples assigned to each node.
        self.is_weight_ = None              # Prediction-only active-node mask.
        self.buffer_std_ = None             # Stores standard deviations for each node (used for distance scaling).
        self.buffer_data_ = None            # Ring buffer for incremental data storage.
        self.sample_counts_ = 0             # Total number of samples processed.
        self.G = nx.Graph()                 # Main graph representing nodes.
        self.lambda_ = INITIAL_LAMBDA       # Interval used to trigger threshold recalculations.
        self.similarity_threshold_ = INITIAL_SIMILARITY_THRESHOLD  # Similarity threshold for node assignment.
        self.recalc_counter_ = 0            # Counter tracking when to recalculate lambda_/threshold.
        self.label_cluster_ = None          # Cluster labels for each node.

        # Variables used for online variance computation.
        self.mean_data_ = None
        self.M2_data_ = None
        self.count_data_ = 0

        # Node-wise per-feature online moments for adaptive subspace weighting.
        self.node_mean_ = None
        self.node_M2_ = None
        self.node_feature_weights_ = None
        self._node_feature_weights_valid_ = False

        # Parameters for the ring buffer.
        self.buffer_data_capacity_ = 0
        self.buffer_data_start_ = 0
        self.buffer_data_count_ = 0

        # Retention length for the ring buffer (keep 2*lambda_ most-recent samples).
        self.buffer_keep_len_ = 0

        self.past_ratioNC_ = 0.0
        self.current_ratioNC_ = 0.0
        self.effective_similarity_threshold_ = 0.0
        self.node_birth_step_ = None

        # Density-PH outputs.
        self.node_to_component_ = {}
        self.node_to_cluster_ = {}        # Training-time merged cluster ids (predict-consistent).
        self._node_persistence_map_ = {}
        self._cluster_persistence_map_ = {}
        self.n_ph_components_ = 0         # Raw PH component count before predict-style merge.
        self.n_clusters_ = 0
        self.num_active_nodes_ = 0        # Prediction-active node count.
        self.ph_eps_ = 0.0

        # Internal counter for periodic density-PH.
        self._num_signal_since_density_ph = 0

        # Internal storage for PH pruning report (set by builder when commit_to_model=True).
        self._ph_pruned_node_ids_ = []

        # Cached training-time adaptive metric (updated periodically from model nodes).
        self._train_metric_center_ = None
        self._train_metric_denom_ = None
        self._train_metric_gamma_ = 0.0
        self._train_metric_last_sample_count_ = -1
        self._train_metric_last_num_nodes_ = -1
        self._train_weight_metric_cache_ = None
        self._train_weight_metric_cache_num_nodes_ = -1

        # Read-only cluster view used by predict(). It is refreshed only
        # during training-time PH updates.
        self._predict_view_ = None
        self._predict_view_ready_ = False

    def _select_ph_persistence_threshold_(
        self,
        base_mode,
        mode_parent,
        mode_persistence,
        x,
        sample_weight,
    ):
        _ = base_mode
        _ = mode_parent
        _ = x
        _ = sample_weight
        return float(_select_persistence_threshold_largest_gap(mode_persistence))

    @property
    def G_(self):
        # ph_view_builder_phida expects model.G_
        return self.G

    def _invalidate_node_feature_weights_(self):
        self._node_feature_weights_valid_ = False

    def _global_feature_variance_(self):
        if self.M2_data_ is None or int(self.count_data_) <= 1:
            return None
        return _safe_variance_from_welford(self.M2_data_, int(self.count_data_))

    def _node_feature_weights_matrix_(self):
        """
        Compute node-wise feature weights from local and global feature variance.

        For node i and feature j, define
            s_ij = v_j^glob / (u_ij + v_j^glob),
        where u_ij is the node-local variance and v_j^glob is the global
        variance. Each node row is then normalized to a probability vector.
        """
        if bool(self._node_feature_weights_valid_) and self.node_feature_weights_ is not None:
            if self.weight_ is not None and self.node_feature_weights_.shape == self.weight_.shape:
                return self.node_feature_weights_

        # If the model has no active nodes yet, no node-wise metric can be formed.
        if self.weight_ is None or int(self.num_nodes_) <= 0:
            self.node_feature_weights_ = None
            self._node_feature_weights_valid_ = True
            return None

        n = int(self.weight_.shape[0])
        d = int(self.weight_.shape[1])
        if self.node_support_ is None or self.node_M2_ is None:
            self.node_feature_weights_ = np.full((n, d), 1.0 / float(d), dtype=float)
            self._node_feature_weights_valid_ = True
            return self.node_feature_weights_

        global_var = self._global_feature_variance_()
        if global_var is None:
            global_var = np.ones(d, dtype=float)

        # Estimate node-local variability; low-support nodes fall back to the
        # global variance so their weights stay conservative.
        counts_local = np.asarray(self.node_support_[:n], dtype=np.float64).reshape(-1, 1)
        denom_local = np.maximum(counts_local - 1.0, 1.0)
        local_var = np.asarray(self.node_M2_[:n, :], dtype=float) / denom_local
        valid_local = np.isfinite(local_var) & (local_var > 0.0)
        local_var = np.where(valid_local, local_var, global_var.reshape(1, -1))
        low_count_mask = counts_local[:, 0] <= 1.0
        if np.any(low_count_mask):
            local_var[low_count_mask, :] = global_var.reshape(1, -1)
        local_var = np.maximum(local_var, 1.0e-12)

        global_var_row = global_var.reshape(1, -1)
        ratio = global_var_row / np.maximum(local_var + global_var_row, 1.0e-12)
        w = _row_normalize_nonneg(ratio)

        self.node_feature_weights_ = w
        self._node_feature_weights_valid_ = True
        return self.node_feature_weights_

    def _store_predict_view_(
        self,
        node_to_component,
        comp_remap,
        active_node_ids,
        weights_metric,
        node_support_active,
        metric_center,
        metric_denom,
        pruned_node_ids=None,
    ):
        """
        Cache the read-only cluster view used by predict().

        The cached view contains:
            - PH+Ward node -> cluster labels
            - active node ids used by the view
            - transformed node prototypes for prediction
            - feature metric used to transform prediction samples
            - merged-cluster support concentration q
        """
        active_node_ids = np.asarray(active_node_ids, dtype=int).reshape(-1)
        weights_metric = np.asarray(weights_metric, dtype=np.float64)
        node_support_active = np.asarray(node_support_active, dtype=np.float64).reshape(-1)
        metric_center = np.asarray(metric_center, dtype=np.float64).reshape(-1)
        metric_denom = np.asarray(metric_denom, dtype=np.float64).reshape(-1)

        node_to_cluster = {}
        valid_pos = []
        for pos, nid in enumerate(active_node_ids.tolist()):
            nid_i = int(nid)
            if nid_i not in node_to_component:
                continue
            comp_id = int(node_to_component[nid_i])
            if comp_id not in comp_remap:
                continue
            node_to_cluster[nid_i] = int(comp_remap[comp_id])
            valid_pos.append(int(pos))

        if len(valid_pos) == 0:
            self._invalidate_predict_view_()
            return {
                "node_to_cluster": {},
                "active_node_ids": np.zeros(0, dtype=int),
                "pruned_node_ids": np.zeros(0, dtype=int),
                "weights_metric": np.zeros((0, 0), dtype=np.float64),
                "metric_center": metric_center.copy(),
                "metric_denom": metric_denom.copy(),
                "node_support_active": np.zeros(0, dtype=np.float64),
                "cluster_labels": np.zeros(0, dtype=int),
                "cluster_member_pos": tuple(),
                "merged_cluster_support": np.zeros(0, dtype=np.float64),
                "q_support_fraction": 1.0,
            }

        valid_pos = np.asarray(valid_pos, dtype=int)
        active_node_ids = active_node_ids[valid_pos]
        weights_metric = weights_metric[valid_pos, :]
        node_support_active = node_support_active[valid_pos]
        node_support_active = np.where(
            np.isfinite(node_support_active) & (node_support_active > 0.0),
            node_support_active,
            1.0,
        )

        cluster_ids_active = np.array(
            [int(node_to_cluster[int(nid)]) for nid in active_node_ids.tolist()],
            dtype=int,
        )
        cluster_labels, inv_c = np.unique(cluster_ids_active, return_inverse=True)
        merged_cluster_support = np.zeros(int(cluster_labels.size), dtype=np.float64)
        cluster_member_pos = []
        for ci in range(int(cluster_labels.size)):
            idx = np.where(inv_c == int(ci))[0].astype(int)
            cluster_member_pos.append(idx)
            if idx.size > 0:
                merged_cluster_support[int(ci)] = float(np.sum(node_support_active[idx]))

        if (not np.all(np.isfinite(merged_cluster_support))) or float(np.sum(merged_cluster_support)) <= 0.0:
            merged_cluster_support = np.ones(int(cluster_labels.size), dtype=np.float64)
        merged_cluster_support = np.maximum(merged_cluster_support, 1.0e-12)
        q_support_fraction = _simpson_concentration_from_mass(
            merged_cluster_support,
            clip_min=1.0e-6,
        )

        if pruned_node_ids is None:
            active_mask = np.zeros(int(self.num_nodes_), dtype=bool)
            active_mask[active_node_ids] = True
            pruned_node_ids = np.where(~active_mask)[0].astype(int, copy=False)
        else:
            pruned_node_ids = np.asarray(pruned_node_ids, dtype=int).reshape(-1)

        view = {
            "node_to_cluster": {int(k): int(v) for k, v in node_to_cluster.items()},
            "active_node_ids": active_node_ids.copy(),
            "pruned_node_ids": pruned_node_ids.copy(),
            "weights_metric": weights_metric.copy(),
            "metric_center": metric_center.copy(),
            "metric_denom": metric_denom.copy(),
            "node_support_active": node_support_active.copy(),
            "cluster_labels": cluster_labels.astype(int, copy=False),
            "cluster_member_pos": tuple(np.asarray(idx, dtype=int) for idx in cluster_member_pos),
            "merged_cluster_support": merged_cluster_support.copy(),
            "q_support_fraction": float(q_support_fraction),
        }
        self._predict_view_ = view
        self._predict_view_ready_ = True
        return view

    def _prepare_predict_view_inputs_(self):
        """
        Materialize the active node set and metric space used by prediction.
        """
        node_to_component = getattr(self, "node_to_component_", None)
        n_nodes = int(getattr(self, "num_nodes_", 0))
        if node_to_component is None or len(node_to_component) == 0 or n_nodes <= 0:
            return None

        if self.weight_ is None or self.node_support_ is None:
            return None

        active_node_ids = np.array(sorted(int(k) for k in node_to_component.keys()), dtype=int)
        active_node_ids = active_node_ids[(active_node_ids >= 0) & (active_node_ids < n_nodes)]
        if active_node_ids.size == 0:
            return None

        if self.is_weight_ is not None and self.is_weight_.size == n_nodes:
            weighted_ids = active_node_ids[self.is_weight_[active_node_ids].astype(bool)]
            if weighted_ids.size > 0:
                active_node_ids = weighted_ids

        weights_active = self.weight_[active_node_ids, :].astype(np.float64, copy=False)
        node_support_active = self.node_support_[active_node_ids].astype(np.float64, copy=False)
        if weights_active.shape[0] <= 0:
            return None

        metric_center, metric_denom, _ = compute_adaptive_feature_metric(weights_active)
        weights_metric = transform_with_feature_metric(weights_active, metric_center, metric_denom)
        return {
            "node_to_component": dict(node_to_component),
            "active_node_ids": active_node_ids,
            "weights_metric": weights_metric,
            "node_support_active": node_support_active,
            "metric_center": metric_center,
            "metric_denom": metric_denom,
        }

    def _build_ph_constrained_predict_view_(self, prepared):
        """
        Build the cached predict view from the PH partition followed by the
        default PH-constrained node-to-cluster mapping stage.
        """
        node_to_component = dict(prepared["node_to_component"])
        comp_persistence = dict(getattr(self, "_cluster_persistence_map_", {}))
        comp_remap = self._merge_prediction_labels_by_ph_support_ward_(
            node_to_component=node_to_component,
            active_node_ids=prepared["active_node_ids"],
            weights_metric=prepared["weights_metric"],
            node_support_active=prepared["node_support_active"],
            component_persistence=comp_persistence,
            enforce_connectivity=True,
        )
        return self._store_predict_view_(
            node_to_component=node_to_component,
            comp_remap=comp_remap,
            active_node_ids=prepared["active_node_ids"],
            weights_metric=prepared["weights_metric"],
            node_support_active=prepared["node_support_active"],
            metric_center=prepared["metric_center"],
            metric_denom=prepared["metric_denom"],
        )

    def _commit_predict_style_merge_to_model_(self):
        """
        Commit predict()-style component merging into training-time PH outputs.

        This keeps self.node_to_component_ as the raw PH partition, and sets:
            - self.node_to_cluster_ : node_id -> merged_cluster_id
            - self.n_clusters_      : merged cluster count (predict-consistent)
            - self.n_ph_components_ : raw PH component count (for inspection)
            - self._predict_view_   : cached read-only cluster view used by predict()
        """
        node_to_component = getattr(self, "node_to_component_", None)
        n_nodes = int(getattr(self, "num_nodes_", 0))
        if node_to_component is None or len(node_to_component) == 0 or n_nodes <= 0:
            self.node_to_cluster_ = {}
            self.n_ph_components_ = 0
            self.n_clusters_ = 0
            self.num_active_nodes_ = 0
            self._invalidate_predict_view_()
            return

        # Raw PH component count.
        try:
            self.n_ph_components_ = int(np.unique(np.array(list(node_to_component.values()), dtype=int)).size)
        except Exception:
            self.n_ph_components_ = 0

        prepared = self._prepare_predict_view_inputs_()
        if prepared is None:
            self.node_to_cluster_ = {}
            self.n_clusters_ = 0
            self.num_active_nodes_ = 0
            self._invalidate_predict_view_()
            return

        view = self._build_ph_constrained_predict_view_(prepared)

        self.node_to_cluster_ = dict(view["node_to_cluster"])
        self.n_clusters_ = int(np.asarray(view["cluster_labels"], dtype=int).size)
        self.num_active_nodes_ = int(np.asarray(view["active_node_ids"], dtype=int).size)

    def _invalidate_train_metric_(self):
        """
        Invalidate cached training-time adaptive metric.
        """
        self._train_metric_center_ = None
        self._train_metric_denom_ = None
        self._train_metric_gamma_ = 0.0
        self._train_metric_last_sample_count_ = -1
        self._train_metric_last_num_nodes_ = -1
        self._train_weight_metric_cache_ = None
        self._train_weight_metric_cache_num_nodes_ = -1

    def _get_train_weights_metric_(self):
        """
        Return node prototypes transformed by the cached training metric.
        The transformed matrix is cached and reused across samples until the
        metric cache is invalidated or the node count changes.
        """
        n_nodes = int(self.num_nodes_)
        if n_nodes <= 0 or self.weight_ is None or self.weight_.shape[0] == 0:
            return None

        weights_now = self.weight_[:n_nodes, :].astype(np.float64, copy=False)
        if self._train_metric_center_ is None or self._train_metric_denom_ is None:
            return weights_now

        cache = self._train_weight_metric_cache_
        if cache is not None:
            same_rows = int(self._train_weight_metric_cache_num_nodes_) == n_nodes
            same_shape = cache.shape == weights_now.shape
            if same_rows and same_shape:
                return cache

        cache = transform_with_feature_metric(
            weights_now,
            center=self._train_metric_center_,
            denom=self._train_metric_denom_,
        )
        self._train_weight_metric_cache_ = np.asarray(cache, dtype=np.float64)
        self._train_weight_metric_cache_num_nodes_ = n_nodes
        return self._train_weight_metric_cache_

    def _update_train_weights_metric_row_(self, node_idx):
        """
        Refresh one cached transformed node row after an in-place prototype update.
        """
        cache = self._train_weight_metric_cache_
        if cache is None:
            return

        s = int(node_idx)
        n_nodes = int(self.num_nodes_)
        if s < 0 or s >= n_nodes:
            return
        if int(self._train_weight_metric_cache_num_nodes_) != n_nodes:
            return
        if cache.shape != self.weight_[:n_nodes, :].shape:
            return

        if self._train_metric_center_ is None or self._train_metric_denom_ is None:
            cache[s, :] = self.weight_[s, :]
            return

        cache[s, :] = transform_with_feature_metric(
            self.weight_[s, :],
            self._train_metric_center_,
            self._train_metric_denom_,
        )

    def _invalidate_predict_view_(self):
        """
        Invalidate the cached read-only cluster view used by predict().
        """
        self._predict_view_ = None
        self._predict_view_ready_ = False

    def _refresh_train_metric_cache_(self, force=False):
        """
        Refresh training-time adaptive metric from current model nodes.

        To avoid heavy recomputation, the metric is recalculated only when:
        - forced, or
        - node count changed, or
        - at least lambda_ new samples were processed since last refresh.
        """
        n_nodes = int(self.num_nodes_)
        if n_nodes <= 0 or self.weight_ is None or self.weight_.shape[0] == 0:
            self._invalidate_train_metric_()
            return

        if not force and self._train_metric_denom_ is not None:
            same_nodes = int(self._train_metric_last_num_nodes_) == n_nodes
            interval = max(int(self.lambda_) if self.lambda_ is not None else 1, 1)
            age = int(self.sample_counts_) - int(self._train_metric_last_sample_count_)
            if same_nodes and age < interval:
                return

        weights_now = self.weight_[:n_nodes, :].astype(np.float64, copy=False)
        # Rebuild the global training metric from the current node prototypes only.
        center, denom, gamma = compute_adaptive_feature_metric(weights_now)
        denom = np.asarray(denom, dtype=np.float64).reshape(-1)
        denom = np.where(np.isfinite(denom) & (denom > 0.0), denom, 1.0)

        self._train_metric_center_ = np.asarray(center, dtype=np.float64).reshape(-1)
        self._train_metric_denom_ = denom
        self._train_metric_gamma_ = float(gamma)
        self._train_metric_last_sample_count_ = int(self.sample_counts_)
        self._train_metric_last_num_nodes_ = n_nodes

    def _transform_with_train_metric_(self, x):
        """
        Transform x with cached training-time adaptive metric.
        Identity transform is used when metric cache is unavailable.
        """
        xx = np.asarray(x, dtype=np.float64)
        if self._train_metric_center_ is None or self._train_metric_denom_ is None:
            return xx
        return transform_with_feature_metric(xx, self._train_metric_center_, self._train_metric_denom_)

    def _current_num_clusters_for_threshold(self):
        # Use the current PHIDA cluster view itself when it is available:
        # ratioNC = current_cluster_count / current_node_count.
        #
        # This stays within the ART-style threshold adaptation framework while
        # reflecting the actual node-to-cluster compression achieved by the
        # present PH grouping stage. When that view is not available yet, fall
        # back to an occupancy-based effective node count.
        n_nodes = int(self.num_nodes_)
        if n_nodes <= 1:
            return float(n_nodes)

        merged_clusters = int(getattr(self, "n_clusters_", 0))
        if merged_clusters > 0:
            return float(min(max(merged_clusters, 1), n_nodes))

        raw_components = int(getattr(self, "n_ph_components_", 0))
        if raw_components > 0:
            return float(min(max(raw_components, 1), n_nodes))

        node_to_component = getattr(self, "node_to_component_", None)
        if isinstance(node_to_component, dict) and len(node_to_component) > 0:
            try:
                unique_components = int(
                    np.unique(np.asarray(list(node_to_component.values()), dtype=int)).size
                )
            except Exception:
                unique_components = 0
            if unique_components > 0:
                return float(min(max(unique_components, 1), n_nodes))

        counts = np.asarray(self.node_support_[:n_nodes], dtype=np.float64).reshape(-1)
        p = counts / float(np.sum(counts))
        p = p[p > 0.0]
        return float(_entropy_effective_count_from_prob(p, fallback=float(n_nodes)))

    def _allow_secondary_update_(self, s1, s2):
        if int(s1) == int(s2):
            return False
        node_to_component = getattr(self, "node_to_component_", None)
        if (
            not isinstance(node_to_component, dict)
            or len(node_to_component) == 0
            or int(self.n_ph_components_) < 2
            or int(self.num_nodes_) < max(2 * int(max(self.lambda_, 1)), 8)
        ):
            return True
        c1 = node_to_component.get(int(s1))
        c2 = node_to_component.get(int(s2))
        if c1 is None or c2 is None:
            return False
        return bool(int(c1) == int(c2))

    def fit(self, samples, y=None):
        """
        Incrementally train the PHIDA cluster model on the given samples.
        Each sample is processed sequentially.
        When enough samples have been processed (lambda_), thresholds may be recalculated.
        """
        samples = samples.astype(np.float64)

        # Process each sample incrementally to update the cluster structure.
        for sample_num in range(samples.shape[0]):

            sample = samples[sample_num:sample_num + 1, :]

            self._append_to_buffer(sample)
            self._cluster_step(sample)

            self.sample_counts_ += 1
            self.recalc_counter_ += 1

            # Recalculate thresholds when due, using only past data (no lookahead)
            if (self.recalc_counter_ >= int(self.lambda_)) and (self.weight_.shape[0] > 2):
                self._refresh_train_metric_cache_(force=False)
                num_clusters_current = float(self._current_num_clusters_for_threshold())

                Lambda_new, similarity_th_new, is_incremental, ratioNC_candidate = self._calculate_lambda_decremental_direction(
                    self.buffer_std_,
                    int(self.lambda_),
                    self.similarity_threshold_,
                    self._get_from_buffer(int(self.lambda_)),
                    self.num_nodes_,
                    num_clusters_current,
                )

                if is_incremental:
                    Lambda_new, similarity_th_new, ratioNC_candidate = self._calculate_lambda_incremental_direction(
                        self.buffer_std_,
                        int(self.lambda_),
                        self.similarity_threshold_,
                        self.num_nodes_,
                        num_clusters_current,
                    )

                self.lambda_ = int(Lambda_new)
                self.similarity_threshold_ = float(similarity_th_new)
                self.past_ratioNC_ = float(ratioNC_candidate)
                self.recalc_counter_ = 0
                self._update_buffer_keep_len()

        if bool(self._force_refresh_each_fit_end) and int(self.num_nodes_) > 0 and int(self.lambda_) > 0:
            self._num_signal_since_density_ph = int(self.lambda_)
            self.__apply_density_ph_periodically_()

        return self

    def predict(self, samples):
        """
        Assign labels from the cached read-only PH+Ward cluster view built
        during training.

        The prediction score is defined by
            score_c(x) = d_min,c(x) + d_q,c(x),
        where:
            d_min,c(x) : nearest-node distance to merged cluster c
            d_q,c(x)   : radius needed to accumulate a support fraction q
                         inside merged cluster c

        The support fraction q is set directly from merged-cluster support
        shares:
            p_c = S_c / sum_r S_r
            q   = sum_c p_c^2
        so prediction depth is determined by the current support concentration
        of the cached PH+Ward clusters.
        """
        samples = samples.astype(np.float64, copy=False)

        n_nodes = int(self.num_nodes_)
        if n_nodes <= 0 or self.weight_ is None or self.weight_.shape[0] == 0:
            return np.zeros(samples.shape[0], dtype=int)

        if (not bool(self._predict_view_ready_)) or (not isinstance(self._predict_view_, dict)):
            raise RuntimeError(
                "predict view is not available; run fit() so training builds the read-only PH+Ward view before predict()."
            )

        view = self._predict_view_

        active_node_ids = np.asarray(view.get("active_node_ids", np.zeros(0, dtype=int)), dtype=int).reshape(-1)
        if active_node_ids.size == 0:
            raise RuntimeError("predict view is empty; training did not produce an active PH+Ward cluster view.")

        weights_metric = np.asarray(view.get("weights_metric", np.zeros((0, 0), dtype=np.float64)), dtype=np.float64)
        metric_center = np.asarray(view.get("metric_center", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        metric_denom = np.asarray(view.get("metric_denom", np.ones(0, dtype=np.float64)), dtype=np.float64)
        node_support = np.asarray(view.get("node_support_active", np.ones(active_node_ids.size, dtype=np.float64)), dtype=np.float64)
        cluster_member_pos = tuple(view.get("cluster_member_pos", tuple()))
        uniq_c = np.asarray(view.get("cluster_labels", np.zeros(0, dtype=int)), dtype=int).reshape(-1)
        q_support_fraction = float(view.get("q_support_fraction", 1.0))

        if weights_metric.shape[0] != active_node_ids.size or uniq_c.size == 0:
            raise RuntimeError("predict view is inconsistent; cached PH+Ward view is incomplete.")

        samples_metric = transform_with_feature_metric(samples, metric_center, metric_denom)
        k_clusters = int(uniq_c.size)
        self.n_clusters_ = int(k_clusters)
        if k_clusters <= 1:
            return np.full(samples.shape[0], int(uniq_c[0]), dtype=int)

        node_support = np.where(np.isfinite(node_support) & (node_support > 0.0), node_support, 1.0)
        cluster_score = np.full((samples_metric.shape[0], k_clusters), np.inf, dtype=np.float64)
        for ci in range(k_clusters):
            if ci >= len(cluster_member_pos):
                continue
            idx = np.asarray(cluster_member_pos[int(ci)], dtype=int).reshape(-1)
            if idx.size == 0:
                continue
            # Each merged cluster is scored from its member nodes only.
            # The score combines:
            #   1. the nearest-node distance inside the cluster, and
            #   2. the distance required to accumulate a support fraction q
            #      inside the same cluster.
            w_ci = node_support[idx].astype(np.float64, copy=False)
            w_ci = np.where(np.isfinite(w_ci) & (w_ci > 0.0), w_ci, 1.0)
            total_w = float(np.sum(w_ci))
            if total_w <= 0.0 or not np.isfinite(total_w):
                continue
            q_eff = float(q_support_fraction * total_w)
            ww_ci = weights_metric[idx, :]
            for sample_i in range(samples_metric.shape[0]):
                diff = ww_ci - samples_metric[sample_i:sample_i + 1, :]
                dist_ci_sq = np.sum(diff * diff, axis=1)
                order = np.argsort(dist_ci_sq, kind="mergesort")
                dist_sorted = dist_ci_sq[order]
                w_sorted = w_ci[order]
                dmin_sq = float(dist_sorted[0])
                if idx.size == 1:
                    cluster_score[sample_i, int(ci)] = dmin_sq
                    continue
                prefix_w = np.cumsum(w_sorted)
                if float(prefix_w[-1]) < q_eff:
                    continue
                reach_idx = int(np.argmax(prefix_w >= q_eff))
                dq_sq = float(dist_sorted[reach_idx])
                cluster_score[sample_i, int(ci)] = np.sqrt(max(dmin_sq, 0.0)) + np.sqrt(max(dq_sq, 0.0))

        out_pos = np.argmin(cluster_score, axis=1).astype(int, copy=False)
        out = uniq_c[out_pos]
        return out.astype(int, copy=False)

    def _merge_prediction_labels_by_ph_support_ward_(
        self,
        node_to_component,
        active_node_ids,
        weights_metric,
        node_support_active,
        component_persistence=None,
        enforce_connectivity=True,
    ):
        """
        Build the default model-only PH+Ward merge map from PH components.

        Let s_c be the PH-component support computed from model-side node
        supports, and let pi_c be the persistence of component c.
        Define the PH-stable component mass by:
            s_c^* = s_c * phi(pi_c)
        where the persistence contribution is saturated by
            phi(pi) = pi / (pi + pi_ref)
        with pi_ref equal to the median positive persistence over current
        components.

        From the normalized PH-stable masses, define the entropy effective
        component count:
            K_ent = exp(-sum_c p_c log p_c)
        and let K_ph be the number of PH components before merging.

        Along the deterministic PH-constrained Ward agglomeration path, choose
        the partition immediately before the largest increase in merge height
        among partitions with K >= ceil(K_ent). This uses only the entropy
        effective count of the PH-stable mass distribution and adds no extra
        hyperparameters.

        Returns a deterministic remap
            original_component_id -> merged_component_id (contiguous from 0).
        """
        active_node_ids = np.asarray(active_node_ids, dtype=int).reshape(-1)
        ww = np.asarray(weights_metric, dtype=np.float64)
        cc = np.asarray(node_support_active, dtype=np.float64).reshape(-1)
        enforce_connectivity = bool(enforce_connectivity)
        comp_persistence = {}
        if isinstance(component_persistence, dict):
            comp_persistence = {
                int(k): float(v) for k, v in component_persistence.items()
            }

        # Convert external node ids to dense row positions in the active-node matrix.
        pos_of_node = {int(nid): int(i) for i, nid in enumerate(active_node_ids.tolist())}
        comp_members = {}
        for nid, cid in node_to_component.items():
            nid_i = int(nid)
            cid_i = int(cid)
            if nid_i not in pos_of_node:
                continue
            pos = pos_of_node[nid_i]
            comp_members.setdefault(cid_i, []).append(int(pos))

        original_comp_ids = sorted(comp_members.keys())
        if len(original_comp_ids) == 1:
            return {int(original_comp_ids[0]): 0}

        pos_comp_persist = np.array(
            [float(comp_persistence.get(int(cid), 0.0)) for cid in original_comp_ids],
            dtype=np.float64,
        )
        pos_comp_persist = pos_comp_persist[
            np.isfinite(pos_comp_persist) & (pos_comp_persist > 0.0)
        ]
        pi_ref = float(np.median(pos_comp_persist)) if pos_comp_persist.size > 0 else 1.0

        cc_mass = np.zeros_like(cc, dtype=np.float64)
        for cid in original_comp_ids:
            idx = np.asarray(comp_members[int(cid)], dtype=int)
            pi_c = float(comp_persistence.get(int(cid), 1.0))
            if not np.isfinite(pi_c) or pi_c < 0.0:
                pi_c = 0.0
            pi_scale = float(pi_c / max(pi_c + pi_ref, 1.0e-12))
            cc_mass[idx] = cc[idx] * pi_scale
        if float(np.sum(cc_mass)) <= 0.0 or not np.all(np.isfinite(cc_mass)):
            cc_mass = cc.astype(np.float64, copy=True)
        cc_mass = np.where(np.isfinite(cc_mass) & (cc_mass > 0.0), cc_mass, 1.0e-12)

        support = {}
        for cid in original_comp_ids:
            idx = np.asarray(comp_members[int(cid)], dtype=int)
            support[int(cid)] = float(np.sum(cc_mass[idx]))

        k_ph = int(len(original_comp_ids))
        counts = np.array([float(support[int(c)]) for c in original_comp_ids], dtype=np.float64)
        total = float(np.sum(counts))
        p = counts / total
        p = p[p > 0.0]
        k_ent = _entropy_effective_count_from_prob(p, fallback=float(len(original_comp_ids)))
        k_target = float(k_ent)
        k_min = int(max(1, min(int(np.ceil(k_target)), k_ph)))

        comp_centroids = np.zeros((k_ph, ww.shape[1]), dtype=np.float64)
        comp_support = np.zeros(k_ph, dtype=np.float64)
        comp_index = {int(cid): int(i) for i, cid in enumerate(original_comp_ids)}

        for cid in original_comp_ids:
            idx = np.asarray(comp_members[int(cid)], dtype=int)
            w = cc_mass[idx]
            ws = float(np.sum(w))
            if ws <= 0.0 or not np.isfinite(ws):
                w = cc[idx]
                ws = float(np.sum(w))
            if ws <= 0.0 or not np.isfinite(ws):
                w = np.ones(idx.size, dtype=np.float64)
                ws = float(max(idx.size, 1))
            i = int(comp_index[int(cid)])
            comp_centroids[i, :] = np.average(ww[idx, :], axis=0, weights=w)
            comp_support[i] = float(max(ws, 1.0e-12))

        group_neighbors = {int(cid): set() for cid in original_comp_ids}
        comp_edges = _auto_knn_edges_euclidean_pruned_symmetric(
            weights=comp_centroids,
            node_support=comp_support,
        )
        for i, j in comp_edges:
            ci = int(original_comp_ids[int(i)])
            cj = int(original_comp_ids[int(j)])
            if ci == cj:
                continue
            group_neighbors[ci].add(cj)
            group_neighbors[cj].add(ci)

        group_weight = {int(cid): float(comp_support[int(comp_index[int(cid)])]) for cid in original_comp_ids}
        group_sumx = {
            int(cid): np.asarray(
                comp_centroids[int(comp_index[int(cid)])] * comp_support[int(comp_index[int(cid)])],
                dtype=np.float64,
            )
            for cid in original_comp_ids
        }

        def _ward_delta(ga: int, gb: int) -> float:
            wa = float(group_weight.get(int(ga), 0.0))
            wb = float(group_weight.get(int(gb), 0.0))
            den = float(wa + wb)
            if den <= 0.0:
                return np.inf
            mua = group_sumx[int(ga)] / wa
            mub = group_sumx[int(gb)] / wb
            diff = mua - mub
            return float((wa * wb / den) * np.dot(diff, diff))

        group_pos = {int(cid): list(pos_list) for cid, pos_list in comp_members.items()}
        group_orig = {int(cid): [int(cid)] for cid in original_comp_ids}
        group_version = {int(cid): 0 for cid in original_comp_ids}

        def _current_remap_and_labels():
            gids = sorted(group_pos.keys())
            gid_to_new = {int(gid): int(i) for i, gid in enumerate(gids)}

            labels = np.empty(active_node_ids.shape[0], dtype=int)
            remap = {}
            for gid in gids:
                gid_i = int(gid)
                new_id = int(gid_to_new[gid_i])
                idx = np.asarray(group_pos[gid_i], dtype=int)
                labels[idx] = new_id
                for ocid in group_orig[gid_i]:
                    remap[int(ocid)] = new_id

            return remap, labels, int(len(gids))

        remap0, labels0, k0 = _current_remap_and_labels()
        partition_trace = [{
            "remap": remap0,
            "labels": labels0,
            "k": int(k0),
            "merge_height": 0.0,
        }]

        merge_heap = []

        def _push_pair(ga: int, gb: int):
            ga_i = int(ga)
            gb_i = int(gb)
            if ga_i == gb_i:
                return
            if ga_i not in group_pos or gb_i not in group_pos:
                return
            a_i = min(ga_i, gb_i)
            b_i = max(ga_i, gb_i)
            if enforce_connectivity and b_i not in group_neighbors.get(a_i, set()):
                return
            delta = _ward_delta(a_i, b_i)
            if not np.isfinite(delta):
                return
            heapq.heappush(
                merge_heap,
                (float(delta), int(a_i), int(b_i), int(group_version[a_i]), int(group_version[b_i])),
            )

        if enforce_connectivity:
            for ga in sorted(group_neighbors.keys()):
                for gb in sorted(group_neighbors.get(int(ga), set())):
                    if int(gb) > int(ga):
                        _push_pair(int(ga), int(gb))
        else:
            group_ids = sorted(group_pos.keys())
            for i in range(len(group_ids) - 1):
                for j in range(i + 1, len(group_ids)):
                    _push_pair(int(group_ids[i]), int(group_ids[j]))

        while len(group_pos) > 1:
            best_pair = None
            best_delta = np.inf
            while len(merge_heap) > 0:
                delta, ga, gb, ver_a, ver_b = heapq.heappop(merge_heap)
                if ga not in group_pos or gb not in group_pos:
                    continue
                if int(group_version[ga]) != int(ver_a) or int(group_version[gb]) != int(ver_b):
                    continue
                if enforce_connectivity and gb not in group_neighbors.get(ga, set()):
                    continue
                best_pair = (int(ga), int(gb))
                best_delta = float(delta)
                break

            if best_pair is None:
                break

            # Merge the Ward-best admissible pair and record the partition after
            # each step. The final cut is selected later from this merge trace.
            keep_gid = int(best_pair[0])
            drop_gid = int(best_pair[1])
            group_pos[keep_gid] = list(group_pos[keep_gid]) + list(group_pos[drop_gid])
            group_orig[keep_gid] = list(group_orig[keep_gid]) + list(group_orig[drop_gid])
            group_weight[keep_gid] = float(group_weight[keep_gid] + group_weight[drop_gid])
            group_sumx[keep_gid] = np.asarray(group_sumx[keep_gid] + group_sumx[drop_gid], dtype=np.float64)
            del group_weight[drop_gid]
            del group_sumx[drop_gid]
            del group_pos[drop_gid]
            del group_orig[drop_gid]

            keep_nei = set(group_neighbors.get(keep_gid, set()))
            drop_nei = set(group_neighbors.get(drop_gid, set()))
            merged_nei = (keep_nei | drop_nei) - {keep_gid, drop_gid}
            merged_nei = {int(v) for v in merged_nei if int(v) in group_pos}
            group_neighbors[keep_gid] = set(merged_nei)
            if drop_gid in group_neighbors:
                del group_neighbors[drop_gid]
            for ng in merged_nei:
                if ng not in group_neighbors:
                    continue
                group_neighbors[ng].discard(drop_gid)
                if ng != keep_gid:
                    group_neighbors[ng].add(keep_gid)

            group_version[keep_gid] = int(group_version.get(keep_gid, 0)) + 1
            if drop_gid in group_version:
                del group_version[drop_gid]

            for ng in sorted(group_neighbors.get(keep_gid, set())):
                _push_pair(keep_gid, int(ng))
            if not enforce_connectivity:
                for gid in sorted(group_pos.keys()):
                    if gid != keep_gid:
                        _push_pair(keep_gid, int(gid))

            remap_now, labels_now, k_now = _current_remap_and_labels()
            partition_trace.append({
                "remap": remap_now,
                "labels": labels_now,
                "k": int(k_now),
                "merge_height": float(best_delta),
            })

            if len(group_pos) <= k_min:
                break

        eligible_idx = [i for i, entry in enumerate(partition_trace) if int(entry["k"]) >= k_min]
        if len(eligible_idx) <= 1:
            return dict(partition_trace[eligible_idx[0] if eligible_idx else -1]["remap"])
        best_idx = int(eligible_idx[0])
        best_gap = -np.inf
        best_height = -np.inf
        for pos in range(1, len(eligible_idx)):
            prev_idx = int(eligible_idx[pos - 1])
            curr_idx = int(eligible_idx[pos])
            gap = float(partition_trace[curr_idx]["merge_height"]) - float(partition_trace[prev_idx]["merge_height"])
            height = float(partition_trace[curr_idx]["merge_height"])
            if (gap > best_gap) or (np.isclose(gap, best_gap) and height > best_height):
                best_gap = gap
                best_height = height
                best_idx = prev_idx

        # Choose the partition immediately before the largest merge-height jump
        # while keeping at least k_min clusters.
        return dict(partition_trace[best_idx]["remap"])

    def _cluster_step(self, sample):
        """
        Assign a single sample to an existing node or create a new node if needed.
        After node maintenance, periodically refresh node_to_component_ by density-PH.
        """
        if self.num_nodes_ < 3:
            self._add_cluster_node(sample)

            if self.num_nodes_ == 2:
                self.buffer_std_[0, 0] = self.buffer_std_[1, 0]
        else:
            self._refresh_train_metric_cache_(force=False)
            if self._train_metric_denom_ is None or self._train_metric_center_ is None:
                weights_metric = self.weight_.astype(np.float64, copy=False)
                sample_metric = sample.astype(np.float64, copy=False)
            else:
                weights_metric = self._get_train_weights_metric_()
                sample_metric = transform_with_feature_metric(
                    sample,
                    center=self._train_metric_center_,
                    denom=self._train_metric_denom_,
                )
            diff = weights_metric - sample_metric
            global_dist = np.sqrt(np.sum(diff * diff, axis=1))
            w_feat = self._node_feature_weights_matrix_()
            if w_feat is None or w_feat.shape != diff.shape:
                distances = global_dist
            else:
                weighted_sq = np.sum(w_feat * (diff * diff), axis=1)
                d = float(max(1, diff.shape[1]))
                # Normalize the weighted quadratic form by the effective
                # dimensionality of the node-wise weights.
                keff = 1.0 / np.maximum(np.sum(w_feat * w_feat, axis=1), 1.0e-12)
                keff = np.clip(keff, 1.0, d)
                local_dist = np.sqrt(np.maximum(keff * weighted_sq, 0.0))
                gamma = float(np.clip(getattr(self, "_train_metric_gamma_", 0.0), 0.0, 1.0))
                # Use the local metric only as a positive penalty. This avoids
                # over-rewarding matches that are only locally good in a small
                # subspace, while still rejecting false neighbors that look
                # close under the global metric.
                phi = float(gamma)
                distances = global_dist + phi * np.maximum(local_dist - global_dist, 0.0)
            std_vec = np.asarray(self.buffer_std_[:, 0], dtype=np.float64)
            std_vec = np.maximum(std_vec, 1.0e-12)
            alpha = 1.0 / std_vec
            alpha = np.where(np.isfinite(alpha) & (alpha > 0.0), alpha, 1.0)
            cand_sims = _inverse_distance_similarity(distances, alpha)

            sorted_pos = np.argsort(distances)
            s1 = int(sorted_pos[0])
            s2 = int(sorted_pos[1]) if sorted_pos.size > 1 else int(s1)
            sim_s1 = float(cand_sims[int(sorted_pos[0])])
            sim_s2 = float(cand_sims[int(sorted_pos[1])]) if sorted_pos.size > 1 else float(sim_s1)

            effective_similarity_threshold = float(self.similarity_threshold_)
            self.effective_similarity_threshold_ = float(effective_similarity_threshold)

            if sim_s1 < effective_similarity_threshold:
                # The closest node is still too dissimilar, so create a new node.
                self._add_cluster_node(sample)
            else:
                # Update the winner node, and optionally update the runner-up
                # when it resonates with the sample and lies in the same PH region.
                self._update_node_with_sample(int(s1), sample)
                if (
                    sim_s2 > effective_similarity_threshold
                    and int(s2) != int(s1)
                    and self._allow_secondary_update_(int(s1), int(s2))
                ):
                    m2_next = float(max(self.node_support_[int(s2)] + 1.0, 1.0))
                    base_lr = 1.0 / m2_next
                    resonance_margin = float(max(1.0 - effective_similarity_threshold, 1.0e-12))
                    confidence = float(np.clip((sim_s2 - effective_similarity_threshold) / resonance_margin, 0.0, 1.0))
                    lr = base_lr * confidence
                    if lr > 0.0:
                        self._update_node_with_sample(
                            int(s2),
                            sample,
                            update_global_scale=False,
                            learning_rate=lr,
                        )
                self._update_is_weight_(s1=int(s1), s2=int(s2), sim_s1=float(sim_s1), sim_s2=float(sim_s2))

        self.__apply_density_ph_periodically_()

    def _refresh_density_ph_raw_(self):
        """
        Refresh only the raw density-PH partition used as input to the
        predict-time grouping stage.

        This step may physically remove nodes proposed by the isolated-node PH pruning step, but it does
        not rebuild the read-only predict view.
        """
        if self.num_nodes_ <= 0:
            self.node_to_component_ = {}
            self.n_clusters_ = 0
            self.ph_eps_ = 0.0
            self._ph_pruned_node_ids_ = []
            self._node_persistence_map_ = {}
            self._cluster_persistence_map_ = {}
            return

        node_ids = self._ph_input_node_ids_()
        if node_ids.size == 0:
            node_ids = np.arange(int(self.num_nodes_), dtype=int)

        weights = self.weight_[node_ids, :].astype(np.float64, copy=False)
        node_support = self.node_support_[node_ids].astype(np.float64, copy=False)

        _, _, _, ph_aux = build_edges_by_density_ph_h0(
            self,
            min_keep_nodes=2,
            prune_isolated_nodes=True,
            commit_to_model=True,
            set_node_attributes=False,
            node_ids=node_ids,
            weights=weights,
            node_support=node_support,
            return_aux=True,
        )
        self._node_persistence_map_ = dict(ph_aux.get("node_persistence", {}))
        self._cluster_persistence_map_ = dict(ph_aux.get("cluster_persistence", {}))

        pruned = self._ph_pruned_node_ids_
        if pruned is not None and len(pruned) > 0:
            pruned = self._filter_isolated_pruned_nodes_(pruned_node_ids=pruned)
            self._physically_remove_nodes_(pruned_node_ids=pruned)

            if self.num_nodes_ > 0:
                node_ids = self._ph_input_node_ids_()
                if node_ids.size == 0:
                    node_ids = np.arange(int(self.num_nodes_), dtype=int)
                weights = self.weight_[node_ids, :].astype(np.float64, copy=False)
                node_support = self.node_support_[node_ids].astype(np.float64, copy=False)

                _, _, _, ph_aux = build_edges_by_density_ph_h0(
                    self,
                    min_keep_nodes=2,
                    prune_isolated_nodes=True,
                    commit_to_model=True,
                    set_node_attributes=False,
                    node_ids=node_ids,
                    weights=weights,
                    node_support=node_support,
                    return_aux=True,
                )
                self._node_persistence_map_ = dict(ph_aux.get("node_persistence", {}))
                self._cluster_persistence_map_ = dict(ph_aux.get("cluster_persistence", {}))
        else:
            self._ph_pruned_node_ids_ = []

    def _ph_input_node_ids_(self):
        """
        Select the node subset used as PH input during training-time refresh.
        """
        n_nodes = int(self.num_nodes_)
        if n_nodes <= 0:
            return np.zeros(0, dtype=int)

        node_ids = np.arange(n_nodes, dtype=int)
        if self.is_weight_ is None or self.is_weight_.size != n_nodes:
            return node_ids

        active_node_ids = node_ids[self.is_weight_.astype(bool)]
        if active_node_ids.size > 0:
            return active_node_ids
        return node_ids

    def __apply_density_ph_periodically_(self):
        """Periodically refresh the PH partition and cached predict view."""
        interval = int(self.lambda_) if self.lambda_ is not None else 0
        if interval <= 0:
            return

        self._num_signal_since_density_ph = int(self._num_signal_since_density_ph) + 1
        if self._num_signal_since_density_ph < interval:
            return

        self._prune_low_support_nodes_idat_style_()

        self._refresh_density_ph_raw_()
        self._commit_predict_style_merge_to_model_()
        self._num_signal_since_density_ph = 0

    def _prune_low_support_nodes_idat_style_(self):
        """
        Remove old singleton-support nodes while preserving sparse components.
        """
        if self.node_support_ is None or int(self.num_nodes_) <= 0:
            return
        if self.node_birth_step_ is None or self.node_birth_step_.size != int(self.num_nodes_):
            return

        support = np.asarray(self.node_support_[: int(self.num_nodes_)], dtype=np.float64).reshape(-1)
        valid_support = support[np.isfinite(support) & (support > 0.0)]
        if valid_support.size == 0:
            return

        support_threshold = 1.0
        low_support_ids = np.where(support <= support_threshold)[0].astype(int, copy=False)
        high_support_ids = np.where(support > support_threshold)[0].astype(int, copy=False)
        if low_support_ids.size == 0 or high_support_ids.size == 0:
            return
        if low_support_ids.size <= int(max(self.lambda_, 0)):
            return

        protected_ids = set()
        eligible_low_support = set(int(nid) for nid in low_support_ids.tolist())
        node_to_component = getattr(self, "node_to_component_", None)
        if isinstance(node_to_component, dict) and len(node_to_component) > 0:
            component_low_support = {}
            component_mature_count = {}
            for nid in low_support_ids.tolist():
                comp_id = node_to_component.get(int(nid))
                if comp_id is None:
                    continue
                component_low_support.setdefault(int(comp_id), []).append(int(nid))
            for nid, comp_id in node_to_component.items():
                nid_i = int(nid)
                cid_i = int(comp_id)
                if 0 <= nid_i < support.size and float(support[nid_i]) > support_threshold:
                    component_mature_count[cid_i] = int(component_mature_count.get(cid_i, 0)) + 1
            for comp_id, member_ids in component_low_support.items():
                mature_count = int(component_mature_count.get(int(comp_id), 0))
                if mature_count >= 2:
                    continue
                eligible_low_support.difference_update(int(nid) for nid in member_ids)
                birth_local = np.asarray(self.node_birth_step_[np.asarray(member_ids, dtype=int)], dtype=np.int64)
                keep_local = int(member_ids[int(np.argmax(birth_local))])
                protected_ids.add(keep_local)

        birth = np.asarray(self.node_birth_step_[low_support_ids], dtype=np.int64)
        order = np.argsort(birth, kind="mergesort")
        ordered_low_support = low_support_ids[order]
        removable = [
            int(nid)
            for nid in ordered_low_support.tolist()
            if int(nid) not in protected_ids and int(nid) in eligible_low_support
        ]
        remove_ids = np.asarray(removable, dtype=int)
        if remove_ids.size > 0:
            self._remove_nodes_by_ids_(remove_ids)

    def _update_node_with_sample(self, node_idx, sample, update_global_scale=True, learning_rate=None):
        """
        Update one winner node with the incoming sample using the model's running-mean update.
        """
        s = int(node_idx)
        if s < 0 or s >= int(self.num_nodes_):
            return

        self.node_support_[s] += 1
        lr = (1.0 / self.node_support_[s]) if learning_rate is None else float(learning_rate)
        lr = float(np.clip(lr, 0.0, 1.0))
        self.weight_[s, :] = self.weight_[s, :] + lr * (sample[0, :] - self.weight_[s, :])

        # Update node-wise per-feature Welford moments for adaptive subspace weights.
        if self.node_mean_ is not None and self.node_M2_ is not None:
            x = np.asarray(sample[0, :], dtype=float)
            c_prev = int(self.node_support_[s]) - 1
            c_new = int(self.node_support_[s])
            if c_prev <= 0:
                self.node_mean_[s, :] = x
                self.node_M2_[s, :] = 0.0
            else:
                delta = x - self.node_mean_[s, :]
                self.node_mean_[s, :] = self.node_mean_[s, :] + delta / float(c_new)
                delta2 = x - self.node_mean_[s, :]
                self.node_M2_[s, :] = self.node_M2_[s, :] + delta * delta2

        self._invalidate_node_feature_weights_()
        self._update_train_weights_metric_row_(s)

        # Update per-node std (Welford).
        if bool(update_global_scale):
            _, current_std = self._update_online_variance(sample)
            self.buffer_std_[s, 0] = max(np.max(current_std), 1.0e-6)
        self._invalidate_predict_view_()

    def _update_is_weight_(self, s1, s2, sim_s1, sim_s2):
        """
        Mark prediction-active nodes using a support threshold on current utilization.
        """
        if self.is_weight_ is None or self.node_support_ is None:
            return
        if s1 < 0 or s1 >= int(self.num_nodes_):
            return
        if s2 < 0 or s2 >= int(self.num_nodes_):
            return
        if sim_s1 <= float(self.similarity_threshold_):
            return
        support_values = np.asarray(self.node_support_, dtype=float).reshape(-1)
        valid_support = support_values[np.isfinite(support_values) & (support_values > 1.0)]
        if valid_support.size == 0:
            return
        # Prediction-active nodes are chosen from sufficiently used nodes only.
        # The current base uses the median support over nodes with support > 1.
        support_threshold = float(np.median(valid_support))
        changed = False
        if float(self.node_support_[s1]) >= support_threshold and not bool(self.is_weight_[s1]):
            self.is_weight_[s1] = True
            changed = True
        if (
            s2 != s1
            and sim_s2 > float(self.similarity_threshold_)
            and float(self.node_support_[s2]) >= support_threshold
            and not bool(self.is_weight_[s2])
        ):
            self.is_weight_[s2] = True
            changed = True
        if changed:
            self._invalidate_predict_view_()

    def _filter_isolated_pruned_nodes_(self, pruned_node_ids):
        pruned = np.asarray(pruned_node_ids, dtype=int).reshape(-1)
        pruned = pruned[np.isfinite(pruned)]
        pruned = np.unique(pruned.astype(int, copy=False))
        pruned = pruned[(pruned >= 0) & (pruned < int(self.num_nodes_))]
        if pruned.size == 0:
            return pruned

        support = None
        if self.node_support_ is not None:
            support = np.asarray(self.node_support_, dtype=float).reshape(-1)

        is_weight = None
        if self.is_weight_ is not None:
            is_weight = np.asarray(self.is_weight_, dtype=bool).reshape(-1)

        keep = np.zeros(pruned.size, dtype=bool)
        for i, nid in enumerate(pruned.tolist()):
            supp_ok = support is not None and 0 <= nid < support.size and float(support[nid]) == 1.0
            inactive_ok = is_weight is not None and 0 <= nid < is_weight.size and (not bool(is_weight[nid]))
            keep[i] = bool(supp_ok and inactive_ok)
        return pruned[keep].astype(int, copy=False)

    def _remove_nodes_by_ids_(self, remove_node_ids):
        """
        Remove specified nodes from arrays/graph and compact node ids.
        """
        if remove_node_ids is None:
            return np.zeros(0, dtype=int)
        if self.num_nodes_ <= 0:
            return np.zeros(0, dtype=int)

        remove_ids = np.asarray(remove_node_ids, dtype=int).reshape(-1)
        remove_ids = remove_ids[np.isfinite(remove_ids)]
        if remove_ids.size == 0:
            return np.zeros(0, dtype=int)

        remove_ids = np.unique(remove_ids.astype(int, copy=False))
        remove_ids = remove_ids[(remove_ids >= 0) & (remove_ids < int(self.num_nodes_))]
        if remove_ids.size == 0:
            return np.zeros(0, dtype=int)

        # Build a keep mask once, then compact every node-aligned array with
        # the same index mapping.
        keep_mask = np.ones(int(self.num_nodes_), dtype=bool)
        keep_mask[remove_ids] = False

        if not np.any(keep_mask):
            self.num_nodes_ = 0
            self.weight_ = None
            self.node_support_ = None
            self.is_weight_ = None
            self.buffer_std_ = None
            self.node_mean_ = None
            self.node_M2_ = None
            self.node_feature_weights_ = None
            self.node_birth_step_ = None
            self.G = nx.Graph()
            self.node_to_component_ = {}
            self.node_to_cluster_ = {}
            self._node_persistence_map_ = {}
            self._cluster_persistence_map_ = {}
            self.n_ph_components_ = 0
            self.n_clusters_ = 0
            self.ph_eps_ = 0.0
            self._ph_pruned_node_ids_ = []
            self._invalidate_node_feature_weights_()
            self._invalidate_train_metric_()
            self._invalidate_predict_view_()
            return remove_ids

        # Compact every node-indexed array with the same keep mask so row ids remain aligned.
        self.weight_ = self.weight_[keep_mask, :].copy()
        self.node_support_ = self.node_support_[keep_mask].copy()
        if self.is_weight_ is not None and self.is_weight_.shape[0] == keep_mask.shape[0]:
            self.is_weight_ = self.is_weight_[keep_mask].copy()
        else:
            self.is_weight_ = np.zeros(int(np.sum(keep_mask)), dtype=bool)
        self.buffer_std_ = self.buffer_std_[keep_mask, :].copy()
        if self.node_mean_ is not None and self.node_mean_.shape[0] == keep_mask.shape[0]:
            self.node_mean_ = self.node_mean_[keep_mask, :].copy()
        else:
            self.node_mean_ = self.weight_.copy()
        if self.node_M2_ is not None and self.node_M2_.shape[0] == keep_mask.shape[0]:
            self.node_M2_ = self.node_M2_[keep_mask, :].copy()
        else:
            self.node_M2_ = np.zeros_like(self.weight_, dtype=float)
        if self.node_birth_step_ is not None and self.node_birth_step_.shape[0] == keep_mask.shape[0]:
            self.node_birth_step_ = self.node_birth_step_[keep_mask].copy()
        else:
            self.node_birth_step_ = np.zeros(int(np.sum(keep_mask)), dtype=np.int64)
        self.node_feature_weights_ = None
        self._invalidate_node_feature_weights_()
        self.num_nodes_ = int(self.weight_.shape[0])

        # Rebuild graph with compact 0..N-1 node ids after deletion.
        new_G = nx.Graph()
        for new_idx in range(int(self.num_nodes_)):
            new_G.add_node(int(new_idx), weight=self.weight_[int(new_idx), :])
        self.G = new_G

        self.node_to_component_ = {}
        self.node_to_cluster_ = {}
        self._node_persistence_map_ = {}
        self._cluster_persistence_map_ = {}
        self.n_ph_components_ = 0
        self.n_clusters_ = 0
        self.ph_eps_ = 0.0
        self._ph_pruned_node_ids_ = []
        self._invalidate_node_feature_weights_()
        self._invalidate_train_metric_()
        self._invalidate_predict_view_()
        return remove_ids

    def _physically_remove_nodes_(self, pruned_node_ids):
        """Remove nodes flagged by the PH builder and rebuild compact node ids."""
        if pruned_node_ids is None:
            return
        if self.num_nodes_ <= 0:
            return

        pruned = np.asarray(pruned_node_ids, dtype=int).reshape(-1)
        pruned = pruned[np.isfinite(pruned)]
        if pruned.size == 0:
            return

        pruned = np.unique(pruned.astype(int, copy=False))
        pruned = pruned[(pruned >= 0) & (pruned < int(self.num_nodes_))]
        if pruned.size == 0:
            return

        self._remove_nodes_by_ids_(pruned)

    def _calculate_lambda_similarity_threshold(self, buffer_std, buffer1, current_idx, data_size, num_nodes, num_clusters, finalize_on_exhaustion=True, suppress_update=False, ratioNC_in=None):
        """
        Decide new Lambda and similarity threshold from the buffered data.
        Stability check uses Cholesky; instability is flagged when Cholesky fails
        or the determinant approximation via Cholesky diagonal product is too small.
        When suppress_update=True, do not write to self.past_ratioNC_ (use ratioNC_in).
        """
        Lambda_new = None
        similarityTh_new = None
        isRenewed = False

        if buffer1 is None or buffer1.size == 0 or buffer1.shape[0] < 2:
            return (Lambda_new, similarityTh_new, isRenewed, self.past_ratioNC_ if ratioNC_in is None else ratioNC_in)

        # normalize buffer_std (scalar or array)
        if buffer_std is None:
            max_std = 1.0
        elif isinstance(buffer_std, (int, float, np.floating)):
            max_std = float(buffer_std)
        else:
            max_std = float(np.max(buffer_std))
        alpha = 1.0 / max(max_std, 1.0e-12)

        buffer1_metric = self._transform_with_train_metric_(buffer1)
        _, inv_dist_mat = self.compute_inverse_distance_matrix(buffer1_metric, alpha)

        unstable = False
        try:
            L = np.linalg.cholesky(inv_dist_mat)
            det_approx = float((np.prod(np.diag(L))) ** 2)
            if det_approx < 1.0e-6:
                unstable = True
        except np.linalg.LinAlgError:
            # Non-PD: declare instability without determinant fallback
            unstable = True

        ratioNC_local = self.past_ratioNC_ if ratioNC_in is None else ratioNC_in

        if unstable or (finalize_on_exhaustion and (current_idx == data_size)):
            Lambda_new = int(buffer1.shape[0])
            n = inv_dist_mat.shape[0]
            inv_dist_mat[np.arange(n), np.arange(n)] = -np.inf
            max_values = np.max(inv_dist_mat, axis=1)
            if (num_nodes is None) or (num_clusters is None):
                similarityTh_new = float(np.mean(max_values))
            else:
                similarityTh_new, ratioNC_local = self._candidate_similarity_threshold(
                    num_clusters, num_nodes, max_values,
                    max(int(self.lambda_) if self.lambda_ is not None else 1, 1),
                    suppress_update=suppress_update, ratioNC_in=ratioNC_local
                )
            isRenewed = True

        return (Lambda_new, similarityTh_new, isRenewed, ratioNC_local)

    def _calculate_lambda_decremental_direction(self, buffer_std, Lambda, similarity_th, buffer1, num_nodes, num_clusters):
        """
        Attempt to reduce lambda_ if the inverse distance matrix indicates instability.
        Uses partial Cholesky factorization checks or determinant checks on submatrices.
        Holds ratioNC locally and does not commit side effects.
        """
        isIncremental = True
        lastValidLambda = int(Lambda)
        lastValidsimilarityTh = float(similarity_th)

        if buffer1 is None or buffer1.size == 0 or buffer1.shape[0] < 2:
            return (lastValidLambda, lastValidsimilarityTh, isIncremental, self.past_ratioNC_)

        max_std = float(np.max(buffer_std)) if (buffer_std is not None and np.size(buffer_std) > 0) else 1.0
        alpha = 1.0 / max(max_std, 1.0e-12)

        buffer1_reversed = self._transform_with_train_metric_(buffer1[::-1])
        dist_mat, inv_dist_mat_full = self.compute_inverse_distance_matrix(buffer1_reversed, alpha)

        full_pd = True
        try:
            L = np.linalg.cholesky(inv_dist_mat_full)
        except np.linalg.LinAlgError:
            full_pd = False

        startN = 2
        endN = buffer1.shape[0]
        ratioNC_local = self.past_ratioNC_

        if full_pd:
            for currentN in range(startN, endN + 1):
                subDetValue = float((np.prod(np.diag(L[:currentN, :currentN]))) ** 2)
                if subDetValue < 1.0e-6:
                    return (lastValidLambda, lastValidsimilarityTh, False, ratioNC_local)
                lastValidLambda = currentN
                subMat = inv_dist_mat_full[0:currentN, 0:currentN].copy()
                np.fill_diagonal(subMat, -np.inf)
                max_values = np.max(subMat, axis=1)
                lastValidsimilarityTh, ratioNC_local = self._candidate_similarity_threshold(
                    num_clusters, num_nodes, max_values, max(int(self.lambda_), 1),
                    suppress_update=True, ratioNC_in=ratioNC_local
                )
        else:
            for currentN in range(startN, endN + 1):
                subDistMat = dist_mat[0:currentN, 0:currentN]
                invDistMat = _inverse_distance_similarity(subDistMat, alpha)
                try:
                    Lsub = np.linalg.cholesky(invDistMat)
                    subDetValue = float((np.prod(np.diag(Lsub))) ** 2)
                    if subDetValue < 1.0e-6:
                        return (lastValidLambda, lastValidsimilarityTh, False, ratioNC_local)
                except np.linalg.LinAlgError:
                    return (lastValidLambda, lastValidsimilarityTh, False, ratioNC_local)
                lastValidLambda = currentN
                np.fill_diagonal(invDistMat, -np.inf)
                max_values = np.max(invDistMat, axis=1)
                lastValidsimilarityTh, ratioNC_local = self._candidate_similarity_threshold(
                    num_clusters, num_nodes, max_values, max(int(self.lambda_), 1),
                    suppress_update=True, ratioNC_in=ratioNC_local
                )

        return (lastValidLambda, lastValidsimilarityTh, isIncremental, ratioNC_local)

    def _calculate_lambda_incremental_direction(self, bufferStd, Lambda, similarityTh, numNodes, numClusters):
        """
        Incrementally increase lambda_ using ONLY past samples (causal).
        Adopt the last stable length lowK immediately before the first instability.
        Hold ratioNC locally and commit outside.
        """
        Lambda = int(Lambda)
        Lambda_new = int(Lambda)
        similarityTh_new = float(similarityTh)

        availPast = int(self.buffer_data_count_)
        if availPast <= Lambda:
            return Lambda_new, similarityTh_new, self.past_ratioNC_

        endIndex = min(2 * Lambda, availPast)

        step = 1
        lowK = Lambda
        highK = min(Lambda + step, endIndex)
        found = False
        ratioNC_local = self.past_ratioNC_

        while highK <= endIndex:
            extend_buffer = self._get_from_buffer(highK)
            Lcand, Scand, ok, ratioNC_local = self._calculate_lambda_similarity_threshold(
                bufferStd, extend_buffer, highK, endIndex, numNodes, numClusters,
                finalize_on_exhaustion=False, suppress_update=True, ratioNC_in=ratioNC_local
            )
            if ok:
                found = True
                break
            else:
                lowK = highK
                step = min(step * 2, endIndex - Lambda)
                highK = min(Lambda + step, endIndex)
                if highK <= lowK:
                    break

        if found:
            Lambda_new = int(lowK)
            stable_buffer = self._get_from_buffer(lowK)
            max_std = float(np.max(bufferStd)) if (bufferStd is not None and np.size(bufferStd) > 0) else 1.0
            alpha = 1.0 / max(max_std, 1.0e-12)
            stable_buffer_metric = self._transform_with_train_metric_(stable_buffer)
            _, inv_dist_mat = self.compute_inverse_distance_matrix(stable_buffer_metric, alpha)
            np.fill_diagonal(inv_dist_mat, -np.inf)
            max_values = np.max(inv_dist_mat, axis=1)
            if (numNodes is None) or (numClusters is None):
                similarityTh_new = float(np.mean(max_values))
            else:
                similarityTh_new, ratioNC_local = self._candidate_similarity_threshold(
                    numClusters, numNodes, max_values, lowK,
                    suppress_update=True, ratioNC_in=ratioNC_local
                )
            return Lambda_new, similarityTh_new, ratioNC_local
        else:
            cap_buffer = self._get_from_buffer(endIndex)
            Lcap, Scap, _, ratioNC_local = self._calculate_lambda_similarity_threshold(
                bufferStd, cap_buffer, endIndex, endIndex, numNodes, numClusters,
                finalize_on_exhaustion=True, suppress_update=True, ratioNC_in=ratioNC_local
            )
            return int(Lcap), float(Scap), ratioNC_local

    def _candidate_similarity_threshold(self, num_clusters, num_nodes, max_values, Lambda, suppress_update=False, ratioNC_in=None):
        """Compute the next similarity threshold from the cluster-to-node ratio."""
        current_ratioNC = float(num_clusters) / float(num_nodes) if num_nodes > 0 else 0.0
        current_ratioNC = float(np.clip(current_ratioNC, 0.0, 1.0))
        self.current_ratioNC_ = current_ratioNC
        mixing_ratio = 1.0 / float(Lambda) if Lambda != 0 else 0.0
        base_ratio = self.past_ratioNC_ if ratioNC_in is None else ratioNC_in
        base_ratio = float(np.clip(base_ratio, 0.0, 1.0))

        mixed_ratioNC = (
            mixing_ratio * current_ratioNC
            + (1.0 - mixing_ratio) * base_ratio
        )
        mixed_ratioNC = float(np.clip(mixed_ratioNC, 0.0, 1.0))

        if not suppress_update:
            self.past_ratioNC_ = mixed_ratioNC

        qValue = mixed_ratioNC
        # Use hazen interpolation to match MATLAB quantile behavior
        new_threshold = np.quantile(max_values, qValue, method='hazen')
        return new_threshold, mixed_ratioNC

    def _update_online_variance(self, new_samples):
        """
        Incrementally update the mean and variance using Welford's algorithm.
        Allows handling new data without storing the entire dataset.
        """
        if new_samples.ndim == 1:
            new_samples = new_samples.reshape(1, -1)
        num_new = new_samples.shape[0]

        if num_new == 0:
            if self.mean_data_ is None or self.mean_data_.size == 0:
                num_features = new_samples.shape[1] if new_samples.ndim == 2 else 0
                current_std = np.zeros(num_features, dtype=np.float64)
            elif self.count_data_ <= 1:
                current_std = np.zeros_like(self.mean_data_)
            else:
                variance = self.M2_data_ / (self.count_data_ - 1)
                variance[variance < 0] = 0
                current_std = np.sqrt(variance)
            return (self, current_std)

        if self.mean_data_ is None or self.mean_data_.size == 0:
            if num_new == 1:
                self.mean_data_ = new_samples[0, :].astype(np.float64, copy=True)
            else:
                self.mean_data_ = np.mean(new_samples, axis=0)
            self.M2_data_ = np.zeros_like(self.mean_data_)
            self.count_data_ = num_new
            current_std = np.zeros_like(self.mean_data_)
        else:
            if num_new == 1:
                x = new_samples[0, :]
                new_count = self.count_data_ + 1
                delta = x - self.mean_data_
                self.mean_data_ = self.mean_data_ + (delta / new_count)
                delta2 = x - self.mean_data_
                self.M2_data_ = self.M2_data_ + (delta * delta2)
                self.count_data_ = new_count
            else:
                new_count = self.count_data_ + num_new
                delta = new_samples - self.mean_data_
                self.mean_data_ = self.mean_data_ + np.sum(delta, axis=0) / new_count
                self.M2_data_ = self.M2_data_ + np.sum(
                    delta * (new_samples - self.mean_data_), axis=0
                )
                self.count_data_ = new_count
            if self.count_data_ <= 1:
                current_std = np.zeros_like(self.mean_data_)
            else:
                variance = self.M2_data_ / (self.count_data_ - 1)
                variance[variance < 0] = 0
                current_std = np.sqrt(variance)
        self._invalidate_node_feature_weights_()
        return (self, current_std)

    def _append_to_buffer(self, new_data):
        """
        Append new_data to the ring buffer.
        Capacity policy: expand with headroom only; never shrink.
        After insertion, the buffer may be trimmed logically to keep only the last buffer_keep_len_ samples.
        """
        if new_data.ndim == 1:
            new_data = new_data.reshape(1, -1)
        num_new, num_cols = new_data.shape

        if self.buffer_data_count_ == 0:
            initial_capacity = max(
                500,
                num_new,
                int(np.ceil(2 * max(1, self.buffer_keep_len_)))
            )
            self.buffer_data_capacity_ = initial_capacity
            self.buffer_data_ = np.zeros((initial_capacity, num_cols), dtype=np.float64)
            self.buffer_data_start_ = 0
            self.buffer_data_count_ = 0

        required_capacity = self.buffer_data_count_ + num_new
        if required_capacity > self.buffer_data_capacity_:
            new_capacity = max(
                self.buffer_data_capacity_ * 2,
                required_capacity * 2,
                int(np.ceil(2 * max(1, self.buffer_keep_len_)))
            )
            new_buffer = np.zeros((new_capacity, num_cols), dtype=np.float64)
            current_data = self._get_from_buffer()
            current_count = current_data.shape[0]
            if current_count > 0:
                new_buffer[0:current_count, :] = current_data
            self.buffer_data_ = new_buffer
            self.buffer_data_capacity_ = new_capacity
            self.buffer_data_start_ = 0
            self.buffer_data_count_ = current_count

        if num_new == 1:
            index = (self.buffer_data_start_ + self.buffer_data_count_) % self.buffer_data_capacity_
            self.buffer_data_[index, :] = new_data[0, :]
        else:
            indices = (
                (self.buffer_data_start_ + np.arange(self.buffer_data_count_, self.buffer_data_count_ + num_new))
                % self.buffer_data_capacity_
            )
            self.buffer_data_[indices, :] = new_data
        self.buffer_data_count_ += num_new

        keep_len = int(self.buffer_keep_len_) if self.buffer_keep_len_ is not None else self.buffer_data_count_
        if keep_len <= 0 or not np.isfinite(keep_len):
            keep_len = self.buffer_data_count_
        if self.buffer_data_count_ > keep_len:
            excess = self.buffer_data_count_ - keep_len
            self.buffer_data_start_ = (self.buffer_data_start_ + excess) % self.buffer_data_capacity_
            self.buffer_data_count_ -= excess

    def _get_from_buffer(self, req_num=None):
        """
        Retrieve the specified number of latest samples from the ring buffer.
        Returns fewer if req_num exceeds the current stored count.
        Oldest → newest, matching MATLAB get_from_buffer.
        """
        if self.buffer_data_ is None:
            return np.array([])
        if req_num is None:
            req_num = self.buffer_data_count_
        if req_num > self.buffer_data_count_:
            req_num = self.buffer_data_count_
        if req_num == 0:
            return np.zeros((0, self.buffer_data_.shape[1]))

        start_idx = (self.buffer_data_start_ + self.buffer_data_count_ - req_num) % self.buffer_data_capacity_
        end_idx = start_idx + req_num

        if end_idx <= self.buffer_data_capacity_:
            data = self.buffer_data_[start_idx:end_idx, :]
        else:
            first_part = self.buffer_data_[start_idx:self.buffer_data_capacity_, :]
            second_part_count = end_idx - self.buffer_data_capacity_
            second_part = self.buffer_data_[0:second_part_count, :]
            data = np.vstack((first_part, second_part))
        return data

    def _clear_buffer(self):
        """
        Clear the ring buffer contents without altering its allocated capacity.
        """
        self.buffer_data_count_ = 0
        self.buffer_data_start_ = 0

    def _update_buffer_keep_len(self):
        """Recompute the retained ring-buffer length after a lambda update."""
        if self.lambda_ is None:
            self.buffer_keep_len_ = self.buffer_data_count_
            return
        keep_len = max(int(self.lambda_), 2 * int(self.lambda_))
        self.buffer_keep_len_ = keep_len

        if self.buffer_data_count_ > self.buffer_keep_len_:
            excess = self.buffer_data_count_ - self.buffer_keep_len_
            self.buffer_data_start_ = (self.buffer_data_start_ + excess) % self.buffer_data_capacity_
            self.buffer_data_count_ -= excess

        target_capacity = int(np.ceil(2 * self.buffer_keep_len_))
        if self.buffer_data_capacity_ < target_capacity:
            if self.buffer_data_ is None:
                # Defer allocation until first append
                self.buffer_data_capacity_ = target_capacity
                return
            num_cols = self.buffer_data_.shape[1]
            new_buffer = np.zeros((target_capacity, num_cols), dtype=np.float64)
            current_data = self._get_from_buffer()
            current_count = current_data.shape[0]
            if current_count > 0:
                new_buffer[0:current_count, :] = current_data
            self.buffer_data_ = new_buffer
            self.buffer_data_capacity_ = target_capacity
            self.buffer_data_start_ = 0
            self.buffer_data_count_ = current_count

    def _add_cluster_node(self, sample):
        """
        Create a new node with the given sample as its prototype.
        """
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        self.num_nodes_ += 1
        idx = self.num_nodes_ - 1

        # Extend node-aligned arrays by one row so the new node is immediately
        # visible to the graph, variance tracker, and PH builder.
        if self.weight_ is None or self.weight_.size == 0:
            self.weight_ = sample.copy()
        else:
            self.weight_ = np.vstack([self.weight_, sample])

        if self.node_support_ is None or self.node_support_.size == 0:
            self.node_support_ = np.array([1], dtype=np.float64)
        else:
            self.node_support_ = np.hstack([self.node_support_, [1.0]])

        if self.is_weight_ is None or self.is_weight_.size == 0:
            self.is_weight_ = np.array([False], dtype=bool)
        else:
            self.is_weight_ = np.hstack([self.is_weight_, [False]])

        if self.buffer_std_ is None or self.buffer_std_.size == 0:
            self.buffer_std_ = np.zeros((1, 1), dtype=np.float64)
        else:
            self.buffer_std_ = np.vstack([self.buffer_std_, np.zeros((1, 1), dtype=np.float64)])

        # Initialize node-wise moments for adaptive feature weights.
        if self.node_mean_ is None or self.node_mean_.size == 0:
            self.node_mean_ = sample.copy()
            self.node_M2_ = np.zeros_like(sample, dtype=float)
        else:
            self.node_mean_ = np.vstack([self.node_mean_, sample.copy()])
            self.node_M2_ = np.vstack([self.node_M2_, np.zeros_like(sample, dtype=float)])

        self._invalidate_node_feature_weights_()

        _, current_std = self._update_online_variance(sample)
        self.buffer_std_[idx, 0] = max(np.max(current_std), 1.0e-6)

        birth_step = np.array([int(self.sample_counts_)], dtype=np.int64)
        if self.node_birth_step_ is None or self.node_birth_step_.size == 0:
            self.node_birth_step_ = birth_step
        else:
            self.node_birth_step_ = np.hstack([self.node_birth_step_, birth_step])

        # Add the new node to main graph, with weight info.
        self.G.add_node(idx)
        self.G.nodes[idx]['weight'] = self.weight_[idx]
        self._invalidate_train_metric_()
        self._invalidate_predict_view_()

    @staticmethod
    def compute_inverse_distance_matrix(X, alpha):
        """
        Compute pairwise Euclidean distances and convert them into inverse distance similarities.
        dist_mat: Euclidean distance matrix
        inv_dist_mat: 1 / (1 + alpha * distance)
        """
        # Use the standard quadratic expansion to form a dense pairwise distance matrix.
        XX = np.sum(X ** 2, axis=1, keepdims=True)
        D_sq = XX + XX.T - 2.0 * np.dot(X, X.T)
        D_sq[D_sq < 0] = 0.0
        dist_mat = np.sqrt(D_sq)
        inv_dist_mat = _inverse_distance_similarity(dist_mat, alpha)
        return (dist_mat, inv_dist_mat)
