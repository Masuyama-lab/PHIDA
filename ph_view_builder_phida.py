# Copyright (c) 2026 Naoki Masuyama
# SPDX-License-Identifier: MIT
#
# This file is part of PHIDA.
# Licensed under the MIT License; see LICENSE in the project root for details.

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx

DEFAULT_MIN_KEEP_NODES = 2
DEFAULT_PRUNE_ISOLATED_NODES = True
DEFAULT_COMMIT_TO_MODEL = True
DEFAULT_SET_NODE_ATTRIBUTES = False
DEFAULT_RETURN_AUX = False


def _feature_weights_for_nodes(
    model,
    node_ids: np.ndarray,
    weights_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Resolve node-wise feature weights aligned to node_ids.

    Returns None when model-side feature weights are unavailable or
    incompatible with the provided node order/shape.
    """
    get_weights = getattr(model, "_node_feature_weights_matrix_", None)
    if not callable(get_weights):
        return None

    try:
        fw_all = get_weights()
    except Exception:
        return None

    if fw_all is None:
        return None

    # Validate the returned matrix before aligning it to the requested node ids.
    fw_all = np.asarray(fw_all, dtype=float)
    if fw_all.ndim != 2:
        return None

    n_rows, n_cols = int(weights_shape[0]), int(weights_shape[1])
    if fw_all.shape[1] != n_cols:
        return None

    ids = np.asarray(node_ids, dtype=int).reshape(-1)
    if ids.size != n_rows:
        return None
    if int(np.min(ids)) < 0 or int(np.max(ids)) >= int(fw_all.shape[0]):
        return None

    fw = fw_all[ids, :]
    if fw.shape != (n_rows, n_cols):
        return None
    if not np.all(np.isfinite(fw)):
        return None
    return fw


def build_edges_by_density_ph_h0(
    model,
    min_keep_nodes: int = DEFAULT_MIN_KEEP_NODES,
    prune_isolated_nodes: bool = DEFAULT_PRUNE_ISOLATED_NODES,
    commit_to_model: bool = DEFAULT_COMMIT_TO_MODEL,
    set_node_attributes: bool = DEFAULT_SET_NODE_ATTRIBUTES,
    node_ids: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    node_support: Optional[np.ndarray] = None,
    return_aux: bool = DEFAULT_RETURN_AUX,
) -> Tuple[Dict[int, int], int, float]:
    """
    Compute node-level clusters via 0D persistence on an auto-k symmetric kNN graph.

    When commit_to_model=True, this function records node ids pruned from the
    PH computation in model._ph_pruned_node_ids_.
    """

    g: nx.Graph = model.G_

    def _pack_return(mapping: Dict[int, int], n_clusters: int, ph_eps: float, aux: Optional[Dict] = None):
        if bool(return_aux):
            return mapping, int(n_clusters), float(ph_eps), (aux if aux is not None else {})
        return mapping, int(n_clusters), float(ph_eps)

    if bool(commit_to_model):
        model._ph_pruned_node_ids_ = []

    if node_ids is None:
        node_ids = np.array(sorted([int(n) for n in g.nodes]), dtype=int)
    else:
        node_ids = np.asarray(node_ids, dtype=int)

    n = int(node_ids.size)

    if n == 0:
        mapping: Dict[int, int] = {}
        if commit_to_model:
            model.node_to_component_ = {}
            model.n_clusters_ = 0
            model.ph_eps_ = 0.0
            model._ph_pruned_node_ids_ = []
        return _pack_return(mapping, 0, 0.0, {})

    if n == 1:
        only_id = int(node_ids[0])
        mapping = {only_id: 0}
        if commit_to_model:
            model.node_to_component_ = mapping
            model.n_clusters_ = 1
            model.ph_eps_ = 0.0
            model._ph_pruned_node_ids_ = []
        if set_node_attributes:
            nx.set_node_attributes(g, {only_id: {"cluster": 0}})
        aux = {
            "cluster_support": {0: 1.0},
            "cluster_persistence": {0: 1.0},
            "cluster_representative_node": {0: only_id},
            "node_persistence": {only_id: 1.0},
            "ph_threshold": 0.0,
        }
        return _pack_return(mapping, 1, 0.0, aux)

    if weights is None:
        weights = np.array([np.asarray(g.nodes[int(nid)]["weight"], dtype=float) for nid in node_ids], dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    if node_support is None:
        rho_raw = np.array(
            [float(g.nodes[int(nid)].get("node_support", 0.0)) for nid in node_ids],
            dtype=float,
        )
    else:
        support_raw = np.asarray(node_support, dtype=float).reshape(-1)
        rho_raw = support_raw

    rho = np.log(rho_raw)

    feature_weights = _feature_weights_for_nodes(
        model=model,
        node_ids=node_ids,
        weights_shape=weights.shape,
    )
    edges_idx_full = _auto_knn_edges_euclidean_pruned_symmetric(
        weights=weights,
        node_support=rho_raw,
        feature_weights=feature_weights,
    )

    if len(edges_idx_full) > 0:
        edges_arr_full = np.asarray(edges_idx_full, dtype=int).reshape(-1, 2)
        deg = np.bincount(edges_arr_full.reshape(-1), minlength=n)
    else:
        edges_arr_full = np.zeros((0, 2), dtype=int)
        deg = np.zeros(n, dtype=int)
    isolated_mask = deg == 0
    use_mask = np.ones(n, dtype=bool)

    if bool(prune_isolated_nodes):
        use_mask = ~isolated_mask

        # Keep a minimum number of nodes in the PH input even when the
        # kNN graph becomes too sparse after isolated-node filtering.
        min_keep = int(max(0, min(int(min_keep_nodes), n)))
        if min_keep > 0:
            if int(use_mask.sum()) < min_keep:
                if n <= min_keep:
                    anchor_idx = np.arange(n, dtype=int)
                else:
                    anchor_idx = np.argsort(-rho)[:min_keep].astype(int)
                use_mask[anchor_idx] = True

        pruned_node_ids = node_ids[~use_mask].tolist()
        if commit_to_model:
            model._ph_pruned_node_ids_ = [int(v) for v in pruned_node_ids]

        if int(use_mask.sum()) == 0:
            mapping = {}
            if commit_to_model:
                model.node_to_component_ = mapping
                model.n_clusters_ = 0
                model.ph_eps_ = 0.0
            return _pack_return(mapping, 0, 0.0, {})

        if int(use_mask.sum()) == 1:
            keep_i = int(np.where(use_mask)[0][0])
            mapping = {int(node_ids[keep_i]): 0}
            if commit_to_model:
                model.node_to_component_ = mapping
                model.n_clusters_ = 1
                model.ph_eps_ = 0.0
            if set_node_attributes:
                nx.set_node_attributes(g, {int(node_ids[keep_i]): {"cluster": 0}})
            aux = {
                "cluster_support": {0: 1.0},
                "cluster_persistence": {0: 1.0},
                "cluster_representative_node": {0: int(node_ids[keep_i])},
                "node_persistence": {int(node_ids[keep_i]): 1.0},
                "ph_threshold": 0.0,
            }
            return _pack_return(mapping, 1, 0.0, aux)
    else:
        if commit_to_model:
            model._ph_pruned_node_ids_ = []

    use_idx = np.where(use_mask)[0].astype(int)
    n_use = int(use_idx.size)

    old_to_new = np.full(n, -1, dtype=int)
    old_to_new[use_idx] = np.arange(n_use, dtype=int)

    if edges_arr_full.shape[0] > 0:
        remapped_edges = old_to_new[edges_arr_full]
        valid_edges = (
            (remapped_edges[:, 0] >= 0)
            & (remapped_edges[:, 1] >= 0)
            & (remapped_edges[:, 0] != remapped_edges[:, 1])
        )
        if np.any(valid_edges):
            edges_idx_arr = np.sort(remapped_edges[valid_edges], axis=1)
            edges_idx_arr = np.unique(edges_idx_arr, axis=0)
            edges_idx = [tuple(row.tolist()) for row in edges_idx_arr]
        else:
            edges_idx = []
    else:
        edges_idx = []

    node_ids_use = node_ids[use_idx]
    rho_use = rho[use_idx]

    adjacency = _edges_to_adjacency(n=n_use, edges_idx=edges_idx)

    # Run 0D persistence on the filtered graph. The resulting modes define the
    # raw PH components before the later Ward-based readout in PHIDA.
    base_mode, mode_parent, mode_persistence = _density_h0_persistence(
        adjacency=adjacency,
        rho=rho_use,
        node_ids=node_ids_use,
    )

    support_raw_use = np.maximum(rho_raw[use_idx], 1.0e-12)
    weights_use_raw = weights[use_idx]
    metric_center_use, metric_denom_use, _ = compute_adaptive_feature_metric(weights_use_raw)
    weights_use_metric = transform_with_feature_metric(
        weights_use_raw,
        center=metric_center_use,
        denom=metric_denom_use,
    )

    support_use = np.log(support_raw_use)
    select_threshold = getattr(model, "_select_ph_persistence_threshold_", None)
    if callable(select_threshold):
        thr = float(select_threshold(
            base_mode=base_mode,
            mode_parent=mode_parent,
            mode_persistence=mode_persistence,
            x=weights_use_metric,
            sample_weight=support_use,
        ))
    else:
        thr = float(_select_persistence_threshold_largest_gap(
            mode_persistence=mode_persistence,
        ))

    # Cut the PH merge tree at the selected persistence level to obtain the
    # raw component labels used by the rest of the pipeline.
    final_mode = _assign_final_modes(
        base_mode=base_mode,
        mode_parent=mode_parent,
        mode_persistence=mode_persistence,
        persistence_threshold=float(thr),
    )
    mode_to_cluster_id = _compress_mode_labels(final_mode=final_mode, node_ids=node_ids_use)
    node_cluster = np.array([mode_to_cluster_id[int(m)] for m in final_mode], dtype=int)

    node_to_component: Dict[int, int] = {int(node_ids_use[i]): int(node_cluster[i]) for i in range(n_use)}
    n_clusters = int(np.unique(node_cluster).size)
    ph_eps = float(thr)

    if commit_to_model:
        model.node_to_component_ = node_to_component
        model.n_clusters_ = n_clusters
        model.ph_eps_ = ph_eps

    if set_node_attributes:
        nx.set_node_attributes(g, {int(k): {"cluster": int(v)} for k, v in node_to_component.items()})

    mode_p_node = np.asarray(mode_persistence[final_mode], dtype=float)
    finite_pos = mode_p_node[np.isfinite(mode_p_node) & (mode_p_node > 0.0)]
    if finite_pos.size > 0:
        p_cap = float(np.max(finite_pos))
    else:
        p_cap = 1.0
    mode_p_node = np.where(np.isinf(mode_p_node), p_cap, mode_p_node)
    mode_p_node = np.where(np.isfinite(mode_p_node) & (mode_p_node >= 0.0), mode_p_node, 0.0)
    cluster_support_map: Dict[int, float] = {}
    cluster_persistence_map: Dict[int, float] = {}
    cluster_representative_node_map: Dict[int, int] = {}
    for cid in np.unique(node_cluster).tolist():
        cid_i = int(cid)
        idx = np.where(node_cluster == cid_i)[0].astype(int)
        if idx.size == 0:
            continue
        w = np.maximum(support_raw_use[idx], 0.0)
        ws = float(np.sum(w))
        cluster_support_map[cid_i] = float(max(ws, 0.0))
        if ws > 0.0:
            cp = float(np.sum(w * mode_p_node[idx]) / ws)
        else:
            cp = float(np.mean(mode_p_node[idx]))
        cluster_persistence_map[cid_i] = float(max(cp, 0.0))
        rep_local = int(final_mode[idx[0]])
        if rep_local < 0 or rep_local >= n_use:
            rep_local = int(idx[0])
        cluster_representative_node_map[cid_i] = int(node_ids_use[rep_local])
    aux = {
        "cluster_support": {int(k): float(v) for k, v in cluster_support_map.items()},
        "cluster_persistence": {int(k): float(v) for k, v in cluster_persistence_map.items()},
        "cluster_representative_node": {int(k): int(v) for k, v in cluster_representative_node_map.items()},
        "node_persistence": {int(node_ids_use[i]): float(mode_p_node[i]) for i in range(n_use)},
        "ph_threshold": float(ph_eps),
    }
    return _pack_return(node_to_component, n_clusters, ph_eps, aux)


def _median_iqr_threshold(x: np.ndarray) -> float:
    # Compute median + 1.5 * IQR on finite values.
    x = np.asarray(x, dtype=float)

    q1, med, q3 = [float(v) for v in np.quantile(x, [0.25, 0.50, 0.75], method="hazen")]
    iqr = float(q3 - q1)
    thr = float(med + 1.5 * iqr)
    return thr


def _select_persistence_threshold_largest_gap(mode_persistence: np.ndarray) -> float:
    mode_persistence = np.asarray(mode_persistence, dtype=float).reshape(-1)
    pos = mode_persistence[np.isfinite(mode_persistence) & (mode_persistence > 0.0)]
    if pos.size == 0:
        return 0.0

    levels = np.unique(pos)
    levels.sort()
    if levels.size <= 1:
        return 0.0

    best_thr = 0.0
    best_gap = -np.inf
    best_height = -np.inf
    prev = 0.0
    for curr in levels.tolist():
        curr_f = float(curr)
        gap = curr_f - float(prev)
        if (gap > best_gap) or (gap == best_gap and curr_f > best_height):
            best_gap = gap
            best_height = curr_f
            best_thr = float(prev)
        prev = curr_f
    return float(best_thr)


def compute_adaptive_feature_metric(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute robust per-feature metric scaling from data matrix x.

    Let s_j be robust feature scales estimated by:
        s_j = IQR_j / 1.349

    Define the scale-dispersion coefficient:
        cv_s = std(s) / mean(s)
    and adapt metric exponent by:
        gamma = max(1 - 1 / cv_s^2, 0)

    Final distance normalization per feature is:
        denom_j = s_j ** gamma

    This keeps the metric unchanged for near-homoscedastic features (cv_s <= 1),
    and progressively applies robust scaling when feature-scale heterogeneity is large.
    """
    x = np.asarray(x, dtype=float)
    n, d = int(x.shape[0]), int(x.shape[1])

    q1, med, q3 = np.quantile(x, [0.25, 0.50, 0.75], axis=0, method="hazen")
    scale = (q3 - q1) / 1.349
    scale = np.where((scale > 0.0) & np.isfinite(scale), scale, 1.0)

    mean_scale = float(np.mean(scale))
    std_scale = float(np.std(scale))
    cv_scale = float(std_scale / mean_scale)
    cv2 = cv_scale * cv_scale
    gamma = float(max(1.0 - (1.0 / cv2), 0.0)) if cv2 > 0.0 else 0.0
    denom = np.power(scale, gamma)
    return np.asarray(med, dtype=float), np.asarray(denom, dtype=float), float(gamma)


def transform_with_feature_metric(x: np.ndarray, center: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """
    Apply feature-wise affine scaling using center and denominator vectors.
    """
    xx = np.asarray(x, dtype=float)
    cc = np.asarray(center, dtype=float).reshape(1, -1)
    dd = np.asarray(denom, dtype=float).reshape(1, -1)

    if xx.ndim == 1:
        return ((xx.reshape(1, -1) - cc) / dd).reshape(-1)
    return (xx - cc) / dd


def _auto_knn_edges_euclidean_pruned_symmetric(
    weights: np.ndarray,
    node_support: Optional[np.ndarray] = None,
    feature_weights: Optional[np.ndarray] = None,
) -> List[Tuple[int, int]]:
    """
    Build a mutual kNN graph with automatically chosen k from node count.

    Let n be the current number of nodes in the PH graph candidate set.
    Neighborhood size is chosen as:
        k = ceil(sqrt(log(n)))

    This keeps k parameter-free and sub-logarithmic, which suppresses
    bridge-like edges compared with k ~ log(n) in larger graphs.

    `node_support` is kept only for API symmetry with the PH builder.
    """
    n = int(weights.shape[0])
    if n <= 1:
        return []

    _ = node_support  # kept for API symmetry with the PH builder.
    log_n = float(np.log(float(n)))
    k_raw = int(np.ceil(np.sqrt(log_n)))
    k = int(max(1, min(k_raw, n - 1)))

    ww_raw = np.asarray(weights, dtype=float)
    center, denom, gamma = compute_adaptive_feature_metric(ww_raw)
    ww_global = transform_with_feature_metric(ww_raw, center=center, denom=denom)
    ww_feat = None
    d = float(max(1, ww_raw.shape[1]))
    phi = 0.0
    if feature_weights is not None:
        ww_feat = np.asarray(feature_weights, dtype=float)
        if ww_feat.shape != ww_raw.shape:
            ww_feat = None
        else:
            phi = float(np.clip(gamma, 0.0, 1.0))

    xx = np.sum(ww_global * ww_global, axis=1, keepdims=True)
    dist_sq = xx + xx.T - 2.0 * np.dot(ww_global, ww_global.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    dist_mat = np.sqrt(dist_sq)
    if ww_feat is not None and phi > 0.0:
        block_size = 128
        for start in range(0, n, block_size):
            stop = min(start + block_size, n)
            diff = ww_global[start:stop, None, :] - ww_global[None, :, :]
            w_avg = 0.5 * (
                ww_feat[start:stop, None, :]
                + ww_feat[None, :, :]
            )
            weighted_sq = np.einsum("bij,bij->bi", w_avg, diff * diff)
            keff = 1.0 / np.maximum(np.einsum("bij,bij->bi", w_avg, w_avg), 1.0e-12)
            keff = np.clip(keff, 1.0, d)
            dist_local = np.sqrt(np.maximum(keff * weighted_sq, 0.0))
            dist_mat[start:stop, :] = dist_mat[start:stop, :] + phi * np.maximum(
                dist_local - dist_mat[start:stop, :],
                0.0,
            )

    dist_mat[np.arange(n), np.arange(n)] = np.inf
    knn_idx = np.argpartition(dist_mat, kth=k - 1, axis=1)[:, :k]
    knn_dist = np.take_along_axis(dist_mat, knn_idx, axis=1)
    knn_order = np.argsort(knn_dist, axis=1, kind="mergesort")
    knn_idx = np.take_along_axis(knn_idx, knn_order, axis=1)
    knn_dist = np.take_along_axis(knn_dist, knn_order, axis=1)

    q1, med, q3 = np.quantile(knn_dist, [0.25, 0.50, 0.75], axis=1, method="hazen")
    local_thr = med + 1.5 * (q3 - q1)
    global_thr = _median_iqr_threshold(knn_dist.reshape(-1))
    keep_mask = knn_dist <= np.minimum(local_thr, global_thr).reshape(-1, 1)

    empty_rows = ~np.any(keep_mask, axis=1)
    if np.any(empty_rows):
        keep_mask[empty_rows, 0] = True

    row_neighbors = []
    for i in range(n):
        kept = knn_idx[i, keep_mask[i]]
        row_neighbors.append({int(v) for v in kept.tolist()})

    edges = []
    for i in range(n):
        for j in sorted(row_neighbors[i]):
            if j <= i:
                continue
            if i in row_neighbors[j]:
                edges.append((int(i), int(j)))
    return edges


def _edges_to_adjacency(n: int, edges_idx: List[Tuple[int, int]]) -> List[np.ndarray]:
    adj: List[List[int]] = [[] for _ in range(int(n))]
    for i, j in edges_idx:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    return [np.array(v, dtype=int) for v in adj]


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(int(n), dtype=int)
        self.rank = np.zeros(int(n), dtype=int)

    def find(self, x: int) -> int:
        x = int(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
        return x

    def union(self, a: int, b: int) -> int:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
            return rb

        if self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
            return ra

        self.parent[rb] = ra
        self.rank[ra] += 1
        return ra


def _density_h0_persistence(
    adjacency: List[np.ndarray],
    rho: np.ndarray,
    node_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 0D density persistence on an undirected graph.

    Nodes are activated in descending rho. When two active components connect,
    the lower-density mode dies and stores:
      mode_parent[dead_mode]      = surviving_mode
      mode_persistence[dead_mode] = rho(dead_mode) - current_level
    """
    n = int(rho.size)
    order = np.lexsort((node_ids, -rho))

    uf = _UnionFind(n=n)
    active = np.zeros(n, dtype=bool)

    comp_mode = np.full(n, -1, dtype=int)
    mode_parent = np.full(n, -1, dtype=int)
    mode_persistence = np.full(n, np.nan, dtype=float)
    base_mode = np.full(n, -1, dtype=int)

    for v in order:
        v = int(v)
        active[v] = True
        comp_mode[v] = v

        for u in adjacency[v]:
            u = int(u)
            if not active[u]:
                continue

            rv = uf.find(v)
            ru = uf.find(u)
            if rv == ru:
                continue

            mv = int(comp_mode[rv])
            mu = int(comp_mode[ru])

            win_mode, lose_mode = _winner_loser_mode(
                m1=mv,
                m2=mu,
                rho=rho,
                node_ids=node_ids,
            )

            new_root = uf.union(rv, ru)
            comp_mode[new_root] = int(win_mode)

            if mode_parent[int(lose_mode)] == -1 and int(lose_mode) != int(win_mode):
                # Persistence is birth density minus death density level.
                mode_parent[int(lose_mode)] = int(win_mode)
                mode_persistence[int(lose_mode)] = float(rho[int(lose_mode)] - rho[v])

        base_mode[v] = int(comp_mode[uf.find(v)])

    # Modes that never die remain as roots of the PH tree.
    for i in range(n):
        if mode_parent[int(i)] == -1:
            # Surviving global modes have infinite persistence.
            mode_persistence[int(i)] = np.inf

    return base_mode, mode_parent, mode_persistence


def _winner_loser_mode(m1: int, m2: int, rho: np.ndarray, node_ids: np.ndarray) -> Tuple[int, int]:
    r1 = float(rho[int(m1)])
    r2 = float(rho[int(m2)])

    if r1 > r2:
        return int(m1), int(m2)
    if r2 > r1:
        return int(m2), int(m1)

    id1 = int(node_ids[int(m1)])
    id2 = int(node_ids[int(m2)])
    if id1 <= id2:
        return int(m1), int(m2)
    return int(m2), int(m1)


def _assign_final_modes(
    base_mode: np.ndarray,
    mode_parent: np.ndarray,
    mode_persistence: np.ndarray,
    persistence_threshold: float,
) -> np.ndarray:
    # Cut the PH tree at persistence_threshold by climbing parent links.
    n = int(base_mode.size)
    final_mode = np.empty(n, dtype=int)

    for i in range(n):
        m = int(base_mode[int(i)])
        while np.isfinite(mode_persistence[m]) and mode_persistence[m] <= float(persistence_threshold):
            p = int(mode_parent[m])
            if p < 0:
                break
            m = p
        final_mode[int(i)] = int(m)

    return final_mode


def _compress_mode_labels(final_mode: np.ndarray, node_ids: np.ndarray) -> Dict[int, int]:
    # Reindex cluster labels in node-id order for deterministic output labels.
    unique_modes = sorted(set(int(m) for m in final_mode.tolist()), key=lambda mi: int(node_ids[int(mi)]))
    return {int(m): int(i) for i, m in enumerate(unique_modes)}
