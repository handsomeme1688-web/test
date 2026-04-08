import copy
import gc
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data.dataloader as DataLoader
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from datapro import EdgeDataset
from model import SuperedgeLearn


class ConfidenceAwarePULoss(nn.Module):
    def __init__(self, pi, eps=1e-8):
        super().__init__()
        self.pi = float(pi)
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels, confidence):
        labels = labels.float()
        confidence = confidence.float().clamp_(0.0, 1.0)

        pos_mask = labels > 0.5
        unlabeled_mask = ~pos_mask
        zero = logits.new_tensor(0.0)

        if pos_mask.any():
            pos_logits = logits[pos_mask]
            rp = self.bce(pos_logits, torch.ones_like(pos_logits)).mean()
            rp_neg = self.bce(pos_logits, torch.zeros_like(pos_logits)).mean()
        else:
            rp = zero
            rp_neg = zero

        if unlabeled_mask.any():
            unlabeled_logits = logits[unlabeled_mask]
            neg_weights = 1.0 - confidence[unlabeled_mask]
            neg_losses = self.bce(unlabeled_logits, torch.zeros_like(unlabeled_logits))
            weight_sum = neg_weights.sum().clamp_min(self.eps)
            ru = (neg_weights * neg_losses).sum() / weight_sum
        else:
            ru = zero

        return self.pi * rp + torch.clamp(ru - self.pi * rp_neg, min=0.0)


class ConfidenceAwarePNULoss(nn.Module):
    def __init__(self, pi, rn_threshold, pu_weight=1.0, sup_weight=1.0, rank_weight=0.2):
        super().__init__()
        self.pu_loss = ConfidenceAwarePULoss(pi=pi)
        self.rn_threshold = float(rn_threshold)
        self.pu_weight = float(pu_weight)
        self.sup_weight = float(sup_weight)
        self.rank_weight = float(rank_weight)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels, confidence):
        labels = labels.float()
        confidence = confidence.float().clamp_(0.0, 1.0)

        pu_term = self.pu_loss(logits, labels, confidence)
        pos_mask = labels > 0.5
        rn_mask = (labels <= 0.5) & (confidence <= self.rn_threshold)

        zero = logits.new_tensor(0.0)
        sup_term = zero
        rank_term = zero

        if pos_mask.any() and rn_mask.any():
            pos_logits = logits[pos_mask]
            rn_logits = logits[rn_mask]

            sup_logits = torch.cat((pos_logits, rn_logits), dim=0)
            sup_targets = torch.cat((torch.ones_like(pos_logits), torch.zeros_like(rn_logits)), dim=0)
            sup_weights = torch.cat(
                (torch.ones_like(pos_logits), (1.0 - confidence[rn_mask]).clamp_min(0.1)),
                dim=0,
            )

            sup_loss = self.bce(sup_logits, sup_targets)
            sup_term = (sup_loss * sup_weights).sum() / sup_weights.sum().clamp_min(1e-8)

            margin = pos_logits.unsqueeze(1) - rn_logits.unsqueeze(0)
            rank_term = torch.nn.functional.softplus(-margin).mean()
        elif pos_mask.any():
            pos_logits = logits[pos_mask]
            sup_term = self.bce(pos_logits, torch.ones_like(pos_logits)).mean()

        return self.pu_weight * pu_term + self.sup_weight * sup_term + self.rank_weight * rank_term


def move_simdata_to_device(sim_data, device):
    return {key: value.to(device) for key, value in sim_data.items()}


def _to_numpy(view):
    if isinstance(view, np.ndarray):
        return view.astype(np.float32)
    return view.detach().cpu().numpy().astype(np.float32)


def build_confidence_views(sim_data):
    views = {
        'mm_f': _to_numpy(sim_data['mm_f']),
        'mm_s': _to_numpy(sim_data['mm_s']),
        'mm_g': _to_numpy(sim_data['mm_g']),
        'dd_t': _to_numpy(sim_data['dd_t']),
        'dd_s': _to_numpy(sim_data['dd_s']),
        'dd_g': _to_numpy(sim_data['dd_g']),
    }
    if 'mm_tg' in sim_data:
        views['mm_tg'] = _to_numpy(sim_data['mm_tg'])
    if 'mm_pw' in sim_data:
        views['mm_pw'] = _to_numpy(sim_data['mm_pw'])
    if 'dd_onto' in sim_data:
        views['dd_onto'] = _to_numpy(sim_data['dd_onto'])

    views['mm'] = ((views['mm_f'] + views['mm_s'] + views['mm_g']) / 3.0).astype(np.float32)
    views['dd'] = ((views['dd_t'] + views['dd_s'] + views['dd_g']) / 3.0).astype(np.float32)
    return views


def resolve_semantic_hyperedge_views(sim_views):
    return {
        'm_function': sim_views['mm_f'],
        'm_target': sim_views.get('mm_tg', sim_views['mm_s']),
        'm_pathway': sim_views.get('mm_pw', sim_views['mm_g']),
        'd_ontology': sim_views.get('dd_onto', sim_views['dd_s']),
        'd_topology': sim_views['dd_t'],
        'd_diffusion': sim_views['dd_g'],
    }


def topk_row_normalize(similarity, topk):
    similarity = np.asarray(similarity, dtype=np.float32).copy()
    np.fill_diagonal(similarity, 0.0)

    if topk > 0 and topk < similarity.shape[1]:
        topk_idx = np.argsort(-similarity, axis=1)[:, :topk]
        sparse = np.zeros_like(similarity, dtype=np.float32)
        row_idx = np.arange(similarity.shape[0])[:, None]
        sparse[row_idx, topk_idx] = similarity[row_idx, topk_idx]
    else:
        sparse = similarity

    row_sum = sparse.sum(axis=1, keepdims=True) + 1e-8
    return (sparse / row_sum).astype(np.float32)


def minmax_scale(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    min_val = float(matrix.min())
    max_val = float(matrix.max())
    return ((matrix - min_val) / (max_val - min_val + 1e-8)).astype(np.float32)


def calculate_cv_confidence(sim_views, edges_train, labels_train, full_shape):
    train_md_matrix = np.zeros(full_shape, dtype=np.float32)
    pos_edges = edges_train[labels_train == 1].astype(np.int64)
    if pos_edges.size > 0:
        train_md_matrix[pos_edges[:, 0], pos_edges[:, 1]] = 1.0

    confidence_scores = sim_views['mm'] @ train_md_matrix @ sim_views['dd']
    min_val = float(confidence_scores.min())
    max_val = float(confidence_scores.max())
    confidence_scores = (confidence_scores - min_val) / (max_val - min_val + 1e-8)

    if pos_edges.size > 0:
        confidence_scores[pos_edges[:, 0], pos_edges[:, 1]] = 1.0
    return confidence_scores.astype(np.float32)


def estimate_class_prior(num_train_pos, full_shape):
    total_pairs = full_shape[0] * full_shape[1]
    return max(num_train_pos / total_pairs, 1e-6)


def calculate_metrics_from_probs(probs, labels, threshold):
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    preds = (probs >= threshold).astype(np.float32)

    metrics = {
        'auc': roc_auc_score(labels, probs),
        'aupr': average_precision_score(labels, probs),
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'precision': precision_score(labels, preds, zero_division=0),
    }
    return metrics


def calculate_metrics(logits, labels, threshold):
    logits = np.asarray(logits, dtype=np.float32)
    probs = torch.sigmoid(torch.from_numpy(logits)).cpu().numpy()
    return calculate_metrics_from_probs(probs, labels, threshold)


def format_metrics(metrics):
    return (
        f"AUC: {metrics['auc']:.4f} | "
        f"AUPR: {metrics['aupr']:.4f} | "
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"F1: {metrics['f1']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"Precision: {metrics['precision']:.4f}"
    )


def predict_logits(model, sim_data, md_m, loader, device):
    model.eval()
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for data, label, conf in loader:
            data = data.to(device, non_blocking=True)
            conf = conf.to(device, non_blocking=True)
            logits = model(sim_data, md_m, data, pair_confidence=conf)
            logits_list.append(logits.cpu())
            labels_list.append(label)

    logits = torch.cat(logits_list).numpy()
    labels = torch.cat(labels_list).numpy()
    return logits, labels


def search_best_threshold(logits, labels):
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    probs = torch.sigmoid(torch.from_numpy(logits)).cpu().numpy()
    return search_best_threshold_from_probs(probs, labels)


def search_best_threshold_from_probs(probs, labels):
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)

    best_threshold = 0.5
    best_score = float('-inf')
    for threshold in np.linspace(0.05, 0.95, 91):
        preds = (probs >= threshold).astype(np.float32)
        f1 = f1_score(labels, preds, zero_division=0)
        acc = accuracy_score(labels, preds)
        score = f1 + 0.1 * acc
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def evaluate(model, sim_data, md_m, loader, device, threshold):
    logits, labels = predict_logits(model, sim_data, md_m, loader, device)
    return calculate_metrics(logits, labels, threshold)


def train_one_epoch(model, sim_data, md_m, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for data, label, conf in loader:
        data = data.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        conf = conf.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(sim_data, md_m, data, pair_confidence=conf)
        loss = criterion(logits, label, conf)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def train_single_bag(
    param,
    sim_data,
    md_m,
    train_loader,
    calib_loader,
    device,
    class_prior,
    rn_threshold,
    seed_offset,
):
    model = SuperedgeLearn(param).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=param.lr, weight_decay=param.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    criterion = ConfidenceAwarePNULoss(
        pi=class_prior,
        rn_threshold=rn_threshold,
        pu_weight=param.loss_pu_weight,
        sup_weight=param.loss_sup_weight,
        rank_weight=param.loss_rank_weight,
    ).to(device)

    best_state = None
    best_score = float('-inf')
    best_threshold = param.threshold
    best_metrics = None
    stale_epochs = 0

    for epoch in range(1, param.epoch + 1):
        train_loss = train_one_epoch(model, sim_data, md_m, train_loader, optimizer, criterion, device)
        calib_logits, calib_labels = predict_logits(model, sim_data, md_m, calib_loader, device)
        calib_threshold = search_best_threshold(calib_logits, calib_labels)
        calib_metrics = calculate_metrics(calib_logits, calib_labels, calib_threshold)
        scheduler.step(calib_metrics['aupr'])

        score = calib_metrics['auc'] + calib_metrics['aupr']
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_threshold = calib_threshold
            best_metrics = calib_metrics
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= param.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        'model': model,
        'threshold': best_threshold,
        'metrics': best_metrics,
    }


def build_fold_md_class(base_md_class, holdout_edges, holdout_labels):
    md_class = base_md_class.copy()
    positive_holdout = holdout_edges[holdout_labels == 1].astype(np.int64)
    if positive_holdout.size > 0:
        md_class[positive_holdout[:, 0], positive_holdout[:, 1]] = 0.0
    return md_class


def estimate_rn_threshold(conf_matrix, edges_train, labels_train, quantile):
    unlabeled_edges = edges_train[labels_train == 0].astype(np.int64)
    if unlabeled_edges.size == 0:
        return 0.0
    conf_scores = conf_matrix[unlabeled_edges[:, 0], unlabeled_edges[:, 1]]
    return float(np.quantile(conf_scores, quantile))


def split_train_calibration(edges_train, labels_train, seed, ratio):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=seed)
    core_idx, calib_idx = next(splitter.split(edges_train, labels_train))
    return (
        edges_train[core_idx],
        labels_train[core_idx],
        edges_train[calib_idx],
        labels_train[calib_idx],
    )


def edges_to_linear_index(edges, num_cols):
    if edges.size == 0:
        return np.empty((0,), dtype=np.int64)
    return edges[:, 0].astype(np.int64) * num_cols + edges[:, 1].astype(np.int64)


def build_unlabeled_pool(true_md, excluded_edges):
    zero_edges = np.column_stack(np.where(true_md == 0)).astype(np.int64)
    if excluded_edges.size == 0:
        return zero_edges

    d_num = true_md.shape[1]
    excluded = np.unique(edges_to_linear_index(excluded_edges.astype(np.int64), d_num))
    zero_linear = edges_to_linear_index(zero_edges, d_num)
    keep_mask = ~np.isin(zero_linear, excluded)
    return zero_edges[keep_mask]


def confidence_stratified_sample(unlabeled_pool, conf_matrix, sample_size, seed, ratios):
    if unlabeled_pool.shape[0] <= sample_size:
        return unlabeled_pool

    rng = np.random.default_rng(seed)
    scores = conf_matrix[unlabeled_pool[:, 0], unlabeled_pool[:, 1]]
    q1, q2 = np.quantile(scores, [1 / 3, 2 / 3])

    low_idx = np.where(scores <= q1)[0]
    mid_idx = np.where((scores > q1) & (scores <= q2))[0]
    high_idx = np.where(scores > q2)[0]
    groups = [low_idx, mid_idx, high_idx]

    target_counts = np.floor(sample_size * np.asarray(ratios, dtype=np.float32)).astype(np.int64)
    target_counts[-1] += sample_size - int(target_counts.sum())

    selected = []
    remaining = np.arange(unlabeled_pool.shape[0], dtype=np.int64)
    remaining_mask = np.ones(unlabeled_pool.shape[0], dtype=bool)

    for group_idx, need in zip(groups, target_counts):
        if need <= 0 or group_idx.size == 0:
            continue
        take = min(int(need), group_idx.size)
        chosen = rng.choice(group_idx, size=take, replace=False)
        selected.append(chosen)
        remaining_mask[chosen] = False

    selected_count = sum(chunk.size for chunk in selected)
    if selected_count < sample_size:
        residual = remaining[remaining_mask]
        extra = rng.choice(residual, size=sample_size - selected_count, replace=False)
        selected.append(extra)

    final_idx = np.concatenate(selected)
    rng.shuffle(final_idx)
    return unlabeled_pool[final_idx]


def build_bag_dataset(param, positive_edges, unlabeled_pool, conf_matrix, seed):
    sample_size = min(
        unlabeled_pool.shape[0],
        max(1, int(len(positive_edges) * getattr(param, 'bag_unlabeled_ratio', 1))),
    )
    sampled_unlabeled = confidence_stratified_sample(
        unlabeled_pool,
        conf_matrix,
        sample_size=sample_size,
        seed=seed,
        ratios=getattr(param, 'bag_confidence_ratios', (0.5, 0.3, 0.2)),
    )

    bag_edges = np.vstack((positive_edges, sampled_unlabeled))
    bag_labels = np.hstack((
        np.ones(len(positive_edges), dtype=np.float32),
        np.zeros(len(sampled_unlabeled), dtype=np.float32),
    ))
    return bag_edges, bag_labels


def build_auxiliary_features(sim_views, edges_train, labels_train, full_shape, knn_k, semantic_topk, enable_semantic):
    train_md = np.zeros(full_shape, dtype=np.float32)
    pos_edges = edges_train[labels_train == 1].astype(np.int64)
    if pos_edges.size > 0:
        train_md[pos_edges[:, 0], pos_edges[:, 1]] = 1.0

    mean2 = 0.5 * (sim_views['mm'] @ train_md + train_md @ sim_views['dd'])

    m_knn = np.argsort(-sim_views['mm'], axis=1)[:, 1:knn_k + 1]
    d_knn = np.argsort(-sim_views['dd'], axis=1)[:, 1:knn_k + 1]
    knn = np.zeros_like(train_md, dtype=np.float32)

    for m_idx in range(train_md.shape[0]):
        knn[m_idx] += train_md[m_knn[m_idx]].mean(axis=0)
    for d_idx in range(train_md.shape[1]):
        knn[:, d_idx] += train_md[:, d_knn[d_idx]].mean(axis=1)

    mirna_degree = train_md.sum(axis=1).astype(np.float32)
    disease_degree = train_md.sum(axis=0).astype(np.float32)

    features = {
        'mean2': mean2.astype(np.float32),
        'knn': knn.astype(np.float32),
        'mirna_degree': mirna_degree,
        'disease_degree': disease_degree,
    }

    if enable_semantic:
        semantic_views = resolve_semantic_hyperedge_views(sim_views)
        m_function = minmax_scale(topk_row_normalize(semantic_views['m_function'], semantic_topk) @ train_md)
        m_target = minmax_scale(topk_row_normalize(semantic_views['m_target'], semantic_topk) @ train_md)
        m_pathway = minmax_scale(topk_row_normalize(semantic_views['m_pathway'], semantic_topk) @ train_md)
        d_ontology = minmax_scale(train_md @ topk_row_normalize(semantic_views['d_ontology'], semantic_topk))
        d_topology = minmax_scale(train_md @ topk_row_normalize(semantic_views['d_topology'], semantic_topk))
        d_diffusion = minmax_scale(train_md @ topk_row_normalize(semantic_views['d_diffusion'], semantic_topk))

        semantic_stack = np.stack(
            (m_function, m_target, m_pathway, d_ontology, d_topology, d_diffusion),
            axis=0,
        )
        features.update({
            'semantic_pair_avg': semantic_stack.mean(axis=0).astype(np.float32),
            'semantic_function_ontology': (0.5 * (m_function + d_ontology)).astype(np.float32),
            'semantic_target_topology': (0.5 * (m_target + d_topology)).astype(np.float32),
            'semantic_pathway_diffusion': (0.5 * (m_pathway + d_diffusion)).astype(np.float32),
        })

    return features


def make_fusion_features(edges, logits, aux_features, conf_matrix):
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim == 1:
        logits = logits[:, None]

    probs = torch.sigmoid(torch.from_numpy(logits)).cpu().numpy()
    mean_logit = logits.mean(axis=1)
    std_logit = logits.std(axis=1)
    mean_prob = probs.mean(axis=1)
    max_prob = probs.max(axis=1)
    min_prob = probs.min(axis=1)
    m_idx = edges[:, 0].astype(np.int64)
    d_idx = edges[:, 1].astype(np.int64)

    mean2 = aux_features['mean2'][m_idx, d_idx]
    knn = aux_features['knn'][m_idx, d_idx]
    mirna_degree = aux_features['mirna_degree'][m_idx]
    disease_degree = aux_features['disease_degree'][d_idx]
    confidence = conf_matrix[m_idx, d_idx]
    feature_columns = [
        logits,
        mean_logit,
        std_logit,
        mean_prob,
        max_prob,
        min_prob,
        mean2,
        knn,
        mirna_degree,
        disease_degree,
        confidence,
        mean2 * knn,
        mean2 - knn,
        mirna_degree * disease_degree,
    ]

    if 'semantic_pair_avg' in aux_features:
        feature_columns.extend((
            aux_features['semantic_pair_avg'][m_idx, d_idx],
            aux_features['semantic_function_ontology'][m_idx, d_idx],
            aux_features['semantic_target_topology'][m_idx, d_idx],
            aux_features['semantic_pathway_diffusion'][m_idx, d_idx],
        ))

    return np.column_stack(feature_columns)


def fit_fusion_models(train_features, train_labels):
    models = [
        LogisticRegression(max_iter=4000),
        GradientBoostingClassifier(random_state=42),
    ]
    for model in models:
        model.fit(train_features, train_labels)
    return models


def predict_fusion_probs(models, features):
    probs = [model.predict_proba(features)[:, 1] for model in models]
    return np.mean(np.stack(probs, axis=0), axis=0)


def run_cross_validation(sim_data, train_data, param):
    device = param.device
    sim_views = build_confidence_views(sim_data)
    sim_data = move_simdata_to_device(sim_data, device)
    true_md = train_data['true_md']

    if getattr(param, 'valid_use_full_data', False):
        train_edges = train_data['full_Edges']
        train_labels = train_data['full_Labels']
        test_edges = np.empty((0, 2), dtype=np.int64)
        test_labels = np.empty((0,), dtype=np.float32)
    else:
        train_edges = train_data['train_Edges']
        train_labels = train_data['train_Labels']
        test_edges = train_data['test_Edges']
        test_labels = train_data['test_Labels']
    full_shape = train_data['true_md'].shape

    base_md_class = build_fold_md_class(train_data['md_class'], test_edges, test_labels)
    splitter = StratifiedKFold(n_splits=param.kfold, shuffle=True, random_state=param.seed)
    fold_metrics = []

    for fold_index, (sub_train_idx, valid_idx) in enumerate(splitter.split(train_edges, train_labels), start=1):
        edges_train = train_edges[sub_train_idx]
        labels_train = train_labels[sub_train_idx]
        edges_valid = train_edges[valid_idx]
        labels_valid = train_labels[valid_idx]

        edges_core, labels_core, edges_calib, labels_calib = split_train_calibration(
            edges_train, labels_train, seed=param.seed + fold_index, ratio=param.calibration_ratio
        )

        fold_md_class = build_fold_md_class(base_md_class, edges_valid, labels_valid)
        fold_md_class = build_fold_md_class(fold_md_class, edges_calib, labels_calib)
        md_m = torch.from_numpy(fold_md_class).to(device)

        class_prior = estimate_class_prior(int((labels_core == 1).sum()), full_shape)
        conf_matrix = calculate_cv_confidence(sim_views, edges_core, labels_core, full_shape)

        train_dataset = EdgeDataset(edges_core, labels_core, conf_matrix=conf_matrix)
        calib_dataset = EdgeDataset(edges_calib, labels_calib, conf_matrix=conf_matrix, use_label_confidence=False)
        valid_dataset = EdgeDataset(edges_valid, labels_valid, conf_matrix=conf_matrix, use_label_confidence=False)

        loader_args = {
            'batch_size': param.batchSize,
            'num_workers': 0,
            'pin_memory': device.type == 'cuda',
        }
        calib_loader = DataLoader.DataLoader(calib_dataset, shuffle=False, **loader_args)
        valid_loader = DataLoader.DataLoader(valid_dataset, shuffle=False, **loader_args)
        rn_threshold = estimate_rn_threshold(conf_matrix, edges_core, labels_core, param.rn_quantile)

        positive_core = edges_core[labels_core == 1].astype(np.int64)
        excluded_edges = np.vstack((edges_calib.astype(np.int64), edges_valid.astype(np.int64)))
        unlabeled_pool = build_unlabeled_pool(true_md, excluded_edges)

        bag_models = []
        bag_calib_logits = []
        bag_valid_logits = []
        bag_thresholds = []

        print(f'========== Fold {fold_index}/{param.kfold} | class prior={class_prior:.6f} ==========')
        for bag_id in range(getattr(param, 'num_bags', 1)):
            start = time.time()
            bag_edges, bag_labels = build_bag_dataset(
                param,
                positive_core,
                unlabeled_pool,
                conf_matrix,
                seed=param.seed + 101 * fold_index + 1009 * bag_id,
            )
            bag_dataset = EdgeDataset(bag_edges, bag_labels, conf_matrix=conf_matrix, use_label_confidence=True)
            bag_train_loader = DataLoader.DataLoader(bag_dataset, shuffle=True, **loader_args)

            bag_result = train_single_bag(
                param=param,
                sim_data=sim_data,
                md_m=md_m,
                train_loader=bag_train_loader,
                calib_loader=calib_loader,
                device=device,
                class_prior=class_prior,
                rn_threshold=rn_threshold,
                seed_offset=bag_id,
            )
            bag_model = bag_result['model']
            calib_logits, calib_labels = predict_logits(bag_model, sim_data, md_m, calib_loader, device)
            valid_logits, valid_labels = predict_logits(bag_model, sim_data, md_m, valid_loader, device)
            bag_calib_logits.append(calib_logits)
            bag_valid_logits.append(valid_logits)
            bag_models.append(bag_model)
            bag_thresholds.append(bag_result['threshold'])

            elapsed = time.time() - start
            print(
                f'Fold {fold_index} Bag {bag_id + 1}/{param.num_bags} | '
                f'Thr: {bag_result["threshold"]:.2f} | '
                f'{format_metrics(bag_result["metrics"])} | '
                f'Time: {elapsed:.2f}s'
            )

        calib_logits = np.stack(bag_calib_logits, axis=1)
        valid_logits = np.stack(bag_valid_logits, axis=1)
        calib_probs = torch.sigmoid(torch.from_numpy(calib_logits)).cpu().numpy().mean(axis=1)
        valid_probs = torch.sigmoid(torch.from_numpy(valid_logits)).cpu().numpy().mean(axis=1)
        best_threshold = search_best_threshold_from_probs(calib_probs, calib_labels)

        if getattr(param, 'enable_fusion', False):
            aux_features = build_auxiliary_features(
                sim_views,
                edges_core,
                labels_core,
                full_shape,
                param.aux_knn_k,
                semantic_topk=getattr(param, 'semantic_topk', 32),
                enable_semantic=getattr(param, 'use_semantic_fusion_features', True),
            )
            calib_features = make_fusion_features(edges_calib, calib_logits, aux_features, conf_matrix)
            valid_features = make_fusion_features(edges_valid, valid_logits, aux_features, conf_matrix)
            fusion_models = fit_fusion_models(calib_features, labels_calib)
            calib_probs = predict_fusion_probs(fusion_models, calib_features)
            valid_probs = predict_fusion_probs(fusion_models, valid_features)
            best_threshold = search_best_threshold_from_probs(calib_probs, calib_labels)

        valid_metrics = calculate_metrics_from_probs(valid_probs, valid_labels, best_threshold)
        print(f'Fold {fold_index} Final | Thr: {best_threshold:.2f} | {format_metrics(valid_metrics)}')
        fold_metrics.append(valid_metrics)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        key: float(np.mean([item[key] for item in fold_metrics]))
        for key in fold_metrics[0].keys()
    }

    print('\n[Final 5-Fold CV]')
    print(format_metrics(summary))
    return summary


def run_independent_test(sim_data, train_data, param):
    device = param.device
    sim_views = build_confidence_views(sim_data)
    sim_data = move_simdata_to_device(sim_data, device)

    train_edges = train_data['train_Edges']
    train_labels = train_data['train_Labels']
    test_edges = train_data['test_Edges']
    test_labels = train_data['test_Labels']
    full_shape = train_data['true_md'].shape

    conf_matrix = calculate_cv_confidence(sim_views, train_edges, train_labels, full_shape)
    md_class = build_fold_md_class(train_data['md_class'], test_edges, test_labels)
    md_m = torch.from_numpy(md_class).to(device)

    edges_core, labels_core, edges_calib, labels_calib = split_train_calibration(
        train_edges, train_labels, seed=param.seed, ratio=param.calibration_ratio
    )

    md_class = build_fold_md_class(md_class, edges_calib, labels_calib)
    md_m = torch.from_numpy(md_class).to(device)

    conf_matrix = calculate_cv_confidence(sim_views, edges_core, labels_core, full_shape)
    train_dataset = EdgeDataset(edges_core, labels_core, conf_matrix=conf_matrix, use_label_confidence=True)
    calib_dataset = EdgeDataset(edges_calib, labels_calib, conf_matrix=conf_matrix, use_label_confidence=False)
    test_dataset = EdgeDataset(test_edges, test_labels, conf_matrix=conf_matrix, use_label_confidence=False)

    loader_args = {
        'batch_size': param.batchSize,
        'num_workers': 0,
        'pin_memory': device.type == 'cuda',
    }
    train_loader = DataLoader.DataLoader(train_dataset, shuffle=True, **loader_args)
    calib_loader = DataLoader.DataLoader(calib_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader.DataLoader(test_dataset, shuffle=False, **loader_args)

    class_prior = estimate_class_prior(int((labels_core == 1).sum()), full_shape)
    model = SuperedgeLearn(param).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=param.lr, weight_decay=param.weight_decay)
    rn_threshold = estimate_rn_threshold(conf_matrix, edges_core, labels_core, param.rn_quantile)
    criterion = ConfidenceAwarePNULoss(
        pi=class_prior,
        rn_threshold=rn_threshold,
        pu_weight=param.loss_pu_weight,
        sup_weight=param.loss_sup_weight,
        rank_weight=param.loss_rank_weight,
    ).to(device)

    best_state = None
    best_threshold = param.threshold
    best_score = float('-inf')
    stale_epochs = 0
    for epoch in range(1, param.epoch + 1):
        start = time.time()
        train_loss = train_one_epoch(model, sim_data, md_m, train_loader, optimizer, criterion, device)
        calib_logits, calib_labels = predict_logits(model, sim_data, md_m, calib_loader, device)
        calib_threshold = search_best_threshold(calib_logits, calib_labels)
        calib_metrics = calculate_metrics(calib_logits, calib_labels, calib_threshold)
        elapsed = time.time() - start
        print(
            f'Test Mode Epoch {epoch:02d} | Loss: {train_loss:.4f} | '
            f'CalibThr: {calib_threshold:.2f} | {format_metrics(calib_metrics)} | Time: {elapsed:.2f}s'
        )

        score = calib_metrics['auc'] + calib_metrics['aupr'] + calib_metrics['f1']
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_threshold = calib_threshold
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= param.patience:
                print(f'Test mode early stop at epoch {epoch}.')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_logits, test_labels = predict_logits(model, sim_data, md_m, test_loader, device)
    calib_logits, calib_labels = predict_logits(model, sim_data, md_m, calib_loader, device)

    test_probs = torch.sigmoid(torch.from_numpy(test_logits)).cpu().numpy()
    calib_probs = torch.sigmoid(torch.from_numpy(calib_logits)).cpu().numpy()

    if getattr(param, 'enable_fusion', False):
        aux_features = build_auxiliary_features(
            sim_views,
            edges_core,
            labels_core,
            full_shape,
            param.aux_knn_k,
            semantic_topk=getattr(param, 'semantic_topk', 32),
            enable_semantic=getattr(param, 'use_semantic_fusion_features', True),
        )
        calib_features = make_fusion_features(edges_calib, calib_logits, aux_features, conf_matrix)
        test_features = make_fusion_features(test_edges, test_logits, aux_features, conf_matrix)
        fusion_models = fit_fusion_models(calib_features, labels_calib)
        calib_probs = predict_fusion_probs(fusion_models, calib_features)
        test_probs = predict_fusion_probs(fusion_models, test_features)
        best_threshold = search_best_threshold_from_probs(calib_probs, calib_labels)

    metrics = calculate_metrics_from_probs(test_probs, test_labels, best_threshold)
    print('\n[Independent Test]')
    print(f'Thr: {best_threshold:.2f} | {format_metrics(metrics)}')
    return metrics


def train_test(sim_data, train_data, param, state):
    if state == 'valid':
        return run_cross_validation(sim_data, train_data, param)
    return run_independent_test(sim_data, train_data, param)
