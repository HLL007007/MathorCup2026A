"""第二问 v2：目标并行、批量组置换及 Elastic Net 训练内缩尾。

运行（在项目目录中；也可用 --data-root 指定项目）：
  python q2_compound_flavor_v2.py --self-check    # 仅合成数据自检，不读取赛题
  python q2_compound_flavor_v2.py --quick         # 全部目标，3 次抽样，流程试跑
  python q2_compound_flavor_v2.py --jobs 4        # 50 个目标，200 次抽样，同时算 4 个目标
  python q2_compound_flavor_v2.py --targets af   # 仅 Aroma/Flavour 30 项
  python q2_compound_flavor_v2.py --only-target A_esters_isoaa --bootstrap 200
  python q2_compound_flavor_v2.py --en-clip-quantile 0  # 恢复旧版 EN 不缩尾的口径
  python q2_compound_flavor_v2.py --plot-only "运行结果/Q2_v2_..."

依赖：numpy pandas scipy scikit-learn>=1.2 joblib>=1.3 shap>=0.45 matplotlib>=3.6。
本文件复用同目录 q1_without_joint.py 的数据读取、折内预处理，不运行第一问。
需要安装时：python -m pip install numpy pandas scipy "scikit-learn>=1.2" "joblib>=1.3" "shap>=0.45" "matplotlib>=3.6"

v2：--jobs 改为目标进程数；每个模型单线程，避免嵌套并行。组置换复用验证集预处理，
每批最多 32 个置换矩阵，不减少树数、折数、候选参数或重采样次数。随机种子保持不变。
EN 默认按训练集 1%/99% 分位数缩尾，内层调参、外层验证、全数据拟合和重采样均一致；
不删除极端样本、不裁剪感官预测值。该变化会改变 EN 结果，应与旧版分开报告。
多进程会增加内存占用；内存紧张可用 --jobs 2 或 --jobs 1。

方法口径：
* 输入是保留缺失的化学表及附件 4 连续真实感官值；不读取第一问预测标签。
* 附件 2 作为描述性 Spearman 网络；验证/抽样中的相关组仅由训练数据重算。
  reference_module 对应完整 231 节点图的 0.7 聚类；local_module 只在相应折/轮次内有效。
* 去掉 5 个衍生汇总项；具体化合物、辅助指标、名称未明确的分析物分别输出。
* RF/Elastic Net：外层 5 折，内层 3 折选参；各折填补、缩尾、缩放均只拟合训练集。
* RF 以折外/袋外 Tree SHAP 的 mean(abs(phi)) 排序；固定 tree_path_dependent。
  这种归因依赖树路径假设，不能识别强相关变量间的唯一因果贡献。
* Elastic Net 按标准化输入上的绝对系数排序。组联合置换采用同一行置换，
  保留组内依赖，但会破坏组间依赖；每个验证折计算所有训练相关组。
* 分层 Bootstrap 保持各风格原样本数；每轮重新拟合，RF 袋外解释，EN 取拟合系数；超参数由全数据
  CV 预先确定后固定。因此它是固定参数条件下的稳定性，不是独立泛化检验。
* 主排序：Top-3 入选频率、候选内排名中位数、平均归一化贡献、字段名。
  正贡献不足 3 个不补零凑数；并列按字段名决胜并显式标记。贡献区间是重采样
  分布的经验分位数，不是因果效应的置信区间；0.6 频率仅为描述性提示线。
* 风格加入、1% 缩尾、加回汇总项，作为相同外层折/主模型参数的 RF 敏感性对照。
  对照不重新调参，不能视为对照方法各自的最优性能。RF 主分析不缩尾，EN 默认 1% 缩尾。
* 化学证据表只提供已知关系核对线索，不自动声称逐物质文献验证；不计算未知 OAV。

输出：仅在用户运行时创建新的时间戳目录，不覆盖输入与历史结果。包括网络节点/边、
划分、参数/搜索成绩、逐目标折外预测/SHAP、组置换、抽样排名和袋外编号、总排名、
Top-3、辅助指标、跨方法 Jaccard、敏感性与证据核对表，及 PNG/PDF 图。
manifest.json 中 status=complete 才表示计算完成；--quick 的结果不作正式结论。
中断后的各目标目录可检查，暂不提供自动续算。默认不导出数万份拟合模型。

参考：Schreurs et al. (2024), https://doi.org/10.1038/s41467-024-46346-0
Zou & Hastie (2005), https://doi.org/10.1111/j.1467-9868.2005.00503.x
Langfelder & Horvath (2008), https://doi.org/10.1186/1471-2105-9-559
SHAP: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs, parallel_config
from threadpoolctl import threadpool_limits
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.metrics import adjusted_rand_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from q1_without_joint import FoldPreprocessor, load_arrays, resolve_data_root

DERIVED = {'kcalperc', 'esters_sum', 'aroma_hops_sum', 'acids_sum', 'sulfur_sum'}
AUXILIARY = {'color.EBC.', 'pH', 'bitterness', 'GFS', 'unfermentables',
             'protein.g.L.', 'betaglucan.mg.L.', 'iron'}
UNRESOLVED = {'x.methyl.1.octanol', 'y.methyl.1.octanol'}
EPS = 1e-12
SOURCE = 'https://doi.org/10.1038/s41467-024-46346-0'
# 仅提供人工核对线索；没有来源线索的字段保持待核查，不生成虚构解释。
CHEMICAL_CLUES = {
    'ethyl_acetate': '酯；果香/溶剂味线索；同源酯及乙醇可能共变，高浓度效应需单独验证。',
    'isopentyl_acetate': '乙酸异戊酯；香蕉香线索；核对 A/F_esters_isoaa。',
    'phenethyl_acetate': '乙酸苯乙酯；花香线索；感官相关可能受混合物影响。',
    'ethanol..v.v.': '乙醇；酒精感、醇厚感及挥发分配线索；注意低/无醇风格混杂。',
    'glycerol': '甘油；醇厚感线索；与发酵醇及酯共变不等于独立作用。',
    'lactic_acid.mg.L.': '乳酸；酸感线索；检查酸啤酒风格影响。',
    'acetic_acid.g.L.': '乙酸；酸感线索；检查酸啤酒风格影响。',
    'geraniol': '香叶醇；花香/玫瑰及柑橘线索；来源可能包括酒花与香料。',
    'citronellol': '香茅醇；花香/玫瑰及柑橘线索；来源可能包括酒花与香料。',
}


def csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def write_json(data, path):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


class TrainPreprocessor(TransformerMixin, BaseEstimator, FoldPreprocessor):
    """为现有预处理器提供 sklearn 克隆接口，参数估计仍由原实现完成。"""
    def __init__(self, clip_quantile=0.):
        self.clip_quantile = clip_quantile

    def fit(self, X, y=None):
        return FoldPreprocessor.fit(self, X)

    def transform(self, X):
        return FoldPreprocessor.transform(self, X)


class ConstantSafeElasticNet(ElasticNet):
    """常数目标的精确解为零系数，避免零容差触发无意义的未收敛警告。"""
    def fit(self, X, y, sample_weight=None, check_input=True):
        if np.ptp(y) == 0:
            self.coef_ = np.zeros(X.shape[1], dtype=float)
            self.intercept_ = float(y[0]) if self.fit_intercept else 0.
            self.n_features_in_, self.n_iter_, self.dual_gap_ = X.shape[1], 0, 0.
            return self
        return super().fit(X, y, sample_weight=sample_weight, check_input=check_input)


def balanced_splits(styles, folds, seed):
    """小风格样本少于折数时仍可用；轮转分配使总体及各风格尽量均衡。"""
    if not 2 <= folds <= len(styles) // 2:
        raise ValueError('每折至少需要两个验证样本，请减少折数。')
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(folds)]
    offset = 0
    groups = np.unique(styles).copy()
    rng.shuffle(groups)
    for style in groups:
        ids = np.flatnonzero(styles == style)
        rng.shuffle(ids)
        for i, idx in enumerate(ids):
            buckets[(offset + i) % folds].append(int(idx))
        offset = (offset + len(ids)) % folds
    all_ids = np.arange(len(styles))
    return [(np.setdiff1d(all_ids, b), np.sort(b)) for b in buckets]


def bootstrap_indices(styles, seed):
    rng = np.random.default_rng(seed)
    train = np.concatenate([rng.choice(ids, size=len(ids), replace=True)
                            for s in np.unique(styles)
                            for ids in [np.flatnonzero(styles == s)]])
    rng.shuffle(train)
    oob = np.setdiff1d(np.arange(len(styles)), np.unique(train))
    if len(oob) < 2:
        raise ValueError('Bootstrap 袋外样本不足 2 个；本次抽样无效，请检查数据规模。')
    return train, oob


def validate_matrix(frame, features):
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError('相关矩阵名称有重复。')
    if set(frame.index) != set(features) or set(frame.columns) != set(features):
        raise ValueError('附件 2 行列名称与化学表不能一一对应。')
    r = frame.loc[features, features].to_numpy(dtype=float)
    if (not np.isfinite(r).all() or np.max(np.abs(r)) > 1 + 1e-8
            or not np.allclose(r, r.T, atol=1e-8) or not np.allclose(np.diag(r), 1)):
        raise ValueError('相关矩阵必须有限、对称、对角为 1，且系数在 [-1,1]。')
    return np.clip((r + r.T) / 2, -1, 1)


def signed_correlation(x):
    # 重采样中的常数/全空列无可定义相关，按无边处理；不把它们合成一个模块。
    r = pd.DataFrame(x).corr(method='spearman', min_periods=3).fillna(0).to_numpy()
    r = np.clip((r + r.T) / 2, -1, 1)
    np.fill_diagonal(r, 1.)
    return r


def cluster_labels(r, cutoff=.7):
    if len(r) == 1:
        return np.ones(1, dtype=int)
    distance = np.clip(1 - np.abs(r), 0, 1)
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=True), method='average')
    return fcluster(tree, t=1-cutoff, criterion='distance')


def positive_top(scores, names, eligible, k=3):
    ids = np.flatnonzero(eligible & np.isfinite(scores) & (scores > EPS))
    return sorted(ids, key=lambda j: (-scores[j], names[j]))[:k]


def rank_candidates(scores, eligible):
    ranks = np.full(len(scores), np.nan)
    count = int(np.sum(eligible))
    ranks[eligible] = count + 1
    ids = np.flatnonzero(eligible & np.isfinite(scores) & (scores > EPS))
    ranks[ids] = pd.Series(scores[ids]).rank(ascending=False, method='average').to_numpy()
    return ranks


def boundary_tie(scores, chosen, eligible):
    if len(chosen) != 3:
        return False
    remaining = eligible.copy()
    remaining[chosen] = False
    return bool(np.any(np.isclose(scores[remaining], scores[chosen[-1]], rtol=1e-9, atol=EPS)))


def normalize(scores):
    if not np.isfinite(scores).all():
        raise ValueError('重要性分数存在非有限值，拒绝继续排名。')
    total = np.sum(np.abs(scores))
    return np.abs(scores) / total if total > EPS else np.zeros_like(scores)


def jaccard(a, b):
    union = set(a) | set(b)
    return len(set(a) & set(b)) / len(union) if union else np.nan


def joint_permute(x, columns, rng):
    changed = x.copy()
    changed[:, columns] = x[rng.permutation(len(x))][:, columns]
    return changed


def group_permutation_changes(model, x, y, modules, repeats, seed, base_mae, batch_size=32):
    """逐列预处理与行置换可交换；保持旧版每组随机序列，合并小批预测。"""
    pipe = model.regressor_
    values = pipe['prep'].transform(x)
    groups = np.unique(modules)
    changes = np.empty((len(groups), repeats))
    pending, positions = [], []

    def flush():
        pred = pipe['model'].predict(np.concatenate(pending, axis=0))
        pred = model.transformer_.inverse_transform(pred.reshape(-1, 1)).reshape(len(pending), len(y))
        if not np.isfinite(pred).all():
            raise ValueError('组置换预测含非有限值。')
        losses = np.abs(pred - y[None, :]).mean(axis=1) - base_mae
        for position, loss in zip(positions, losses):
            changes[position] = loss
        pending.clear()
        positions.clear()

    for i, module in enumerate(groups):
        cols = np.flatnonzero(modules == module)
        rng = np.random.default_rng(seed + int(module))
        for repeat in range(repeats):
            pending.append(joint_permute(values, cols, rng))
            positions.append((i, repeat))
            if len(pending) == batch_size:
                flush()
    if pending:
        flush()
    return changes


def rho(x, y):
    pair = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(pair) < 3 or pair.x.nunique() < 2 or pair.y.nunique() < 2:
        return np.nan
    return float(pair.corr(method='spearman').iloc[0, 1])


def estimator(name, clip, seed, jobs):
    model = (RandomForestRegressor(n_estimators=150, max_depth=8, random_state=seed, n_jobs=jobs)
             if name == 'RF' else ConstantSafeElasticNet(max_iter=50000, tol=1e-5, selection='cyclic'))
    pipe = Pipeline([('prep', TrainPreprocessor(clip)), ('model', model)])
    return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())


def grids(name, quick):
    prefix = 'regressor__model__'
    if name == 'RF':
        values = {'n_estimators': [30 if quick else 150], 'max_depth': [8],
                  'min_samples_leaf': [3] if quick else [2, 5],
                  'max_features': [.7] if quick else [.5, .8]}
    else:
        values = {'alpha': [.01, .1] if quick else np.logspace(-3, 1, 9).tolist(),
                  'l1_ratio': [.5] if quick else [.1, .5, .9]}
    return {prefix + k: v for k, v in values.items()}


def tune(name, x, y, styles, args, seed):
    cv = balanced_splits(styles, args.inner_folds, seed)
    clip = args.en_clip_quantile if name == 'EN' else 0.
    search = GridSearchCV(estimator(name, clip, seed, 1), grids(name, args.quick),
                          scoring='neg_mean_absolute_error', cv=cv, n_jobs=1,
                          error_score='raise', refit=True)
    # 未收敛会使重要性排序不可信，停止并报告，避免忽略后继续输出排名。
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        search.fit(x, y)
    return search.best_estimator_, search.best_params_, pd.DataFrame({
        'parameters': [json.dumps(p, sort_keys=True) for p in search.cv_results_['params']],
        'inner_MAE': -search.cv_results_['mean_test_score'],
        'inner_MAE_std': search.cv_results_['std_test_score'],
        'rank': search.cv_results_['rank_test_score'],
    })


def fit_fixed(name, params, x, y, clip, seed, jobs=1):
    model = estimator(name, clip, seed, jobs).set_params(**params)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        return model.fit(x, y)


def explain_rf(model, x):
    import shap
    pipe = model.regressor_
    values = pipe['prep'].transform(x)
    explainer = shap.TreeExplainer(pipe['model'], feature_perturbation='tree_path_dependent')
    phi = np.asarray(explainer.shap_values(values, check_additivity=True))
    if phi.ndim == 3 and phi.shape[-1] == 1:
        phi = phi[:, :, 0]
    if phi.shape != values.shape or not np.isfinite(phi).all():
        raise ValueError(f'SHAP 维度或数值错误：{phi.shape}，期望 {values.shape}')
    base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    if not np.allclose(base + phi.sum(axis=1), pipe['model'].predict(values), rtol=1e-5, atol=1e-6):
        raise ValueError('SHAP 加性核验失败。')
    return phi * float(model.transformer_.scale_[0])


def coefficients(model):
    return model.regressor_['model'].coef_ * float(model.transformer_.scale_[0])


def metrics(y, prediction, baseline):
    if not np.isfinite(prediction).all():
        raise ValueError('预测存在非有限值。')
    mae = mean_absolute_error(y, prediction)
    base_mae = mean_absolute_error(y, baseline)
    return {'MAE': mae, 'RMSE': np.sqrt(mean_squared_error(y, prediction)),
            'R2': r2_score(y, prediction) if np.var(y) > EPS else np.nan,
            'baseline_MAE': base_mae,
            'MAE_improvement': 1 - mae/base_mae if base_mae > EPS else np.nan}


def load_inputs(args):
    x, y, meta, all_names, targets, files = load_arrays(args.data_root, args.targets)
    if args.only_target:
        unknown = set(args.only_target) - set(targets)
        if unknown:
            raise ValueError(f'目标不在所选范围内：{sorted(unknown)}')
        ids = [targets.index(t) for t in dict.fromkeys(args.only_target)]
        y, targets = y[:, ids], [targets[i] for i in ids]
    if meta.tasting_category_fine.isna().any():
        raise ValueError('风格标签存在缺失。')
    files['correlation'] = args.data_root / '处理结果/最终数据/2_化合物相关矩阵.csv'
    files['feature_metadata'] = args.data_root / '处理结果/处理记录/结构整理（第4版）/05_化合物字段说明.csv'
    frame = pd.read_csv(files['correlation'], index_col=0, encoding='utf-8-sig')
    reference = validate_matrix(frame, all_names)
    catalog = pd.read_csv(files['feature_metadata'], encoding='utf-8-sig')
    if catalog.field_name_R.duplicated().any() or set(catalog.field_name_R) != set(all_names):
        raise ValueError('化学字段说明不能与输入一一对应。')
    if (DERIVED | AUXILIARY | UNRESOLVED) - set(all_names):
        raise ValueError('字段分类与当前数据版本不匹配，需重新核对。')
    catalog = catalog.set_index('field_name_R').loc[all_names].reset_index()
    catalog['kind'] = ['derived' if n in DERIVED else 'auxiliary' if n in AUXILIARY
                       else 'unresolved' if n in UNRESOLVED else 'compound' for n in all_names]
    catalog['in_main_model'] = ~catalog.field_name_R.isin(DERIVED)
    catalog['eligible_top3'] = catalog.kind.eq('compound')
    keep = catalog.in_main_model.to_numpy()
    names = list(np.array(all_names)[keep])
    main_catalog = catalog.loc[keep].reset_index(drop=True)
    # 原始缺失保留；与已填补全表重算的相关系数含义不同。
    observed = pd.DataFrame(x).corr(method='spearman', min_periods=3).to_numpy()
    if not np.isfinite(observed).all():
        raise ValueError('原始全样本相关存在不可定义项，需核对附件 2 来源。')
    discrepancy = float(np.max(np.abs(observed - reference)))
    if discrepancy > 1e-6:
        raise ValueError(f'附件 2 与当前原始数据 Spearman 不吻合（最大差 {discrepancy}）。')
    return dict(x=x[:, keep], x_all=x, y=y, meta=meta, styles=meta.tasting_category_fine.astype(str).to_numpy(),
                names=names, all_names=all_names, targets=targets, catalog=catalog,
                main_catalog=main_catalog, reference=reference, files=files,
                correlation_max_difference=discrepancy)


def export_network(data, out):
    r, names = data['reference'], data['all_names']
    stats = []
    reference_groups = cluster_labels(r, .7)
    for cutoff in (.5, .6, .7):
        adjacency = np.abs(r) >= cutoff
        np.fill_diagonal(adjacency, False)
        labels = cluster_labels(r, cutoff)
        i, j = np.where(np.triu(adjacency, 1))
        csv(pd.DataFrame({'source': np.array(names)[i], 'target': np.array(names)[j],
                          'rho': r[i, j], 'weight': np.abs(r[i, j]),
                          'sign': np.where(r[i, j] >= 0, 'positive', 'negative')}),
            out / f'edges_{cutoff:.1f}.csv')
        csv(pd.DataFrame({'feature': names, 'local_module': labels}), out / f'modules_{cutoff:.1f}.csv')
        stats.append({'cutoff': cutoff, 'nodes': len(names), 'edges': len(i),
                      'isolates': int((adjacency.sum(axis=0) == 0).sum()),
                      'components': connected_components(adjacency, directed=False, return_labels=False),
                      'clusters': len(np.unique(labels)),
                      'module_ARI_vs_0.7': adjusted_rand_score(reference_groups, labels)})
    nodes = data['catalog'].copy()
    nodes['reference_module'] = reference_groups
    weight = np.abs(r) * (np.abs(r) >= .6)
    np.fill_diagonal(weight, 0)
    nodes['degree_0.6'] = (weight > 0).sum(axis=0)
    nodes['strength_0.6'] = weight.sum(axis=0)
    csv(nodes, out / 'nodes.csv')
    csv(pd.DataFrame(stats), out / 'threshold_sensitivity.csv')
    return nodes


def save_sampling(data, args, out):
    styles, beers = data['styles'], data['meta'].beer.to_numpy()
    folds = balanced_splits(styles, args.folds, args.seed)
    fold_rows, inner_rows, module_rows, boot_rows = [], [], [], []
    fold_modules, draws = [], []
    for k, (train, valid) in enumerate(folds, 1):
        for role, ids in [('train', train), ('validation', valid)]:
            fold_rows.extend({'fold': k, 'role': role, 'beer': int(beers[i]), 'style': styles[i]} for i in ids)
        inner = balanced_splits(styles[train], args.inner_folds, args.seed + k)
        for h, (it, iv) in enumerate(inner, 1):
            for role, ids in [('train', train[it]), ('validation', train[iv])]:
                inner_rows.extend({'outer_fold': k, 'inner_fold': h, 'role': role,
                                   'beer': int(beers[i])} for i in ids)
        labels = cluster_labels(signed_correlation(data['x'][train]), args.cluster_cutoff)
        fold_modules.append(labels)
        module_rows.extend({'phase': 'outer_cv', 'repeat': k, 'feature': n, 'local_module': int(m)}
                           for n, m in zip(data['names'], labels))
    for b in range(args.bootstrap):
        train, oob = bootstrap_indices(styles, args.seed + 10000 + b)
        labels = cluster_labels(signed_correlation(data['x'][train]), args.cluster_cutoff)
        draws.append((train, oob, labels))
        unique, counts = np.unique(train, return_counts=True)
        boot_rows.extend({'repeat': b+1, 'role': 'train', 'beer': int(beers[i]), 'count': int(c)}
                         for i, c in zip(unique, counts))
        boot_rows.extend({'repeat': b+1, 'role': 'oob', 'beer': int(beers[i]), 'count': 0} for i in oob)
        module_rows.extend({'phase': 'bootstrap', 'repeat': b+1, 'feature': n, 'local_module': int(m)}
                           for n, m in zip(data['names'], labels))
    final_inner = balanced_splits(styles, args.inner_folds, args.seed + 9000)
    for h, (it, iv) in enumerate(final_inner, 1):
        for role, ids in [('train', it), ('validation', iv)]:
            inner_rows.extend({'outer_fold': 'full_data_parameter_selection', 'inner_fold': h,
                               'role': role, 'beer': int(beers[i])} for i in ids)
    csv(pd.DataFrame(fold_rows), out / 'outer_splits.csv')
    csv(pd.DataFrame(inner_rows), out / 'inner_splits.csv')
    csv(pd.DataFrame(boot_rows), out / 'bootstrap_samples.csv')
    csv(pd.DataFrame(module_rows), out / 'training_modules.csv')
    return folds, fold_modules, draws


def style_features(train_style, valid_style):
    levels = np.unique(train_style)
    return ((train_style[:, None] == levels).astype(float),
            (valid_style[:, None] == levels).astype(float))


def summarize_bootstrap(scores, names, eligible):
    normalized = np.array([normalize(row) for row in scores])
    ranks = np.array([rank_candidates(row, eligible) for row in normalized])
    selected = np.zeros_like(normalized, dtype=bool)
    for b, row in enumerate(normalized):
        selected[b, positive_top(row, names, eligible)] = True
    return pd.DataFrame({
        'feature': names, 'eligible': eligible,
        'top3_frequency': selected.mean(axis=0),
        'rank_median': np.median(ranks, axis=0),
        'importance_share_mean': normalized.mean(axis=0),
        'importance_share_p025': np.quantile(normalized, .025, axis=0),
        'importance_share_p975': np.quantile(normalized, .975, axis=0),
    })


def consensus(summary, available=None, k=3):
    pool = summary.loc[summary.eligible & (summary.top3_frequency > 0)
                       & (summary.importance_share_mean > EPS)].copy()
    if available is not None:
        pool = pool.loc[pool.feature.isin(available)]
    return pool.sort_values(['top3_frequency', 'rank_median', 'importance_share_mean', 'feature'],
                            ascending=[False, True, False, True]).head(k)


def analyze_target(data, target_index, args, folds, fold_modules, draws, out):
    target = data['targets'][target_index]
    x, y, names, styles = data['x'], data['y'][:, target_index], data['names'], data['styles']
    catalog = data['main_catalog']
    eligible = catalog.eligible_top3.to_numpy()
    auxiliary = catalog.kind.eq('auxiliary').to_numpy()
    out.mkdir(parents=True, exist_ok=False)
    n, p = x.shape
    oof = {'RF': np.full(n, np.nan), 'EN': np.full(n, np.nan)}
    phi_oof = np.full((n, p), np.nan)
    en_cv = []
    baseline = np.full(n, np.nan)
    variants = ['with_style', 'clip_1pct', 'with_derived'] if not args.no_sensitivity else []
    variant_predictions = {v: np.full(n, np.nan) for v in variants}
    variant_scores = {v: np.zeros(p) for v in variants}
    fold_metrics, group_rows, params_rows, search_rows = [], [], [], []
    for k, ((train, valid), modules) in enumerate(zip(folds, fold_modules), 1):
        baseline[valid] = np.median(y[train])
        for method in ('RF', 'EN'):
            model, parameters, search = tune(method, x[train], y[train], styles[train], args, args.seed+k)
            params_rows.append({'phase': 'outer_cv', 'fold': k, 'method': method,
                                'parameters': json.dumps(parameters, sort_keys=True)})
            search_rows.append(search.assign(phase='outer_cv', fold=k, method=method))
            pred = model.predict(x[valid])
            oof[method][valid] = pred
            fold_metrics.append({'target': target, 'method': method, 'fold': k,
                                 **metrics(y[valid], pred, baseline[valid])})
            if method == 'EN':
                en_cv.append(np.abs(coefficients(model)) * len(valid)/n)
                continue
            phi_oof[valid] = explain_rf(model, x[valid])
            base_mae = mean_absolute_error(y[valid], pred)
            group_changes = group_permutation_changes(model, x[valid], y[valid], modules,
                                                      args.permutations, args.seed+k*1000, base_mae)
            for module, changes in zip(np.unique(modules), group_changes):
                cols = np.flatnonzero(modules == module)
                group_rows.append({'target': target, 'fold': k, 'local_module': int(module),
                                   'members': '|'.join(names[c] for c in cols), 'size': len(cols),
                                   'delta_MAE_mean': np.mean(changes), 'delta_MAE_std': np.std(changes),
                                   'repeats': args.permutations})
            for variant in variants:
                xt, xv, clip = x[train], x[valid], 0.
                if variant == 'with_style':
                    st, sv = style_features(styles[train], styles[valid])
                    xt, xv = np.column_stack([xt, st]), np.column_stack([xv, sv])
                elif variant == 'clip_1pct':
                    clip = .01
                else:
                    xt, xv = data['x_all'][train], data['x_all'][valid]
                modified = fit_fixed('RF', parameters, xt, y[train], clip, args.seed+k)
                variant_predictions[variant][valid] = modified.predict(xv)
                ph = explain_rf(modified, xv)
                if variant == 'with_derived':
                    ph = ph[:, [data['all_names'].index(name) for name in names]]
                else:
                    ph = ph[:, :p]
                variant_scores[variant] += np.abs(ph).sum(axis=0) / n
        print(f'  {target}: 外层 {k}/{args.folds} 完成', flush=True)
    cv_scores = {'RF': np.abs(phi_oof).mean(axis=0), 'EN': np.sum(en_cv, axis=0)}
    performance = pd.DataFrame([{'target': target, 'method': m, **metrics(y, pred, baseline)}
                                for m, pred in {**oof, **variant_predictions}.items()])
    predictions = data['meta'].copy()
    predictions['observed'] = y
    predictions['training_median'] = baseline
    for method, pred in {**oof, **variant_predictions}.items():
        predictions[method] = pred
    csv(predictions, out / 'oof_predictions.csv')
    shap_frame = pd.DataFrame(phi_oof, columns=names)
    shap_frame.insert(0, 'beer', data['meta'].beer.to_numpy())
    csv(shap_frame, out / 'oof_shap.csv')
    csv(performance, out / 'performance.csv')
    csv(pd.DataFrame(fold_metrics), out / 'fold_metrics.csv')
    csv(pd.DataFrame(group_rows), out / 'group_permutation.csv')

    final_parameters, full_coefficient = {}, None
    for method in ('RF', 'EN'):
        model, parameters, search = tune(method, x, y, styles, args, args.seed+9000)
        final_parameters[method] = parameters
        params_rows.append({'phase': 'full_data_for_bootstrap', 'fold': 0, 'method': method,
                            'parameters': json.dumps(parameters, sort_keys=True)})
        search_rows.append(search.assign(phase='full_data_for_bootstrap', fold=0, method=method))
        if method == 'EN':
            full_coefficient = coefficients(model)
    csv(pd.DataFrame(params_rows), out / 'selected_parameters.csv')
    csv(pd.concat(search_rows, ignore_index=True), out / 'parameter_search.csv')

    boot_scores = {m: np.zeros((args.bootstrap, p)) for m in ('RF', 'EN')}
    selections, oob_metrics = [], []
    for b, (train, oob_ids, modules) in enumerate(draws):
        for method in ('RF', 'EN'):
            clip = args.en_clip_quantile if method == 'EN' else 0.
            fitted = fit_fixed(method, final_parameters[method], x[train], y[train], clip,
                               args.seed+20000+b)
            score = (np.abs(explain_rf(fitted, x[oob_ids])).mean(axis=0) if method == 'RF'
                     else np.abs(coefficients(fitted)))
            boot_scores[method][b] = score
            share = normalize(score)
            chosen = positive_top(share, names, eligible)
            tied = boundary_tie(share, chosen, eligible)
            for rank, j in enumerate(chosen, 1):
                selections.append({'repeat': b+1, 'method': method, 'rank': rank, 'feature': names[j],
                                   'importance_share': share[j], 'boundary_tie': tied,
                                   'local_module': int(modules[j]),
                                   'module_members': '|'.join(names[c] for c in np.flatnonzero(modules == modules[j]))})
            oob_metrics.append({'repeat': b+1, 'method': method, 'oob_n': len(oob_ids),
                                'selected_count': len(chosen),
                                **metrics(y[oob_ids], fitted.predict(x[oob_ids]),
                                          np.full(len(oob_ids), np.median(y[train])))})
        if (b+1) % 10 == 0 or b+1 == args.bootstrap:
            print(f'  {target}: Bootstrap {b+1}/{args.bootstrap}', flush=True)
    csv(pd.DataFrame(selections, columns=['repeat', 'method', 'rank', 'feature', 'importance_share',
                                        'boundary_tie', 'local_module', 'module_members']), out / 'bootstrap_top3.csv')
    csv(pd.DataFrame(oob_metrics), out / 'bootstrap_oob_diagnostics.csv')
    summaries, selected, auxiliary_selected, boundary_rows = [], [], [], []
    references = data['network_nodes'].set_index('field_name_R').loc[names, 'reference_module'].to_numpy()
    for method in ('RF', 'EN'):
        wide = pd.DataFrame(boot_scores[method], columns=names)
        wide.insert(0, 'repeat', np.arange(1, args.bootstrap+1))
        csv(wide, out / f'bootstrap_importance_{method}.csv')
        summary = summarize_bootstrap(boot_scores[method], names, eligible)
        summary['target'], summary['method'] = target, method
        summary['cv_importance'] = cv_scores[method]
        summary['cv_share'] = normalize(cv_scores[method])
        summary['spearman_rho'] = [rho(x[:, j], y) for j in range(p)]
        summary['shap_x_rho'] = [rho(x[:, j], phi_oof[:, j]) for j in range(p)]
        summary['EN_full_coefficient'] = full_coefficient
        summary['reference_module'] = references
        summary['chemical_name'] = catalog.chemical_name.to_numpy()
        summary['kind'] = catalog.kind.to_numpy()
        summary['unit'] = catalog.unit.to_numpy()
        summary['representation_status'] = catalog['核验状态'].to_numpy()
        group_values = {name: [] for name in names}
        for group in group_rows:
            for name in group['members'].split('|'):
                group_values[name].append(group['delta_MAE_mean'])
        # 各折的组可变化：此处按化合物追踪其所在训练组，而非合并局部组编号。
        summary['associated_group_delta_MAE_mean_RF'] = [np.mean(group_values[name]) for name in names]
        summary['associated_group_positive_fold_fraction_RF'] = [np.mean(np.array(group_values[name]) > 0)
                                                                for name in names]
        perf = performance.loc[performance.method.eq(method)].iloc[0]
        summary['OOF_R2'] = perf.R2
        summary['OOF_MAE_improvement'] = perf.MAE_improvement
        summary['prediction_supported'] = bool(perf.R2 > 0 and perf.MAE_improvement > 0)
        available = [names[j] for j in np.flatnonzero(cv_scores[method] > EPS)]
        first_four = consensus(summary, available, k=4)
        top = first_four.head(3).copy()
        gap = (float(first_four.iloc[2].top3_frequency - first_four.iloc[3].top3_frequency)
               if len(first_four) == 4 else np.nan)
        boundary_rows.append({'target': target, 'method': method, 'selected_count': len(top),
                              'fourth_candidate': first_four.iloc[3].feature if len(first_four) == 4 else '',
                              'frequency_gap_3_to_4': gap,
                              'frequency_tie_at_boundary': bool(gap == 0)})
        top['frequency_gap_3_to_4'] = gap
        top['frequency_tie_at_boundary'] = bool(gap == 0)
        top['rank'] = np.arange(1, len(top)+1)
        top['stability_flag'] = np.where(top.top3_frequency >= .6, 'frequency>=0.6 (descriptive)', 'unstable')
        summaries.append(summary)
        selected.append(top)
        aux = summarize_bootstrap(boot_scores[method], names, auxiliary)
        aux_top = consensus(aux, available)
        aux_top = aux_top.merge(summary[['feature', 'target', 'method', 'chemical_name', 'OOF_R2']], on='feature')
        aux_top['rank'] = np.arange(1, len(aux_top)+1)
        auxiliary_selected.append(aux_top)
    summary_all = pd.concat(summaries, ignore_index=True)
    top_all = pd.concat(selected, ignore_index=True)
    aux_all = pd.concat(auxiliary_selected, ignore_index=True)
    csv(summary_all, out / 'importance_summary.csv')
    csv(top_all, out / 'top3.csv')
    csv(aux_all, out / 'auxiliary_top3.csv')
    csv(pd.DataFrame(boundary_rows), out / 'selection_boundaries.csv')
    rf_top = top_all.loc[top_all.method.eq('RF'), 'feature'].tolist()
    en_top = top_all.loc[top_all.method.eq('EN'), 'feature'].tolist()
    rf_cv = [names[j] for j in positive_top(normalize(cv_scores['RF']), names, eligible)]
    comparison = {'target': target, 'RF_count': len(rf_top), 'EN_count': len(en_top),
                  'RF_top3': '|'.join(rf_top), 'EN_top3': '|'.join(en_top),
                  'consensus_Jaccard': jaccard(rf_top, en_top),
                  'RF_consensus_vs_OOF_Jaccard': jaccard(rf_top, rf_cv)}
    sensitivity = []
    for variant in variants:
        alt = [names[j] for j in positive_top(normalize(variant_scores[variant]), names, eligible)]
        for j, name in enumerate(names):
            sensitivity.append({'target': target, 'variant': variant, 'feature': name,
                                'main_cv_importance': cv_scores['RF'][j],
                                'variant_cv_importance': variant_scores[variant][j],
                                'variant_top3': name in alt, 'top3_Jaccard_vs_main_OOF': jaccard(rf_cv, alt)})
    sensitivity = pd.DataFrame(sensitivity, columns=['target', 'variant', 'feature', 'main_cv_importance',
                                                    'variant_cv_importance', 'variant_top3', 'top3_Jaccard_vs_main_OOF'])
    csv(sensitivity, out / 'sensitivity.csv')
    evidence = []
    reference_corr = data['reference']
    for _, row in top_all.loc[top_all.method.eq('RF')].iterrows():
        reference_j = data['all_names'].index(row.feature)
        correlated = sorted([(name, float(reference_corr[reference_j, data['all_names'].index(name)]))
                             for name in names if name != row.feature], key=lambda pair: -abs(pair[1]))[:3]
        evidence.append({**row.to_dict(),
                         'correlated_neighbors': '|'.join(f'{name}:{value:.3f}' for name, value in correlated),
                         'chemical_clue': CHEMICAL_CLUES.get(row.feature, '尚未收录逐物质文献线索，需要人工核查气味、来源和浓度。'),
                         'reference_to_check': SOURCE if row.feature in CHEMICAL_CLUES else '',
                         'verification_status': '候选关联；化学证据待人工核实，非因果结论',
                         'OAV_status': '未计算：需先确认物理浓度、单位与啤酒介质阈值'})
    evidence_frame = pd.DataFrame(evidence, columns=list(top_all.columns) +
                                 ['correlated_neighbors', 'chemical_clue', 'reference_to_check', 'verification_status', 'OAV_status'])
    csv(evidence_frame, out / 'chemical_evidence_checklist.csv')
    return performance, top_all, aux_all, summary_all, comparison, evidence_frame


def plot_results(root):
    """只读取已保存的源数据；单图单轴，供运行后人工检查，不宣称投稿级视觉验收。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') not in {'computed', 'complete'}:
        raise ValueError('只允许绘制已完成计算的结果目录。')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)

    def save(fig, name):
        fig.savefig(figures / f'{name}.png', dpi=300, bbox_inches='tight')
        fig.savefig(figures / f'{name}.pdf', bbox_inches='tight')
        plt.close(fig)

    nodes = pd.read_csv(root / 'network/nodes.csv')
    nodes = nodes.sort_values(['reference_module', 'field_name_R']).reset_index(drop=True)
    edges = pd.read_csv(root / 'network/edges_0.6.csv')
    angle = np.arange(len(nodes)) * 2*np.pi/len(nodes)
    positions = np.column_stack([np.cos(angle), np.sin(angle)])
    lookup = {name: i for i, name in enumerate(nodes.field_name_R)}
    fig, ax = plt.subplots(figsize=(9, 9), layout='constrained')
    segments = [[positions[lookup[e.source]], positions[lookup[e.target]]] for e in edges.itertuples()]
    ax.add_collection(LineCollection(segments, colors=np.where(edges.rho > 0, '#648AA3', '#BE8061'),
                                     linewidths=.3 + .6*edges.weight.to_numpy(), alpha=.13))
    sizes = 12 + 60*nodes['strength_0.6'].to_numpy()/max(nodes['strength_0.6'].max(), 1)
    for kind, color in [('compound', '#648AA3'), ('auxiliary', '#BE8061'),
                        ('derived', '#7C6A93'), ('unresolved', '#777777')]:
        mask = nodes.kind.eq(kind).to_numpy()
        ax.scatter(positions[mask, 0], positions[mask, 1], s=sizes[mask], color=color, label=kind, zorder=3)
    ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), aspect='equal')
    ax.axis('off')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, .02), ncol=2)
    ax.set_title(f'Chemical association network: {len(nodes)} nodes, {len(edges)} edges\n'
                 'Spearman |rho| >= 0.6; blue: positive, orange: negative; size: strength')
    save(fig, '01_association_network')
    csv(nodes.assign(plot_x=positions[:, 0], plot_y=positions[:, 1]), figures / 'network_positions.csv')

    top = pd.read_csv(root / 'top3.csv')
    main = top.loc[top.method.eq('RF')].copy()
    targets = manifest['targets']
    for page, start in enumerate(range(0, len(targets), 10), 1):
        subset = main.loc[main.target.isin(targets[start:start+10])].sort_values(['target', 'rank'])
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, max(4, len(subset)*.29)), layout='constrained')
        labels = [f'{r.target} | {r.feature}' for r in subset.itertuples()]
        ypos = np.arange(len(subset))
        ax.barh(ypos, subset.top3_frequency, color=np.where(subset.prediction_supported, '#648AA3', '#B8B8B8'))
        ax.set_yticks(ypos, labels, fontsize=8)
        ax.invert_yaxis()
        ax.set(xlim=(0, 1), xlabel='Bootstrap Top-3 inclusion frequency',
               title=f'RF candidates | B={manifest["parameters"]["bootstrap"]} | gray: weak prediction evidence')
        save(fig, f'02_inclusion_frequency_{page:02d}')
        csv(subset, figures / f'inclusion_frequency_{page:02d}_source.csv')
    performance = pd.read_csv(root / 'performance.csv')
    eligible_targets = performance.loc[performance.method.eq('RF') & (performance.R2 > 0)
                                       & (performance.MAE_improvement > 0)].sort_values('R2', ascending=False)
    examples = eligible_targets.target.head(3).tolist()
    for target in examples:
        folder = root / 'targets' / target
        x = pd.read_csv(root / 'input_features.csv').set_index('beer')
        ph = pd.read_csv(folder / 'oof_shap.csv').set_index('beer').loc[x.index]
        for row in main.loc[main.target.eq(target)].itertuples():
            valid = x[row.feature].notna()
            fig, ax = plt.subplots(figsize=(6.4, 4.3), layout='constrained')
            ax.scatter(x.loc[valid, row.feature], ph.loc[valid, row.feature], s=15, alpha=.65, color='#648AA3')
            ax.axhline(0, color='#999999', linewidth=.7)
            ax.set(xlabel=f'{row.feature} (supplied representation)', ylabel='OOF SHAP (sensory units)',
                   title=f'{target}: {row.feature}\nn={int(valid.sum())}; missing input omitted, no causal claim')
            save(fig, f'03_{target}_{row.rank}_{row.feature}')
            csv(pd.DataFrame({'beer': x.index[valid], 'x': x.loc[valid, row.feature],
                              'oof_shap': ph.loc[valid, row.feature]}),
                figures / f'03_{target}_{row.rank}_source.csv')


def self_check():
    """显式选择才执行；合成数据核验，不创建文件或读取赛题。"""
    train = np.array([[1., np.nan], [3., np.nan], [5., np.nan]])
    pre = TrainPreprocessor().fit(train)
    before = pre.imputer.statistics_.copy()
    values = pre.transform(np.array([[10000., 7.], [np.nan, np.nan]]))
    np.testing.assert_array_equal(before, pre.imputer.statistics_)
    assert values.shape == (2, 2) and np.isfinite(values).all()
    assert abs(values[1, 0]) < EPS
    styles = np.array(['a']*8 + ['b']*3 + ['c'])
    selected, oob = bootstrap_indices(styles, 7)
    assert pd.Series(styles[selected]).value_counts().to_dict() == pd.Series(styles).value_counts().to_dict()
    assert not set(selected) & set(oob)
    assert set(selected) | set(oob) == set(range(len(styles)))
    folds = balanced_splits(styles, 3, 4)
    np.testing.assert_array_equal(np.sort(np.concatenate([v for _, v in folds])), np.arange(len(styles)))
    assert all(not set(t) & set(v) for t, v in folds)
    z = np.column_stack([np.arange(10), -np.arange(10), np.ones(10)])
    r = signed_correlation(z)
    labels = cluster_labels(r, .7)
    assert labels[0] == labels[1] and labels[0] != labels[2]
    permuted = joint_permute(z, [0, 1], np.random.default_rng(4))
    np.testing.assert_array_equal(permuted[:, 0], -permuted[:, 1])
    np.testing.assert_array_equal(permuted[:, 2], z[:, 2])
    assert not np.array_equal(permuted[:, 0], z[:, 0])
    names, eligible = ['d', 'b', 'a', 'c'], np.ones(4, bool)
    assert positive_top(np.zeros(4), names, eligible) == []
    assert positive_top(np.ones(4), names, eligible) == [2, 1, 3]
    assert boundary_tie(np.ones(4), [2, 1, 3], eligible)
    assert consensus(summarize_bootstrap(np.zeros((3, 4)), names, eligible)).empty
    frame = pd.DataFrame([[1., .2], [.2, 1.]], index=['b', 'a'], columns=['b', 'a'])
    assert validate_matrix(frame, ['a', 'b']).shape == (2, 2)
    frame.iloc[0, 1] = .8
    try:
        validate_matrix(frame, ['a', 'b'])
    except ValueError:
        pass
    else:
        raise AssertionError('没有拒绝非对称相关矩阵。')
    rng = np.random.default_rng(5)
    x = rng.normal(size=(24, 3))
    y = 2*x[:, 0] - x[:, 1]
    model = fit_fixed('RF', {'regressor__model__n_estimators': 5, 'regressor__model__max_depth': 2},
                      x[:18], y[:18], 0, 5)
    assert explain_rf(model, x[18:]).shape == (6, 3)
    # 与旧版逐次预测对照，覆盖缺失值、缩尾、多列组、跨组批次及不满批次。
    check_x = x.copy()
    check_x[[2, 20], 0] = np.nan
    check_x[18, 1] = 10000.
    check_modules = np.array([1, 1, 4])
    for clip in (0., .01):
        checked_model = fit_fixed('RF', {'regressor__model__n_estimators': 5},
                                  check_x[:18], y[:18], clip, 5)
        base = mean_absolute_error(y[18:], checked_model.predict(check_x[18:]))
        expected = []
        for module in np.unique(check_modules):
            perm_rng = np.random.default_rng(71 + int(module))
            cols = np.flatnonzero(check_modules == module)
            expected.append([mean_absolute_error(y[18:], checked_model.predict(
                joint_permute(check_x[18:], cols, perm_rng))) - base for _ in range(3)])
        for batch_size in (1, 4, 32):
            actual = group_permutation_changes(checked_model, check_x[18:], y[18:],
                                               check_modules, 3, 71, base, batch_size)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    constant = fit_fixed('EN', {'regressor__model__alpha': .1}, x[:18], np.ones(18), 0, 5)
    np.testing.assert_allclose(constant.predict(x[18:]), 1.)
    np.testing.assert_array_equal(coefficients(constant), np.zeros(3))
    args = argparse.Namespace(inner_folds=2, jobs=1, quick=True, en_clip_quantile=.01)
    fitted, _, search = tune('EN', x[:18], y[:18], np.array(['a']*9+['b']*9), args, 5)
    assert np.isfinite(fitted.predict(x[18:])).all() and len(search) == 2
    prep = fitted.regressor_['prep']
    expected_bounds = np.quantile(x[:18], [.01, .99], axis=0)
    np.testing.assert_allclose(prep.lower, expected_bounds[0])
    np.testing.assert_allclose(prep.upper, expected_bounds[1])
    extreme = np.full((1, 3), 1e12)
    np.testing.assert_allclose(prep.transform(extreme), prep.transform(prep.upper[None, :]))
    np.testing.assert_allclose(prep.upper, expected_bounds[1])
    assert parse_args([]).en_clip_quantile == .01 and parse_args([]).bootstrap == 200
    assert parse_args(['--en-clip-quantile', '0']).en_clip_quantile == 0.
    print('合成数据自检通过；未读取赛题、未生成文件。')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--targets', choices=['all', 'af', 'flavor'], default='all')
    parser.add_argument('--only-target', nargs='+')
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--inner-folds', type=int, default=3)
    parser.add_argument('--permutations', type=int, default=5)
    parser.add_argument('--cluster-cutoff', type=float, default=.7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--jobs', type=int, default=4, help='同时计算的目标数；每个模型单线程，内存紧张用 1 或 2')
    parser.add_argument('--en-clip-quantile', type=float, default=.01,
                        help='EN 训练内缩尾比例，默认 0.01；设 0 恢复旧版不缩尾口径')
    parser.add_argument('--quick', action='store_true', help='仅流程试跑：3 次抽样、3/2 折、30 棵树、1 次置换')
    parser.add_argument('--no-sensitivity', action='store_true', help='跳过风格/缩尾/汇总项对照，并在 manifest 记录')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--plot-only', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args(argv)
    if args.self_check and args.plot_only:
        parser.error('--self-check 与 --plot-only 不能同时使用。')
    if args.quick:
        args.bootstrap, args.folds, args.inner_folds, args.permutations = 3, 3, 2, 1
    if min(args.bootstrap, args.permutations) < 1 or min(args.folds, args.inner_folds) < 2:
        parser.error('抽样/置换至少 1 次；内外层折数至少 2。')
    if not 0 < args.cluster_cutoff < 1 or args.jobs == 0 or args.jobs < -1:
        parser.error('聚类阈值必须在 (0,1)，jobs 为正整数或 -1。')
    if not 0 <= args.en_clip_quantile < .5:
        parser.error('EN 缩尾比例必须在 [0,0.5)。')
    if not 0 <= args.seed < 2**32 - 100000:
        parser.error('seed 超出支持范围。')
    return args


def target_worker(data, j, args, folds, modules, draws, out):
    target = data['targets'][j]
    print(f'[{j+1}/{len(data["targets"])}] 开始 {target}', flush=True)
    started = perf_counter()
    # 同样覆盖 jobs=1，防止 BLAS 自行占满线程。
    with threadpool_limits(limits=1):
        result = analyze_target(data, j, args, folds, modules, draws, out / 'targets' / target)
    elapsed = perf_counter() - started
    print(f'完成 {target}：{elapsed/60:.1f} 分钟', flush=True)
    return result, elapsed


def run(args):
    started = perf_counter()
    import shap  # 在创建结果目录前检查关键依赖。
    if not args.no_plots:
        import matplotlib
    args.data_root = resolve_data_root(args.data_root)
    data = load_inputs(args)
    parent = args.output_root or args.data_root / '运行结果'
    out = parent / ('Q2_v2_' + ('QUICK_' if args.quick else '') + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    inputs = {key: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
              for key, path in data['files'].items()}
    scripts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [Path(__file__), Path(__file__).with_name('q1_without_joint.py')]}
    workers = min(effective_n_jobs(args.jobs), len(data['targets']))
    manifest = {'status': 'running', 'protocol': 'Q2_network_RF_EN_v2', 'parameters': vars(args),
                'parallelism': {'target_workers': workers, 'model_threads': 1, 'permutation_batch_size': 32},
                'EN_preprocessing': f'training-only impute, winsorize q={args.en_clip_quantile}, scale; nested CV and bootstrap',
                'targets': data['targets'], 'n_samples': len(data['meta']), 'main_features': len(data['names']),
                'inputs': inputs, 'script_sha256': scripts, 'python': platform.python_version(),
                'packages': {p: version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'shap', 'joblib']},
                'correlation_max_difference': data['correlation_max_difference'],
                'bootstrap_scope': 'style-stratified; fixed full-data CV hyperparameters; OOB diagnostic only',
                'shap_mode': 'tree_path_dependent; original sensory units',
                'figures_visually_verified': False, 'quick_results_not_for_inference': args.quick}
    write_json(manifest, out / 'manifest.json')
    print(f'结果目录：{out}\n并行目标数：{workers}；每个模型单线程；EN 缩尾：{args.en_clip_quantile:g}', flush=True)
    try:
        data['network_nodes'] = export_network(data, out / 'network')
        csv(data['catalog'], out / 'feature_catalog.csv')
        features = pd.DataFrame(data['x'], columns=data['names'])
        features.insert(0, 'beer', data['meta'].beer.to_numpy())
        csv(features, out / 'input_features.csv')
        folds, modules, draws = save_sampling(data, args, out / 'sampling')
        with parallel_config(backend='loky', inner_max_num_threads=1):
            completed = Parallel(n_jobs=workers, pre_dispatch='n_jobs', batch_size=1, max_nbytes=None)(
                delayed(target_worker)(data, j, args, folds, modules, draws, out)
                for j in range(len(data['targets'])))
        results = [result for result, _ in completed]
        manifest['target_elapsed_seconds'] = dict(zip(data['targets'], [seconds for _, seconds in completed]))
        for index, filename in [(0, 'performance.csv'), (1, 'top3.csv'), (2, 'auxiliary_top3.csv'),
                                (3, 'importance_summary.csv'), (5, 'chemical_evidence_checklist.csv')]:
            csv(pd.concat([r[index] for r in results], ignore_index=True), out / filename)
        csv(pd.DataFrame([r[4] for r in results]), out / 'method_comparison.csv')
        manifest['status'] = 'computed'
        write_json(manifest, out / 'manifest.json')
        if not args.no_plots:
            plot_results(out)
        manifest['status'] = 'complete'
        manifest['elapsed_seconds'] = perf_counter() - started
        write_json(manifest, out / 'manifest.json')
        print(f'完成：{out}\n先检查 performance.csv，再查看 top3.csv 与化学证据核对表。', flush=True)
    except BaseException as error:
        manifest['elapsed_seconds'] = perf_counter() - started
        manifest['status'] = 'computed' if manifest['status'] == 'computed' else 'failed'
        manifest['error'] = f'{type(error).__name__}: {error}'
        write_json(manifest, out / 'manifest.json')
        raise


def main(argv=None):
    args = parse_args(argv)
    if args.self_check:
        self_check()
    elif args.plot_only:
        plot_results(args.plot_only.expanduser().resolve())
    else:
        run(args)


if __name__ == '__main__':
    main()
