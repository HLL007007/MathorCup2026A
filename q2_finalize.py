"""Read existing Q2 v2 results; export one evidence table and four PNG figures.

Python >=3.10. Install: python -m pip install numpy pandas matplotlib networkx
Run from project root: python q2_finalize.py
Check only: python q2_finalize.py --self-check
Paths are relative to this file by default. No model training, network access,
skills, q1 imports, or machine-specific paths are required.
Existing same-named deliverables in --output are regenerated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, lines, font_manager
import networkx as nx

RUN = "Q2_v2_20261028_003954_649276"
BLUE, ORANGE, TEAL, GRAY = "#256A95", "#C87938", "#37867D", "#87939C"
CSV_HEADERS = {
    "target": "感官维度", "rank": "候选排名", "feature": "化合物字段",
    "chemical_display_name": "化合物名称", "top3_frequency": "随机森林入选频率",
    "prediction_evidence": "目标预测评价", "selection_evidence": "入选稳定性分档",
    "chemical_evidence_class": "化学证据类别", "eligible": "符合候选条件",
    "rank_median": "重采样中位排名", "importance_share_mean": "重要性份额均值",
    "importance_share_p025": "重要性份额2.5%分位", "importance_share_p975": "重要性份额97.5%分位",
    "method": "模型", "cv_importance": "折外重要性", "cv_share": "折外重要性份额",
    "spearman_rho": "输入与感官的秩相关", "shap_x_rho": "输入与SHAP的秩相关",
    "EN_full_coefficient": "弹性网全数据系数", "reference_module": "参考相关组编号",
    "chemical_name": "原表化合物名称", "kind": "原字段分类", "unit": "原始单位",
    "representation_status": "数值表示核对结果",
    "associated_group_delta_MAE_mean_RF": "相关组置换后MAE平均增量",
    "associated_group_positive_fold_fraction_RF": "相关组置换MAE增量为正的折占比",
    "OOF_R2": "随机森林折外R²", "OOF_MAE_improvement": "随机森林MAE相对改善率",
    "prediction_supported": "是否满足预测双指标标准", "frequency_gap_3_to_4": "第3与第4名频率差",
    "frequency_tie_at_boundary": "入选边界是否同频", "stability_flag": "频率标记",
    "top3_frequency_EN": "同一化合物的弹性网入选频率", "OOF_R2_EN": "弹性网折外R²",
    "EN_consensus_top3": "是否进入弹性网共识前三", "direction_note": "关联方向说明",
    "chemical_evidence_decision": "化学证据结论", "reference_ids": "参考文献编号",
    "identity_decision": "化合物身份处理结论", "reference_urls": "参考文献链接",
    "target_specific_interpretation": "本感官维度的解释边界", "review_date": "核查日期",
    "OAV_decision": "气味活度值处理结论",
}
REFS = {
    "R1": "https://doi.org/10.1038/s41467-024-46346-0",
    "R2": "https://doi.org/10.3390/separations6040046",
    "R3": "https://doi.org/10.1002/j.2050-0416.2010.tb00428.x",
    "R4": "https://www.sciencedirect.com/science/article/pii/S0308814607007844",
    "R5": "https://doi.org/10.3390/foods12051064",
    "R6": "https://doi.org/10.3390/foods15061113",
    "R7": "https://www.tandfonline.com/doi/abs/10.1094/ASBCJ-2017-3007-01",
    "R8": "https://ec.europa.eu/health/scientific_committees/consumer_safety/opinions/sccnfp_opinions_97_04/sccp_out26_en.htm",
    "R9": "https://echa.europa.eu/substance-information/-/substanceinfo/100.002.801",
    "R10": "https://doi.org/10.1111/j.1467-9868.2005.00503.x",
}

# Evidence describes compounds, not causal validation of every selected target.
CHEMISTRY = {}


def evidence_group(fields, category, finding, references, identity="按原字段报告；未进行标准品身份复验"):
    for field in fields.split("|"):
        if field in CHEMISTRY:
            raise ValueError(f"Duplicate evidence entry: {field}")
        CHEMISTRY[field] = (category, finding, references, identity)


evidence_group("ethanol..v.v.|glycerol", "发酵产物与口感背景",
               "R1讨论Ethanol及混合物添加的感官影响；混合效应不能分离Glycerol的单独贡献，也不能证明每个入选目标的因果关系。", "R1")
evidence_group("lactic_acid.mg.L.|acetic_acid.g.L.", "有机酸背景",
               "Lactic acid、Acetic acid与酸味解释相容；前者在乙酸感中排名较高不等于它是后者的化学来源，仍须考虑共同发酵与风格共变。", "R1")
evidence_group("X4VG|eugenol", "酚类感官研究",
               "R4研究啤酒挥发性酚及4VG；4VG的感官证据不能直接迁移给其他酚类或所有入选感官目标。", "R4")
evidence_group("linalool|citronellol", "单萜醇感官研究",
               "R3研究Linalool背景下Geraniol与Citronellol的协同，R5研究酵母引起的酒花香气变化；单变量排名不能推断单独添加效果。", "R3|R5")
evidence_group("isopentyl_acetate|ethyl_hexanoate|ethyl_acetate", "酯类香气与混合物研究",
               "R1、R5支持发酵酯参与复杂香气；Isoamyl acetate与isoaa感官目标相容，对总体喜好或其他目标仍仅作关联解释。", "R1|R5")
evidence_group("ethyl_octanoate", "个体酯类感官研究",
               "R7研究Ethyl octanoate的感官贡献与混合效应，支持其参与啤酒风味，但不能认定其为直接苦味来源。", "R7")
evidence_group("ethyl.phenylacetate", "检出与混合添加证据",
               "R2报告Ethyl phenylacetate，R1将其纳入混合添加；不能误作Phenethyl acetate，也不能据酸度排名宣称它直接产酸。", "R1|R2")
evidence_group("acetaldehyde", "发酵挥发物背景",
               "R1包含Acetaldehyde的风味预测；其酯香或麦芽感排名仅支持预测关联，Acetaldehyde本身不是酯类。", "R1")
evidence_group("methylpyrazine|X2.acetylpyrrole|X5.methylfurfural|X2.acetylfuran", "特种麦芽来源证据",
               "R6报告特种麦芽中的吡嗪、乙酰吡咯和呋喃类与焙烤、焦糖、面包香背景；支持来源合理性，不等于本数据已分离个体作用。", "R6")
evidence_group("DMS", "硫化物感官背景",
               "R5报告Dimethyl sulfide及感官、挥发物关联；对热带香等非典型入选目标没有独立因果验证。", "R5")
evidence_group("CO2.PSI.", "气体相关分析指标",
               "PSI表示压力测量，不等于溶解Carbon dioxide浓度；保留原气体候选分类，不直接将压力换算为浓度或气味活度值。", "R1",
               "Carbon dioxide身份明确；压力换算浓度所需条件不足")
evidence_group("SO2.mg.L.", "硫化物分析信号",
               "本轮检索不足以支持该候选对当前目标的独立感官作用，仅保留统计关联。", "")
evidence_group("ethyl_decanoate|ethyl.tetradecanoate|ethyl_isovalerate|ethyl.lactate|butyl.acetate|isopentyl.butanoate|isopentyl.hexanoate|octyl_acetate|pentyl.octanoate|linalool.isovalerate|methyl.geranate|neryl.propanoate|terpinen.4.ol.acetate",
               "酯类类别支持",
               "R2、R5提供酯类检出或香气背景；本轮不足以确认本字段对当前目标的独立作用，不套用其他酯的气味阈值。", "R2|R5")
evidence_group("isoamyl_alcohol|isobutanol|n.hexanol|n.heptanol|n.nonanol|X1.octen.3.ol|X2.ethyl.1.hexanol|X2.undecanol",
               "醇类类别支持",
               "啤酒挥发物研究提供醇类背景，但未分离本字段对当前目标的独立作用；尤其不能把N-hexanol误作Hexanal。", "R2|R5")
evidence_group("E.caryophyllene|X9.epi..E..caryophyllene|X1.10.di.epi.cubenol|alloaromadendrene.epoxide|alpha.calacorene|alpha.eudesmol|alpha.muurolol|gamma.eudesmol|tau.cadinol",
               "萜类检出或类别支持",
               "R2报告包括Epicubenol及Alloaromadendrene oxide在内的部分倍半萜；检出不证明感官活性，异构体或环氧化物名称也不可无条件互换，仅作相关分析信号。", "R2")
evidence_group("beta.phellandrene|camphor|cis.dihydro.beta.terpineol|eucalyptol|isoborneol|nerol.oxide|terpinen.4.ol|terpinolene|X.E..alpha.ionone|X.E..beta.ionone",
               "萜类及相关挥发物背景",
               "R2、R5提供类别与共变背景；不足以确认本字段对当前目标的独立作用，不将Linalool或Citronellol的证据直接迁移至本字段。", "R2|R5")
evidence_group("X.E..2.nonenal|X.E.E..2.4.decadienal", "醛类分析信号",
               "保留原字段的构型信息；现有直接研究不足以证明本字段对当前目标的独立作用，不仅按所属醛类指定气味。", "")
evidence_group("X2.ethylhexylsalicylate", "身份冲突：仅报告分析信号",
               "字段指向2-ethylhexyl salicylate（CAS 118-60-5），名称表却写2-ethylhexyl acrylate（CAS 103-11-7）；R8、R9确认两者不同，但不能确定实际测量列身份，不赋予确定机制。", "R8|R9",
               "身份冲突；保留原字段、数值及排名，不计为身份已确认的关键化合物")
evidence_group("delta.terpinyl.acetate", "异构体标注不足",
               "字段含delta前缀，名称表为Terpiryl acetate，不能唯一确定异构体；保留统计排名，不套用其他Terpinyl acetate异构体的气味证据。", "",
               "现有注释不足以确认异构体，仅作分析信号")
evidence_group("ethyl..E..linalool", "衍生物标注不足",
               "原名称为Ethyl linalool，不等于普通Linalool，R3不能直接验证该字段；单独保留其统计关联。", "R3",
               "保留Ethyl linalool字段，不与Linalool合并")


def display_target(target):
    # Match the original English target identifiers used in Question 1.
    return target


def display_name(value):
    return str(value).replace("Î±", "alpha").replace("Î²", "beta").replace("Î³", "gamma")


def load_results(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["parameters"]["quick"]:
        raise ValueError("A complete, non-quick result is required.")
    data = {"root": root, "manifest": manifest}
    for key, filename in [("perf", "performance.csv"), ("top", "top3.csv"),
                          ("summary", "importance_summary.csv"), ("catalog", "feature_catalog.csv"),
                          ("comparison", "method_comparison.csv"), ("inputs", "input_features.csv")]:
        data[key] = pd.read_csv(root / filename)
    for key, filename in [("nodes", "nodes.csv"), ("edges", "edges_0.6.csv"),
                          ("thresholds", "threshold_sensitivity.csv")]:
        data[key] = pd.read_csv(root / "network" / filename)
    targets = manifest["targets"]
    assert len(targets) == len(set(targets)) == 50
    assert manifest["parameters"]["bootstrap"] == 200
    rf = data["top"].query("method == 'RF'")
    assert len(rf) == 150 and rf.groupby("target")["rank"].apply(list).apply(lambda x: sorted(x) == [1, 2, 3]).all()
    assert set(rf.feature) == set(CHEMISTRY), "Evidence coverage must match all unique RF candidates."
    assert data["perf"].groupby(["target", "method"]).size().eq(1).all()
    assert set(data["edges"].source) | set(data["edges"].target) <= set(data["nodes"].field_name_R)
    return data


def build_table(data):
    table = data["top"].query("method == 'RF'").copy()
    en = data["summary"].query("method == 'EN'")[["target", "feature", "top3_frequency", "OOF_R2"]]
    table = table.merge(en, on=["target", "feature"], suffixes=("", "_EN"), validate="one_to_one")
    en_top = set(zip(data["top"].query("method == 'EN'").target,
                     data["top"].query("method == 'EN'").feature))
    table["EN_consensus_top3"] = [(t, f) in en_top for t, f in zip(table.target, table.feature)]
    table["chemical_display_name"] = table.chemical_name.map(display_name)
    conflict = table.feature.eq("X2.ethylhexylsalicylate")
    table.loc[conflict, "chemical_display_name"] = "X2.ethylhexylsalicylate（身份冲突）"
    table["prediction_evidence"] = np.where(table.prediction_supported, "满足预测双指标标准", "未同时满足预测双指标标准")
    table["selection_evidence"] = np.select(
        [table.top3_frequency.ge(.8), table.top3_frequency.ge(.6)],
        ["频率较高（≥80%）", "频率中等（60%–80%）"], default="频率较低（<60%）")
    table["direction_note"] = table.shap_x_rho.map(
        lambda v: "SHAP与输入的总体秩相关为正；不表示全区间单调" if v > 0 else
        "SHAP与输入的总体秩相关为负；不表示全区间单调" if v < 0 else "无总体秩方向")
    for index, column in enumerate(["chemical_evidence_class", "chemical_evidence_decision", "reference_ids", "identity_decision"]):
        table[column] = table.feature.map(lambda f: CHEMISTRY[f][index])
    table["reference_urls"] = table.reference_ids.map(lambda s: " | ".join(REFS[k] for k in s.split("|") if k))
    direct_pairs = {
        ("ethanol..v.v.", "alcohol"), ("lactic_acid.mg.L.", "acidity"),
        ("lactic_acid.mg.L.", "lactic"), ("acetic_acid.g.L.", "acidity"),
        ("X4VG", "X4vg"), ("isopentyl_acetate", "A_esters_isoaa"),
        ("isopentyl_acetate", "F_esters_isoaa"),
        ("ethyl_acetate", "A_esters_ethac"), ("ethyl_acetate", "F_esters_ethac"),
        ("linalool", "A_hops_citrus"), ("citronellol", "A_hops_citrus"),
        ("citronellol", "F_hops_citrus"),
    }
    table["target_specific_interpretation"] = [
        "与既有化学/感官关系相容；本数据尚无单物质干预验证" if (f, t) in direct_pairs
        else "仅保留候选关联；无充分直接证据证明其独立决定本感官维度"
        for f, t in zip(table.feature, table.target)]
    table["review_date"] = "2026-10-28"
    table["OAV_decision"] = "不计算；需逐物质确认原浓度、单位、异构体与啤酒介质阈值"
    assert table.representation_status.notna().all()
    leading = ["target", "rank", "feature", "chemical_display_name", "top3_frequency",
               "prediction_evidence", "selection_evidence", "chemical_evidence_class"]
    return table[leading + [c for c in table.columns if c not in leading]]


def configure(font_path=None):
    if font_path is not None:
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        available = {f.name for f in font_manager.fontManager.ttflist}
        candidates = ["Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC",
                      "Source Han Sans SC", "SimHei", "Heiti SC", "WenQuanYi Micro Hei", "SimSun"]
        family = next((name for name in candidates if name in available), None)
        if family is None:
            raise RuntimeError("未找到中文字体。请安装思源黑体/Noto Sans CJK SC，或使用 --font 字体文件路径。")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 13,
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 11,
        "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "legend.frameon": False, "legend.fontsize": 11,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def export_png(fig, output, filename, dpi, axes=None):
    fig.canvas.draw()
    # In-memory alignment check; no QA sidecar files are created.
    if axes is not None:
        boxes = [ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) for ax in axes]
        widths = np.array([b.width for b in boxes]) * 72
        heights = np.array([b.height for b in boxes]) * 72
        assert np.ptp(widths) <= 1.5 and np.ptp(heights) <= 1.5, "Unequal panel geometry"
        for i, first in enumerate(boxes):
            for second in boxes[i+1:]:
                assert not first.overlaps(second), "Panel plot areas overlap"
    renderer = fig.canvas.get_renderer()
    outer = fig.bbox
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text() and text.get_in_layout():
            assert text.get_fontsize() >= 10, "Figure text must remain readable."
            bounds = text.get_window_extent(renderer)
            if bounds.width and bounds.height:
                assert bounds.x0 >= outer.x0 - 2 and bounds.x1 <= outer.x1 + 2, f"Text outside figure: {text.get_text()}"
                assert bounds.y0 >= outer.y0 - 2 and bounds.y1 <= outer.y1 + 2, f"Text outside figure: {text.get_text()}"
    fig.savefig(output / filename, dpi=dpi, facecolor="white")
    plt.close(fig)


def performance_figure(data, output, dpi):
    perf = data["perf"].pivot(index="target", columns="method", values="R2")
    order = perf.RF.sort_values(ascending=False).index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 9))
    fig.subplots_adjust(left=.17, right=.97, bottom=.09, top=.88, wspace=.68)
    for ax, targets in zip(axes, [order[:25], order[25:]]):
        rf, en = perf.loc[targets, "RF"].to_numpy(), perf.loc[targets, "EN"].to_numpy()
        y = np.arange(len(targets))
        for row in y[::2]:
            ax.axhspan(row-.5, row+.5, color="#F4F6F7", lw=0)
        ax.hlines(y, np.minimum(rf, en), np.maximum(rf, en), color="#ADB8BF", lw=1.2, zorder=2)
        ax.scatter(rf, y, s=32, c=BLUE, label="随机森林", zorder=3)
        ax.scatter(en, y, s=30, c=ORANGE, marker="D", label="弹性网", zorder=3)
        ax.axvline(0, color="#555D63", lw=.8, ls="--")
        ax.set(yticks=y, yticklabels=[display_target(t) for t in targets],
               ylim=(len(y)-.4, -.6), xlim=(-.085, .78), xlabel="折外预测 R²")
        ax.tick_params(axis="y", length=0, labelsize=11)
        ax.grid(axis="x", color="#E2E7EB", lw=.5)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
    fig.suptitle("感官指标的预测表现", y=.97, fontsize=19, weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=2)
    export_png(fig, output, "01_predictive_performance.png", dpi, list(axes))


def top3_figures(table, order, output, dpi):
    # The figures and exported CSV use the same candidate rows, not rank-only placeholders.
    indexed = table.set_index(["target", "rank"])
    assert indexed.index.is_unique
    assert len(order) == len(set(order)) == 50
    ambiguous = {"X2.ethylhexylsalicylate", "delta.terpinyl.acetate", "ethyl..E..linalool"}
    cmap = colors.LinearSegmentedColormap.from_list("inclusion", ["#F3F7FA", "#8CB8CB", "#236886", "#123D57"])
    shown = set()
    for offset, filename in [(0, "02_top3_compounds_01_25.png"), (25, "03_top3_compounds_26_50.png")]:
        targets = order[offset:offset+25]
        values = np.array([[indexed.loc[(target, rank), "top3_frequency"] for rank in (1, 2, 3)]
                           for target in targets])
        fig, ax = plt.subplots(figsize=(11, 12.5))
        fig.subplots_adjust(left=.23, right=.98, bottom=.14, top=.88)
        img = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        cell_text = []
        for i, target in enumerate(targets):
            for j, rank in enumerate((1, 2, 3)):
                row = indexed.loc[(target, rank)]
                name = row.feature if row.feature == "X2.ethylhexylsalicylate" else row.chemical_display_name
                marker = "†" if row.feature in ambiguous else ""
                label = textwrap.fill(f"{name}{marker} ({100*row.top3_frequency:g}%)", width=31,
                                      break_long_words=False)
                assert label.count("\n") <= 1, "Increase cell height rather than shrink text."
                cell_text.append(ax.text(j, i, label, ha="center", va="center", fontsize=11.5,
                                         linespacing=1.0, color="white" if row.top3_frequency >= .65 else "#18394C"))
                shown.add((target, rank))
        ax.set(xticks=range(3), xticklabels=["第1候选", "第2候选", "第3候选"],
               yticks=range(25), yticklabels=[display_target(t) for t in targets])
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 25, 1), minor=True)
        ax.xaxis.tick_top()
        ax.tick_params(length=0, labelsize=11, pad=8)
        ax.grid(which="minor", color="white", linewidth=1.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        for tick, target in zip(ax.get_yticklabels(), targets):
            if not bool(indexed.loc[(target, 1), "prediction_supported"]):
                tick.set_color("#7D858D")
        for spine in ax.spines.values():
            spine.set_visible(False)
        cax = fig.add_axes([.31, .075, .61, .015])
        cb = fig.colorbar(img, cax=cax, orientation="horizontal", ticks=[0, .2, .4, .6, .8, 1])
        cb.set_ticklabels(["0", "20", "40", "60", "80", "100"])
        cb.set_label("进入前三的重采样频率（%）", fontsize=11)
        cb.outline.set_visible(False)
        fig.suptitle(f"各感官维度的 Top-3 关键因素候选（{offset+1}–{offset+25}）",
                     y=.975, fontsize=19, weight="bold")
        fig.text(.5, .935, "随机森林主结果：化合物名称与入选频率", ha="center", fontsize=12, color="#53616C")
        fig.text(.5, .012, "†：身份或名称注释存在问题    灰色维度：预测未通过双指标评价",
                 ha="center", fontsize=10, color="#53616C")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        area = ax.get_window_extent(renderer)
        for label in cell_text:
            bounds = label.get_window_extent(renderer)
            assert bounds.width <= area.width/3*.96, f"Name too wide: {label.get_text()}"
            assert bounds.height <= area.height/25*.96, f"Name too tall: {label.get_text()}"
        export_png(fig, output, filename, dpi)
    assert shown == set(indexed.index), "All 150 CSV candidate rows must be plotted exactly once."


def network_figure(data, output, dpi):
    nodes, edges = data["nodes"], data["edges"]
    graph = nx.Graph()
    graph.add_nodes_from(nodes.field_name_R)
    graph.add_weighted_edges_from(edges[["source", "target", "weight"]].itertuples(index=False, name=None))
    seeds = ["ethanol..v.v.", "ethyl_octanoate", "lactic_acid.mg.L.", "X4VG"]
    selected = set(data["top"].query("method == 'RF'").feature)
    main = set(data["catalog"].query("in_main_model").field_name_R)
    local = set(seeds)
    for seed in seeds:
        neighbors = sorted((n for n in graph.neighbors(seed) if n in main),
                           key=lambda n: (-graph[seed][n]["weight"], n))[:3]
        local.update(neighbors)
    fig, axs = plt.subplots(1, 2, figsize=(10, 6.3))
    fig.subplots_adjust(left=.035, right=.97, bottom=.16, top=.82, wspace=.14)
    isolated = sorted(nx.isolates(graph))
    connected = graph.subgraph(set(graph) - set(isolated))
    components = sorted(nx.connected_components(connected), key=lambda c: (-len(c), sorted(c)))
    layouts = []
    for component in components:
        # Stable node order keeps layout reproducible across Python processes.
        part = nx.Graph()
        part.add_nodes_from(sorted(component))
        part.add_edges_from(graph.subgraph(component).edges(data=True))
        layouts.append(part)
    pos = nx.spring_layout(layouts[0], seed=42,
                           iterations=300, weight="weight", scale=.67)
    for part, angle in zip(layouts[1:], np.linspace(0, 2*np.pi, len(layouts)-1, endpoint=False)):
        centre = np.array([np.cos(angle), np.sin(angle)]) * .84
        pos.update(nx.spring_layout(part, seed=42, scale=.055, center=centre))
    for node, angle in zip(isolated, np.linspace(0, 2*np.pi, len(isolated), endpoint=False)):
        pos[node] = np.array([np.cos(angle), np.sin(angle)]) * 1.04
    all_names = list(graph)
    ax = axs[0]
    for sign, color, style in [("positive", "#B7C8D2", "-"), ("negative", "#D1A791", "--")]:
        subset = list(edges.loc[edges.sign.eq(sign), ["source", "target"]].itertuples(index=False, name=None))
        nx.draw_networkx_edges(graph, pos, edgelist=subset, ax=ax, edge_color=color, width=.45, style=style, alpha=.65)
    nx.draw_networkx_nodes(graph, pos, nodelist=all_names, ax=ax,
                          node_size=[19 if n in selected else 7 for n in all_names],
                          node_color=[BLUE if n in selected else "#C8CED3" for n in all_names], linewidths=0)
    ax.set_title("a  整体相关网络", loc="left", fontsize=14, weight="bold", pad=12)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    ax.set_axis_off()
    ax = axs[1]
    # Positions only aid readability. They do not encode chemical effect sizes.
    coords = {
        "ethanol..v.v.": (-.1, .30), "ethyl_octanoate": (.63, .30),
        "ethyl_hexanoate": (.44, .92), "glycerol": (-.72, .94),
        "ethyl_acetate": (-.80, -.12), "ethyl_decanoate": (.88, -.12),
        "lactic_acid.mg.L.": (-.52, -.92), "X4VG": (.66, -.92),
    }
    assert local == set(coords), "The specified local example changed; update its layout explicitly."
    subedges = edges[edges.source.isin(local) & edges.target.isin(local)]
    label_offsets = {
        "ethanol..v.v.": (-5, -14), "ethyl_octanoate": (12, 8),
        "ethyl_hexanoate": (0, 10), "glycerol": (0, 10),
        "ethyl_acetate": (0, -14), "ethyl_decanoate": (0, -14),
        "lactic_acid.mg.L.": (0, -14), "X4VG": (0, -14),
    }
    labels = {"ethanol..v.v.": "Ethanol", "ethyl_octanoate": "Ethyl\noctanoate",
              "ethyl_hexanoate": "Ethyl hexanoate", "glycerol": "Glycerol",
              "ethyl_acetate": "Ethyl acetate", "ethyl_decanoate": "Ethyl decanoate",
              "lactic_acid.mg.L.": "Lactic acid", "X4VG": "4VG"}
    for row in subedges.itertuples():
        x1, y1 = coords[row.source]; x2, y2 = coords[row.target]
        ax.plot([x1, x2], [y1, y2], color=BLUE if row.rho > 0 else ORANGE,
                lw=.6+2*abs(row.rho), alpha=.5, ls="-" if row.rho > 0 else "--", zorder=1)
    for node, xy in coords.items():
        ax.scatter(*xy, s=115 if node in seeds else 52, c=TEAL if node in seeds else "#D0DFE6",
                   edgecolors="white", linewidth=.8, zorder=3)
        ax.annotate(labels[node], xy, xytext=label_offsets[node], textcoords="offset points",
                    ha="left" if node == "ethyl_octanoate" else "center",
                    va="bottom" if label_offsets[node][1] > 0 else "top", fontsize=12)
    ax.set_title("b  代表性候选及其邻域", loc="left", fontsize=14, weight="bold", pad=12)
    ax.set_xlim(-1.22, 1.28); ax.set_ylim(-1.50, 1.35); ax.set_axis_off()
    fig.suptitle("化学指标的共变关系", y=.97, fontsize=19, weight="bold")
    fig.text(.5, .90, "相关阈值 |ρ| ≥ 0.6", ha="center", fontsize=12, color="#53616C")
    handles = [
        lines.Line2D([], [], marker="o", color="none", markerfacecolor=BLUE, markeredgewidth=0, markersize=7, label="入选候选"),
        lines.Line2D([], [], marker="o", color="none", markerfacecolor=TEAL, markeredgewidth=0, markersize=8, label="重点展示"),
        lines.Line2D([], [], color=BLUE, lw=1.5, label="正相关"),
        lines.Line2D([], [], color=ORANGE, lw=1.5, ls="--", label="负相关"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .035), ncol=4, fontsize=11)
    export_png(fig, output, "04_chemical_network.png", dpi, list(axs))
    return sorted(local), len(subedges)


def self_check(data, table):
    assert len(table) == 150 and table.feature.nunique() == 66
    assert table[["target", "rank"]].duplicated().sum() == 0
    assert table.reference_ids.fillna("").str.split("|").map(lambda a: all(not k or k in REFS for k in a)).all()
    counts = table.groupby("target").size()
    assert counts.eq(3).all() and len(counts) == 50
    assert set(table.target) == set(data["manifest"]["targets"])
    assert table.groupby("target").feature.nunique().eq(3).all()
    assert table.kind.eq("compound").all(), "Top-3 must exclude auxiliary and derived features."
    assert set(table.columns) == set(CSV_HEADERS), "Every exported column needs a Chinese header."
    assert table.top3_frequency.between(0, 1).all()
    assert not table.chemical_evidence_decision.str.contains("待核查|待人工").any()
    for target in data["manifest"]["targets"]:
        chosen = pd.read_csv(data["root"] / "targets" / target / "bootstrap_top3.csv")
        observed = chosen.query("method == 'RF'").feature.value_counts() / 200
        actual = table[table.target.eq(target)]
        np.testing.assert_allclose(actual.top3_frequency, actual.feature.map(observed).fillna(0), atol=1e-12)
    print("Checks passed: 50 targets; 150 RF candidates; 66 evidence entries; bootstrap frequencies match.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "Q2")
    parser.add_argument("--font", type=Path, help="Optional font file (TTF/OTF/TTC)")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.dpi < 300:
        parser.error("Use --dpi >= 300 for these print figures.")
    if args.results is None:
        defaults = [Path(__file__).resolve().parent / "运行结果" / RUN,
                    Path(__file__).resolve().parent / "results"]
        args.results = next((p for p in defaults if (p / "manifest.json").exists()), None)
        if args.results is None:
            parser.error("Result directory not found. Pass --results PATH, or place the v2 result in ./results.")
    root, output = args.results.expanduser().resolve(), args.output.expanduser().resolve()
    if output == root or root in output.parents:
        parser.error("Place deliverables outside the original result directory.")
    data = load_results(root)
    table = build_table(data)
    self_check(data, table)
    if args.self_check:
        return
    output.mkdir(parents=True, exist_ok=True)
    configure(args.font)
    performance_figure(data, output, args.dpi)
    top3_figures(table, data["manifest"]["targets"], output, args.dpi)
    local, n_edges = network_figure(data, output, args.dpi)
    localized = table.copy()
    for column in ["eligible", "prediction_supported", "frequency_tie_at_boundary", "EN_consensus_top3"]:
        localized[column] = localized[column].map({True: "是", False: "否"})
    localized["method"] = localized.method.map({"RF": "随机森林"})
    localized["kind"] = "化合物字段"
    localized["stability_flag"] = localized.stability_flag.map({
        "unstable": "频率低于60%", "frequency>=0.6 (descriptive)": "频率≥60%（描述性）"})
    localized.rename(columns=CSV_HEADERS).to_csv(output / "Q2_final_top3_evidence.csv", index=False, encoding="utf-8-sig")
    print(f"Saved 4 PNG figures and 1 evidence CSV to {output}")
    print(f"Local network: {len(local)} nodes, {n_edges} edges. No models trained.")
    print("Source manifest SHA256:", hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest())


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
