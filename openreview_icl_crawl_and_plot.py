#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenReview（ICLR/ICML）ICL 论文统计：title+abstract 检索 + 细分 taxonomy 分类 + 作图

本版修复：
1) 中文显示问题：
   - 更稳健地设置 Matplotlib 字体：font.sans-serif 列表 + font.family='sans-serif'
   - 支持 --font 显式指定字体（mac 常用 "PingFang SC"，Win 常用 "Microsoft YaHei"，Linux 常用 "Noto Sans CJK SC"）
   - 类别标签里带 emoji/图标时，缺少 emoji 字体会出现乱码/方块，因此绘图时默认去掉前缀 emoji。

2) 折线图“只有点没有线”：
   - 常见原因是某些类别只在某一年出现 => 每条线只有一个点，看起来像“没连线”
   - 解决：对 years 做 reindex（缺失年份补 0），并强制 linestyle='-'

3) 仅用已保存数据重画图（不重复抓取）：
   - 使用 --plot_only 直接读取 outdir/icl_papers_filtered.csv 生成图片
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import requests
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

VERSION = "v3.0"


def _display_label(s: str) -> str:
    """Strip leading emoji/symbols so Chinese text renders even if emoji fonts are missing."""
    return re.sub(r"^[^\u4e00-\u9fffA-Za-z0-9]+\s*", "", str(s)).strip()


# ----------------------------
# 1) ICL 过滤：Title + Abstract
# ----------------------------
ICL_TERMS = [
    r"\bin[- ]context\b",
    r"\bin[- ]context learning\b",
    r"\bICL\b",
    r"\b(in[- ]context) (reason|learn|adapt|generaliz)\w*",
    r"\bmany[- ]shot\b",
    r"\bfew[- ]shot\b",
]
ICL_REGEX = re.compile("|".join(ICL_TERMS), flags=re.IGNORECASE)


# ----------------------------
# 2) Taxonomy - 基于ICL研究核心问题的分类体系
# ----------------------------
@dataclass(frozen=True)
class Category:
    key: str
    label: str
    patterns: Tuple[str, ...]


# 1. Prompt工程与优化（合并原B1、B2）
PROMPT_ENGINEERING = Category(
    "prompt_eng",
    "📚 Prompt工程与优化",
    (
        r"\bprompt (engineering|design|optimization|learning|tuning)\b",
        r"\bexample selection\b|\bdemonstration selection\b|\bexemplar selection\b",
        r"\bselect(ing)? (examples|demonstrations|exemplars)\b",
        r"\bprompt ordering\b|\border(ing)? demonstrations\b|\bpermutation\b",
        r"\bcompose(d)? demonstrations\b|\bstructure(d)? prompt\b",
        r"\bretrieve demonstrations\b|\bfew[- ]shot (example|prompt)\b",
        r"\btemplate\b.*\b(design|optimization)\b|\binstruction (following|tuning)\b",
    ),
)

# 2. 推理与思维链（原B3）
REASONING_COT = Category(
    "reasoning_cot",
    "🧠 推理与思维链",
    (
        r"\bchain[- ]of[- ]thought\b|\bCoT\b|\bscratchpad\b",
        r"\bself[- ]consistency\b|\btree[- ]of[- ]thought\b|\bgraph[- ]of[- ]thought\b",
        r"\bmultistep\b|\bmulti[- ]step\b|\bmultiple step\b",
        r"\breason(er|ing)\b.*\b(trace|path|step|chain|process)\b",
        r"\bdeliberat(e|ion)\b|\bthought (generation|process)\b",
        r"\bmany[- ]shot\b|\bmany[- ]step\b",
        r"\bintermediate (reasoning|step|output)\b|\bstep[- ]by[- ]step\b",
        r"\bcomplex reasoning\b|\blogical reasoning\b|\bmathematical reasoning\b",
    ),
)

# 3. 机理理解与可解释性（原A + 部分F）
MECHANISM_THEORY = Category(
    "mechanism_theory",
    "🔬 机理理解与可解释性",
    (
        r"\btheor(y|etical)\b.*\b(ICL|in[- ]context)\b",
        r"\bmechanis\w*\b.*\b(ICL|in[- ]context)\b",
        r"\binduction head(s)?\b|\bcircuit(s)?\b.*\b(analysis|discover)\b",
        r"\binterpretab\w*|\bexplainab\w*|\bunderstanding\b.*\b(ICL|in[- ]context)\b",
        r"\bassociative memory\b|\bhopfield\b|\bmeta[- ]learn\w*",
        r"\bimplicit (learning|gradient)\b|\bin[- ]weights\b",
        r"\bprovab\w*|\bconvergence\b|\blearning dynamics\b",
        r"\battribution\b|\bprobe\b|\bdiagnostic\b.*\bICL\b",
    ),
)

# 4. 模型训练与架构（精简后的D）
MODEL_TRAINING = Category(
    "model_training",
    "🏗️ 模型训练与架构",
    (
        r"\bpretrain\w*\b|\bfine[- ]tun\w*\b|\btraining\b",
        r"\barchitecture\b|\bmodel design\b|\bneural architecture\b",
        r"\bstate space model\b|\bxLSTM\b|\bmamba\b|\bretention\b",
        r"\bsequence model(ing)?\b|\bmixture of experts\b|\bMoE\b",
        r"\btransformer (variant|architecture|model)\b",
        r"\battention (mechanism|variant|pattern|head)\b",
        r"\bposition(al)? (encoding|embedding|interpolation)\b",
        r"\blayer (normalization|norm)\b|\bactivation function\b",
        r"\bmodel (scaling|size|capacity|parameter)\b",
        r"\bbackbone\b|\bfoundation model\b|\blarge language model\b.*\barchitecture\b",
    ),
)

# 5. 效率优化（合并C1、C2、C3）
EFFICIENCY = Category(
    "efficiency",
    "⚡ 效率优化",
    (
        r"\bcontext compression\b|\bprompt compression\b",
        r"\bcompress(ion|ing)?\b.*\b(ICL|context|prompt)\b",
        r"\bdistill(at|ation)\w*\b.*\b(ICL|context|in[- ]context)\b",
        r"\b(in[- ]context )?autoencoder\b|\bcontext distillation\b",
        r"\bkv cache\b|\bkey[- ]value cache\b|\bcache\b.*\boptimization\b",
        r"\bprefill\b|\bthroughput\b|\blatency\b.*\b(optimization|reduction)\b",
        r"\befficient (attention|inference)\b|\blinear attention\b|\bflash[- ]?attention\b",
        r"\blength generaliz\w*\b|\blength extrapolat\w*\b",
        r"\btrain short.*infer long\b|\blong[- ]short\b",
        r"\bcontext length\b.*\b(generaliz\w*|extrapolat\w*|extension)\b",
        r"\bpositional extrapolat\w*\b|\bRoPE\b.*\bscaling\b",
    ),
)

# 6. 评测基准与数据集（精简后的F）
EVALUATION = Category(
    "evaluation",
    "📊 评测基准与数据集",
    (
        r"\bbenchmark\b.*\b(ICL|in[- ]context|few[- ]shot)\b",
        r"\b(evaluation|testbed|dataset)\b.*\b(ICL|in[- ]context|few[- ]shot)\b",
        r"\bnew (benchmark|dataset|task)\b",
        r"\bmeasure\b|\bmetric\b.*\b(ICL|in[- ]context)\b",
        r"\bablation (study|experiment)\b|\bempirical (study|analysis)\b",
        r"\bsurvey\b|\bliterature review\b",
    ),
)

# 7. 应用：Agent与工具使用（原E）
APPLICATION_AGENT = Category(
    "application_agent",
    "🤖 应用：Agent与工具使用",
    (
        r"\bagent(s)?\b.*\b(ICL|in[- ]context|few[- ]shot)\b",
        r"\bplanning\b.*\b(agent|ICL|in[- ]context)\b",
        r"\btool (use|usage|calling|learning)\b",
        r"\bfunction calling\b|\bAPI (call|usage)\b",
        r"\baction (sequence|selection)\b|\btrajectory\b",
        r"\breasoning and acting\b|\bReAct\b",
        r"\baudited reasoning\b|\bemergent abilit\w*\b",
    ),
)

# 8. 可靠性与安全（合并B6和G）
RELIABILITY_SAFETY = Category(
    "reliability_safety",
    "🛡️ 可靠性与安全",
    (
        r"\bcalibrat\w*|\buncertaint\w*|\bconfidence (estimation|calibration)\b",
        r"\breliabilit\w*\b|\brobust\w*\b.*\b(ICL|in[- ]context)\b",
        r"\bselective prediction\b|\babstain\b|\breject option\b",
        r"\bunlearning\b|\bforget(ting)?\b|\bmachine unlearning\b",
        r"\bprivacy\b.*\b(ICL|in[- ]context)\b|\bdata leakage\b",
        r"\battack\b.*\b(ICL|prompt)\b|\bbackdoor\b|\badversarial\b",
        r"\bjailbreak\b|\bprompt injection\b",
        r"\bwatermark\b|\bsafety\b|\brefusal\b",
        r"\bhallucination\b|\bfaithful\w*\b",
    ),
)

# 9. 特定技术方法（原B4、B5等）
SPECIFIC_METHODS = Category(
    "specific_methods",
    "🎯 特定技术方法",
    (
        r"\bnearest neighbor\b|\b(k[- ]?nn|kNN)\b.*\b(ICL|in[- ]context)\b",
        r"\bnonparametric\b.*\b(ICL|learning)\b|\bprototype(s)?\b",
        r"\bcalibration[- ]free\b|\bembedding[- ]based inference\b",
        r"\bvector database\b|\bretrieval[- ]augmented\b",
        r"\bmistake(s)?\b.*\b(learning|correction)\b",
        r"\berror(s)?\b.*\b(analysis|learning|feedback)\b",
        r"\bcounterexample(s)?\b|\bfrom mistakes\b",
        r"\bprinciple learning\b|\brule induction\b",
        r"\bself[- ]correction\b|\bself[- ]refinement\b|\bself[- ]improvement\b",
        r"\bcontrastive\b.*\b(ICL|learning)\b|\bsymbol tuning\b",
    ),
)

CATEGORY_PRIORITY: List[Category] = [
    EVALUATION,           # 优先识别评测类（避免被其他类误判）
    APPLICATION_AGENT,    # Agent应用（特征明显）
    REASONING_COT,        # 推理与CoT（特征明显）
    PROMPT_ENGINEERING,   # Prompt工程
    SPECIFIC_METHODS,     # 特定方法（避免被大类吸收）
    EFFICIENCY,           # 效率优化
    RELIABILITY_SAFETY,   # 可靠性与安全
    MECHANISM_THEORY,     # 机理理论
    MODEL_TRAINING,       # 模型训练（最后，避免过度匹配）
]
DEFAULT_LABEL = "🧺 其他/未归类"


def classify(text: str) -> str:
    for cat in CATEGORY_PRIORITY:
        for p in cat.patterns:
            if re.search(p, text, flags=re.IGNORECASE):
                return cat.label
    return DEFAULT_LABEL


def safe_json(resp: requests.Response) -> Union[dict, list]:
    try:
        return resp.json()
    except Exception:
        snippet = resp.text[:400].replace("\n", " ")
        raise RuntimeError(f"Non-JSON response (status={resp.status_code}). Head: {snippet}")


def extract_notes(payload: Union[dict, list]) -> List[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("notes", "data", "results"):
            if k in payload and isinstance(payload[k], list):
                return payload[k]
    return []


def http_get(baseurl: str, path: str, params: Dict, timeout: int) -> Union[dict, list]:
    url = f"{baseurl.rstrip('/')}{path}"
    headers = {"User-Agent": "ICL-survey-bot/3.0 (requests)", "Accept": "application/json"}
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return safe_json(r)


def fetch_notes_paginated(baseurl: str, invitation: str, extra_params: Dict, limit: int = 1000, timeout: int = 60, verbose: bool = True) -> List[dict]:
    all_notes: List[dict] = []
    offset = 0
    while True:
        params = {"invitation": invitation, "limit": limit, "offset": offset}
        params.update(extra_params or {})
        payload = http_get(baseurl, "/notes", params=params, timeout=timeout)
        notes = extract_notes(payload)
        if not notes:
            if verbose:
                print(f"    page offset={offset}: 0 notes (stop). total={len(all_notes)}")
            break
        all_notes.extend(notes)
        if verbose:
            print(f"    page offset={offset}: +{len(notes)} notes (total={len(all_notes)})")
        offset += limit
        if offset > 200000:
            break
    return all_notes


def normalize_note(note: dict) -> Tuple[str, str]:
    c = note.get("content", {}) or {}

    # 处理API v2格式（字段可能是dict with 'value'）和API v1格式（直接字符串）
    def extract_value(field):
        if field is None:
            return ""
        if isinstance(field, dict):
            return str(field.get("value", ""))
        return str(field)

    title = extract_value(c.get("title")).strip()
    abstract = extract_value(c.get("abstract")).strip()
    tldr = extract_value(c.get("TL;DR") or c.get("TLDR")).strip()

    if (not abstract) and tldr:
        abstract = tldr
    return title, abstract


def invitation_candidates(conf: str, year: int) -> List[str]:
    venue = f"{conf}.cc/{year}/Conference"
    # API v2 格式（2023+主要使用）
    candidates = [
        f"{venue}/-/Submission",
        f"{venue}/-/Blind_Submission",
        f"{venue}/-/Paper",
    ]
    # API v1/旧格式（作为备选）
    venue_lower = f"{conf}.cc/{year}/conference"
    candidates.extend([
        f"{venue_lower}/-/submission",
        f"{venue_lower}/-/blind_submission",
        f"{venue_lower}/-/Blind_Submission",
    ])
    return candidates


def try_fetch_accepted(conf: str, year: int, verbose: bool, timeout: int) -> Tuple[str, str, int, int, List[dict]]:
    venueid = f"{conf}.cc/{year}/Conference"
    # API v2 endpoint优先（2023+主要使用）
    baseurls = ["https://api2.openreview.net", "https://api.openreview.net"]
    invs = invitation_candidates(conf, year)
    select = "id,number,content.title,content.abstract,content.TL;DR,content.TLDR,content.venueid,content.venue"

    last_errs = []
    for base in baseurls:
        for inv in invs:
            if verbose:
                print(f"[{conf} {year}] probing base={base} invitation={inv}", flush=True)

            # 尝试1: 通过content.venueid过滤accepted papers（API v2推荐方式）
            extra = {"select": select, "content.venueid": venueid}
            try:
                acc_notes = fetch_notes_paginated(base, inv, extra_params=extra, limit=1000, timeout=timeout, verbose=verbose)
                if acc_notes:
                    if verbose:
                        print(f"  ✓ found {len(acc_notes)} accepted papers via content.venueid", flush=True)
                    return base, inv, -1, len(acc_notes), acc_notes
            except Exception as e:
                last_errs.append(f"{base} {inv} (venueid filter): {str(e)[:200]}")
                if verbose:
                    print(f"  !! venueid filter failed: {e}", flush=True)

            # 尝试2: 获取所有submissions，手动过滤accepted（备选方案）
            extra2 = {"select": select}
            try:
                subs = fetch_notes_paginated(base, inv, extra_params=extra2, limit=1000, timeout=timeout, verbose=False)
                if subs:
                    # 手动过滤accepted papers
                    accepted = []
                    for note in subs:
                        content = note.get("content", {}) or {}
                        note_venueid = content.get("venueid", "")
                        venue = content.get("venue", "")
                        # 检查是否为accepted paper
                        if (note_venueid == venueid or
                            (venue and ("accept" in venue.lower() or f"{conf} {year}" in venue))):
                            accepted.append(note)

                    if accepted:
                        if verbose:
                            print(f"  ✓ found {len(accepted)} accepted papers from {len(subs)} submissions (manual filter)", flush=True)
                        return base, inv, len(subs), len(accepted), accepted
                    else:
                        last_errs.append(f"{base} {inv}: has submissions={len(subs)} but no accepted papers found")
                        if verbose:
                            print(f"  !! {len(subs)} submissions but no accepted papers", flush=True)
            except Exception as e:
                last_errs.append(f"{base} {inv} (all submissions): {str(e)[:200]}")
                if verbose:
                    print(f"  !! fetch all submissions failed: {e}", flush=True)

    raise RuntimeError("Could not fetch any submissions/accepted. Errors (last 12):\n" + "\n".join(last_errs[-12:]))


def set_cjk_font(preferred_font: str = None):
    candidates = [
        "PingFang SC", "Heiti SC", "Songti SC",
        "Microsoft YaHei", "SimHei",
        "Noto Sans CJK SC", "Noto Sans CJK", "Source Han Sans SC", "WenQuanYi Zen Hei",
    ]
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}

    chosen = preferred_font if preferred_font else None
    if not chosen:
        for name in candidates:
            if name in available:
                chosen = name
                break

    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
        plt.rcParams["font.family"] = "sans-serif"
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["font.family"] = "sans-serif"
        print(
            "[WARN] No CJK font found by name. Chinese may not render. "
            "Install a CJK font (e.g., PingFang SC / Microsoft YaHei / Noto Sans CJK SC) "
            "or run with --font '<Font Name>'.",
            file=sys.stderr,
            flush=True,
        )

    plt.rcParams["axes.unicode_minus"] = False


def _wrap(s: str, width: int = 18) -> str:
    out, cur = [], ""
    for ch in s:
        cur += ch
        if len(cur) >= width and ch not in (" ", "：", ":", "/", "-"):
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return "\n".join(out)


def plot_donut_pie(df: pd.DataFrame, outpath: str, title: str, subtitle: str):
    counts = df["category"].value_counts()
    labels = counts.index.tolist()
    disp_labels = [_display_label(x) for x in labels]
    values = counts.values.tolist()

    fig = plt.figure(figsize=(13.5, 9), dpi=220)
    ax = fig.add_subplot(111)

    wedges, _, autotexts = ax.pie(
        values,
        startangle=90,
        counterclock=False,
        autopct=lambda p: f"{p:.1f}%" if p >= 2.0 else "",
        pctdistance=0.78,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.2),
    )
    for t in autotexts:
        t.set_fontsize(10)

    ax.set_title(title, fontsize=18, pad=16)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=11)

    total = int(counts.sum())
    ax.text(0, 0, f"ICL相关\n{total} 篇", ha="center", va="center", fontsize=16, fontweight="bold")

    legend_labels = [f"{_wrap(lab)}\n{val}篇" for lab, val in zip(disp_labels, values)]
    ax.legend(
        wedges, legend_labels,
        title="研究方向（规则分类）",
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=False, fontsize=10.5, title_fontsize=12,
        labelspacing=0.8, handlelength=1.2,
    )

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_trend(df: pd.DataFrame, outpath: str, title: str, topk: int, years: List[int]):
    pivot = (
        df.groupby(["year", "category"]).size().reset_index(name="count")
          .pivot(index="year", columns="category", values="count")
          .fillna(0).astype(int).sort_index()
    )

    years = [int(y) for y in years]
    pivot = pivot.reindex(years, fill_value=0)

    totals = pivot.sum(axis=0).sort_values(ascending=False)
    keep = totals.head(topk).index.tolist()
    pivot_small = pivot[keep].copy() if keep else pivot.copy()
    if len(totals) > topk:
        pivot_small["🧺 其他（长尾）"] = pivot.drop(columns=keep).sum(axis=1)

    fig = plt.figure(figsize=(15.5, 7.8), dpi=220)
    ax = fig.add_subplot(111)

    for col in pivot_small.columns:
        ax.plot(
            pivot_small.index,
            pivot_small[col],
            marker="o",
            linestyle="-",
            linewidth=2,
            label=_display_label(col),
        )

    ax.set_title(title, fontsize=18, pad=14)
    ax.set_xlabel("年份", fontsize=12)
    ax.set_ylabel("论文数量（篇）", fontsize=12)
    ax.set_xticks(years)
    ax.grid(True, linestyle="--", alpha=0.35)

    yearly_total = pivot.sum(axis=1)
    ymax = max(1, pivot_small.max(axis=1).max())
    for x, y in yearly_total.items():
        ax.annotate(f"总计 {int(y)}", (x, ymax), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=10)

    ax.set_ylim(0, max(2, ymax + 3))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    ap.add_argument("--confs", nargs="+", default=["ICLR", "ICML"])
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--quiet", action="store_true", help="减少日志输出")
    ap.add_argument("--plot_only", action="store_true", help="仅从已保存的 CSV 生成图像（不重新抓取）")
    ap.add_argument("--data_csv", default=None, help="plot_only 模式下使用的 CSV 路径（默认 outdir/icl_papers_filtered.csv）")
    ap.add_argument("--font", default=None, help="指定 Matplotlib 字体名称（解决中文显示问题，例如 'PingFang SC'）")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_cjk_font(args.font)

    verbose = not args.quiet
    print(f"Running {os.path.basename(__file__)} {VERSION}", flush=True)

    if args.plot_only:
        csv_path = args.data_csv or os.path.join(args.outdir, "icl_papers_filtered.csv")
        if not os.path.exists(csv_path):
            print(f"[ERROR] plot_only=True but CSV not found: {csv_path}", file=sys.stderr)
            return
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        for col in ("year", "category"):
            if col not in df.columns:
                print(f"[ERROR] CSV missing required column: {col}", file=sys.stderr)
                return

        pie_out = os.path.join(args.outdir, "icl_pie_donut_refined.png")
        trend_out = os.path.join(args.outdir, "icl_trend_lines_refined.png")

        confs_str = " & ".join(args.confs)
        year_min, year_max = min(args.years), max(args.years)
        subtitle = "口径：OpenReview title+abstract（已抓取并保存）；分类：规则匹配（可复现）"

        plot_donut_pie(
            df, pie_out,
            title=f"{confs_str}（{year_min}–{year_max}）ICL 相关论文：研究方向占比（细分 taxonomy，title+abstract）",
            subtitle=subtitle
        )
        plot_trend(
            df, trend_out,
            title=f"{confs_str}（{year_min}–{year_max}）ICL 相关论文：研究方向发文趋势（细分 taxonomy，title+abstract）",
            topk=args.topk,
            years=args.years,
        )

        print("\n[OK] Plot regenerated from saved CSV:")
        print(" -", pie_out)
        print(" -", trend_out)
        return

    rows: List[Dict] = []
    meta_rows: List[Dict] = []

    for conf in args.confs:
        for year in args.years:
            try:
                base, inv, sub_n, acc_n, acc_notes = try_fetch_accepted(conf, year, verbose=verbose, timeout=args.timeout)
                meta_rows.append({
                    "conf": conf, "year": year,
                    "baseurl": base, "invitation": inv,
                    "submissions": sub_n, "accepted": acc_n
                })
                if verbose:
                    print(f"[{conf} {year}] ✅ accepted fetched: {acc_n} (base={base}, inv={inv})", flush=True)

                for n in tqdm(acc_notes, disable=not verbose, desc=f"{conf}-{year} filter"):
                    title, abstract = normalize_note(n)
                    text = f"{title}\n{abstract}"
                    if not ICL_REGEX.search(text):
                        continue
                    cat = classify(text)
                    rows.append({
                        "conf": conf, "year": year,
                        "title": title, "abstract": abstract,
                        "category": cat
                    })

                if verbose:
                    print(f"[{conf} {year}] ICL matched: {sum(1 for r in rows if r['conf']==conf and r['year']==year)}", flush=True)

            except Exception as e:
                meta_rows.append({
                    "conf": conf, "year": year,
                    "baseurl": "", "invitation": "",
                    "submissions": 0, "accepted": 0,
                    "error": str(e)[:400]
                })
                print(f"[{conf} {year}] ❌ FAILED: {e}", file=sys.stderr, flush=True)

    meta = pd.DataFrame(meta_rows)
    meta_path = os.path.join(args.outdir, "fetch_meta.csv")
    meta.to_csv(meta_path, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    df_path = os.path.join(args.outdir, "icl_papers_filtered.csv")
    df.to_csv(df_path, index=False, encoding="utf-8-sig")

    if df.empty:
        print("\nNo papers matched ICL_REGEX under the accepted set.", flush=True)
        print("Try relaxing ICL_TERMS OR verify that accepted set is fetched correctly.", flush=True)
        print(f"See: {meta_path}", flush=True)
        return

    pie_out = os.path.join(args.outdir, "icl_pie_donut_refined.png")
    trend_out = os.path.join(args.outdir, "icl_trend_lines_refined.png")

    confs_str = " & ".join(args.confs)
    year_min, year_max = min(args.years), max(args.years)
    subtitle = "口径：OpenReview title+abstract；accepted 过滤优先用 content.venueid；分类：规则匹配（可复现）"

    plot_donut_pie(
        df, pie_out,
        title=f"{confs_str}（{year_min}–{year_max}）ICL 相关论文：研究方向占比（细分 taxonomy，title+abstract）",
        subtitle=subtitle
    )
    plot_trend(
        df, trend_out,
        title=f"{confs_str}（{year_min}–{year_max}）ICL 相关论文：研究方向发文趋势（细分 taxonomy，title+abstract）",
        topk=args.topk,
        years=args.years,
    )

    print("\nSaved:", flush=True)
    print(" -", df_path)
    print(" -", meta_path)
    print(" -", pie_out)
    print(" -", trend_out)


if __name__ == "__main__":
    main()
