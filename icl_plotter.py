#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICL论文数据可视化模块

提供饼图、折线图等可视化功能。
"""

from __future__ import annotations

import re
import sys
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


def _display_label(s: str) -> str:
    """去除标签中的emoji前缀，解决字体显示问题"""
    return re.sub(r"^[^\u4e00-\u9fffA-Za-z0-9]+\s*", "", str(s)).strip()


def _wrap(s: str, width: int = 18) -> str:
    """文本换行"""
    out, cur = [], ""
    for ch in s:
        cur += ch
        if len(cur) >= width and ch not in (" ", "：", ":", "/", "-"):
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return "\n".join(out)


def set_cjk_font(preferred_font: str = None):
    """设置中文字体"""
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
            "Install a CJK font or run with --font '<Font Name>'.",
            file=sys.stderr,
            flush=True,
        )

    plt.rcParams["axes.unicode_minus"] = False


def plot_donut_pie(df: pd.DataFrame, outpath: str, title: str, subtitle: str,
                   center_text: str = None, legend_title: str = None):
    """
    绘制甜甜圈饼图

    Args:
        df: 数据框
        outpath: 输出路径
        title: 图表标题
        subtitle: 副标题
        center_text: 中心文字（默认为"相关论文"）
        legend_title: 图例标题（默认为"研究方向分类"）
    """
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
    # 使用传入的中心文字或默认值
    center_label = center_text if center_text else "相关论文"
    ax.text(0, 0, f"{center_label}\n{total} 篇", ha="center", va="center", fontsize=16, fontweight="bold")

    legend_labels = [f"{_wrap(lab)}\n{val}篇" for lab, val in zip(disp_labels, values)]
    # 使用传入的图例标题或默认值
    legend_label = legend_title if legend_title else "研究方向分类"
    ax.legend(
        wedges, legend_labels,
        title=legend_label,
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=False, fontsize=10.5, title_fontsize=12,
        labelspacing=0.8, handlelength=1.2,
    )

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_trend(df: pd.DataFrame, outpath: str, title: str, topk: int, years: List[int],
               other_label: str = None):
    """
    绘制趋势折线图

    Args:
        df: 数据框
        outpath: 输出路径
        title: 图表标题
        topk: 显示top k个类别
        years: 年份列表
        other_label: "其他"类别的标签（默认为"其他（长尾）"）
    """
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
        # 使用传入的标签或默认值
        other_category_label = other_label if other_label else "其他（长尾）"
        pivot_small[other_category_label] = pivot.drop(columns=keep).sum(axis=1)

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
