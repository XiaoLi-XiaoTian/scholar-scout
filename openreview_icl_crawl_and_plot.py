#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenReview（ICLR/ICML）论文统计工具：title+abstract 检索 + 细分 taxonomy 分类 + 作图

v4.1 版本：
- 模块化设计：icl_taxonomy（分类体系）、icl_fetcher（数据抓取）、
  icl_plotter（可视化）、icl_classifier（分类器）
- 支持LLM分类：可选择使用OpenAI兼容的API进行智能分类
- 混合分类策略：LLM + 规则回退，提高准确性
- 支持断点续传和缓存
- ✨ 新功能：支持自定义主题和类别，灵活适配不同研究领域
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

# 导入新创建的模块
from icl_taxonomy import RuleClassifier
from icl_fetcher import try_fetch_accepted, normalize_note, is_icl_related
from icl_plotter import set_cjk_font, plot_donut_pie, plot_trend
from icl_classifier import LLMClassifier, HybridClassifier
from custom_taxonomy import (
    parse_categories_string, create_custom_taxonomy,
    CustomRuleClassifier, create_topic_filter
)
from config_loader import load_config

VERSION = "v4.2"


def main():
    # 加载配置文件
    config = load_config()

    ap = argparse.ArgumentParser(
        description="OpenReview 论文统计工具 v4.2 - 支持配置文件和自定义主题",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 使用配置文件（推荐）:
   编辑 config.json 文件，然后直接运行：
   python3 openreview_icl_crawl_and_plot.py

2. 命令行覆盖配置:
   python3 openreview_icl_crawl_and_plot.py --years 2024 2025 --use_llm

3. 自定义主题和类别:
   python3 openreview_icl_crawl_and_plot.py \\
     --topic "multimodal learning" \\
     --categories "视觉语言融合:vision,language,VLM;跨模态检索:retrieval,cross-modal"

注意：命令行参数优先级高于配置文件
        """
    )

    # ========== 配置文件参数 ==========
    ap.add_argument("--config", default="config.json",
                    help="配置文件路径（默认: config.json）")

    # ========== 自定义主题和类别参数 ==========
    ap.add_argument("--topic", default=None,
                    help="自定义研究主题（用于论文过滤），如 'multimodal learning'。不指定则从配置文件读取或使用ICL默认主题")
    ap.add_argument("--categories", default=None,
                    help="自定义类别定义，格式: '类别1:关键词1,关键词2;类别2:关键词A,关键词B'。不指定则从配置文件读取或使用ICL默认分类")

    # ========== 基础参数 ==========
    ap.add_argument("--years", nargs="+", type=int, default=None,
                    help=f"要抓取的年份列表（默认从配置: {config.years}）")
    ap.add_argument("--confs", nargs="+", default=None,
                    help=f"要抓取的会议列表（默认从配置: {config.conferences}）")
    ap.add_argument("--outdir", default=None,
                    help=f"输出目录（默认从配置: {config.output_dir}）")
    ap.add_argument("--topk", type=int, default=None,
                    help=f"趋势图中显示的top类别数量（默认从配置: {config.topk_trends}）")
    ap.add_argument("--timeout", type=int, default=None,
                    help=f"HTTP请求超时时间（秒）（默认从配置: {config.fetch_timeout}）")

    # 运行模式
    ap.add_argument("--quiet", action="store_true", default=None,
                    help="减少日志输出")
    ap.add_argument("--plot_only", action="store_true", default=None,
                    help="仅从已保存的CSV生成图像（不重新抓取）")
    ap.add_argument("--data_csv", default=None,
                    help="plot_only模式下使用的CSV路径")

    # 可视化参数
    ap.add_argument("--font", default=None,
                    help=f"指定Matplotlib字体名称（默认从配置: {config.font or '自动'}）")

    # LLM分类器参数
    ap.add_argument("--use_llm", action="store_true", default=None,
                    help=f"使用LLM进行智能分类（默认从配置: {config.use_llm}）")
    ap.add_argument("--llm_api_base", default=None,
                    help=f"LLM API base URL（默认从配置: {config.api_base}）")
    ap.add_argument("--llm_api_key", default=None,
                    help="LLM API key（优先级：命令行 > 配置文件 > 环境变量OPENAI_API_KEY）")
    ap.add_argument("--llm_model", default=None,
                    help=f"LLM模型名称（默认从配置: {config.model}）")
    ap.add_argument("--llm_batch_size", type=int, default=None,
                    help=f"LLM批处理大小（默认从配置: {config.llm_batch_size}）")
    ap.add_argument("--llm_max_rpm", type=int, default=None,
                    help=f"LLM API每分钟最大请求数（默认从配置: {config.max_rpm}）")
    ap.add_argument("--llm_cache_file", default=None,
                    help=f"LLM分类结果缓存文件（默认从配置: {config.cache_file}）")
    ap.add_argument("--llm_confidence_threshold", type=float, default=None,
                    help=f"LLM分类置信度阈值（默认从配置: {config.llm_confidence_threshold}）")
    ap.add_argument("--checkpoint_file", default=None,
                    help=f"分类checkpoint文件（默认从配置: {config.checkpoint_file}）")

    args = ap.parse_args()

    # 合并配置：命令行参数优先于配置文件
    years = args.years if args.years is not None else config.years
    confs = args.confs if args.confs is not None else config.conferences
    outdir = args.outdir if args.outdir is not None else config.output_dir
    topk = args.topk if args.topk is not None else config.topk_trends
    timeout = args.timeout if args.timeout is not None else config.fetch_timeout
    quiet = args.quiet if args.quiet is not None else config.quiet
    plot_only = args.plot_only if args.plot_only is not None else config.plot_only
    data_csv = args.data_csv if args.data_csv else config.data_csv
    font = args.font if args.font is not None else config.font
    use_llm = args.use_llm if args.use_llm is not None else config.use_llm
    llm_api_base = args.llm_api_base if args.llm_api_base is not None else config.api_base
    llm_api_key = args.llm_api_key if args.llm_api_key is not None else config.api_key
    llm_model = args.llm_model if args.llm_model is not None else config.model
    llm_batch_size = args.llm_batch_size if args.llm_batch_size is not None else config.llm_batch_size
    llm_max_rpm = args.llm_max_rpm if args.llm_max_rpm is not None else config.max_rpm
    llm_cache_file = args.llm_cache_file if args.llm_cache_file is not None else config.cache_file
    llm_confidence_threshold = args.llm_confidence_threshold if args.llm_confidence_threshold is not None else config.llm_confidence_threshold
    checkpoint_file = args.checkpoint_file if args.checkpoint_file is not None else config.checkpoint_file
    topic = args.topic if args.topic is not None else config.custom_topic
    categories = args.categories if args.categories is not None else config.custom_categories

    # 创建一个合并后的args对象（用于后续代码兼容）
    class MergedArgs:
        pass

    merged_args = MergedArgs()
    merged_args.years = years
    merged_args.confs = confs
    merged_args.outdir = outdir
    merged_args.topk = topk
    merged_args.timeout = timeout
    merged_args.quiet = quiet
    merged_args.plot_only = plot_only
    merged_args.data_csv = data_csv
    merged_args.font = font
    merged_args.use_llm = use_llm
    merged_args.llm_api_base = llm_api_base
    merged_args.llm_api_key = llm_api_key
    merged_args.llm_model = llm_model
    merged_args.llm_batch_size = llm_batch_size
    merged_args.llm_max_rpm = llm_max_rpm
    merged_args.llm_cache_file = llm_cache_file
    merged_args.llm_confidence_threshold = llm_confidence_threshold
    merged_args.checkpoint_file = checkpoint_file
    merged_args.topic = topic
    merged_args.categories = categories

    args = merged_args

    os.makedirs(args.outdir, exist_ok=True)
    set_cjk_font(args.font)

    verbose = not args.quiet
    print(f"Running {os.path.basename(__file__)} {VERSION}", flush=True)

    # =============================================================================
    # plot_only 模式：仅重新生成图表
    # =============================================================================
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

        # 如果category列是category_label，需要确认
        if 'category_label' in df.columns and 'category' not in df.columns:
            df['category'] = df['category_label']

        pie_out = os.path.join(args.outdir, "icl_pie_donut_refined.png")
        trend_out = os.path.join(args.outdir, "icl_trend_lines_refined.png")

        confs_str = " & ".join(args.confs)
        year_min, year_max = min(args.years), max(args.years)
        subtitle = "口径：OpenReview title+abstract；分类：规则或LLM（可复现）"

        plot_donut_pie(
            df, pie_out,
            title=f"{confs_str}（{year_min}–{year_max}）ICL 相关论文：研究方向占比（细分 taxonomy）",
            subtitle=subtitle
        )
        plot_trend(
            df, trend_out,
            title=f"{confs_str}（{year_min}–{year_max}）ICL 相关论文：研究方向发文趋势（细分 taxonomy）",
            topk=args.topk,
            years=args.years,
        )

        print("\n[OK] Plot regenerated from saved CSV:")
        print(" -", pie_out)
        print(" -", trend_out)
        return

    # =============================================================================
    # 数据抓取 + 分类 + 绘图
    # =============================================================================

    # 1. 判断是否使用自定义模式（只有当topic和categories都非空时才启用）
    use_custom_mode = (args.topic is not None and args.topic != "") or \
                      (args.categories is not None and args.categories != "")

    if use_custom_mode:
        # 验证参数
        if not args.topic or not args.categories:
            print("[ERROR] --topic and --categories must be used together and cannot be empty", file=sys.stderr)
            print("Example: --topic 'multimodal learning' --categories '类别1:kw1,kw2;类别2:kw3'", file=sys.stderr)
            return

        try:
            # 解析自定义类别
            categories_dict = parse_categories_string(args.categories)
            custom_categories, default_key, default_label = create_custom_taxonomy(categories_dict)

            # 创建主题过滤器
            topic_regex, topic_desc = create_topic_filter(args.topic)

            if verbose:
                print(f"\n[Custom Mode] Topic: {args.topic}")
                print(f"[Custom Mode] Defined {len(categories_dict)} categories:")
                for i, (label, keywords) in enumerate(categories_dict.items(), 1):
                    print(f"  {i}. {label}: {', '.join(keywords[:5])}" + ("..." if len(keywords) > 5 else ""))
                print()

            # 使用自定义分类器
            rule_classifier = CustomRuleClassifier(custom_categories, default_key, default_label)

            # 自定义过滤函数
            def is_topic_related(title: str, abstract: str) -> bool:
                text = f"{title}\n{abstract}"
                return topic_regex.search(text) is not None

            filter_func = is_topic_related
            topic_name = args.topic

        except Exception as e:
            print(f"[ERROR] Failed to parse custom categories: {e}", file=sys.stderr)
            return
    else:
        # 默认ICL模式
        if verbose:
            print(f"\n[Default Mode] Using ICL taxonomy with 9 predefined categories")

        rule_classifier = RuleClassifier()
        filter_func = is_icl_related
        topic_name = "ICL"

    # 2. 初始化LLM分类器（如果需要）
    llm_classifier = None

    if args.use_llm:
        try:
            api_key = args.llm_api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("[WARN] --use_llm specified but no API key provided. Using rule-based classification only.",
                      file=sys.stderr)
            else:
                llm_classifier = LLMClassifier(
                    api_base=args.llm_api_base,
                    api_key=api_key,
                    model=args.llm_model,
                    cache_file=os.path.join(args.outdir, args.llm_cache_file),
                    max_rpm=args.llm_max_rpm
                )
                print(f"[LLM] Using {args.llm_model} for classification")
        except ImportError as e:
            print(f"[WARN] Cannot use LLM classifier: {e}. Using rule-based classification.",
                  file=sys.stderr)

    hybrid_classifier = HybridClassifier(
        llm_classifier=llm_classifier,
        rule_classifier=rule_classifier,
        confidence_threshold=args.llm_confidence_threshold
    )

    # 3. 抓取数据
    rows: List[Dict] = []
    meta_rows: List[Dict] = []
    all_papers_for_classification = []  # 用于分类

    for conf in args.confs:
        for year in args.years:
            try:
                base, inv, sub_n, acc_n, acc_notes = try_fetch_accepted(
                    conf, year, verbose=verbose, timeout=args.timeout
                )
                meta_rows.append({
                    "conf": conf, "year": year,
                    "baseurl": base, "invitation": inv,
                    "submissions": sub_n, "accepted": acc_n
                })
                if verbose:
                    print(f"[{conf} {year}] ✅ accepted fetched: {acc_n} (base={base}, inv={inv})", flush=True)

                # 使用对应的过滤函数（自定义或默认ICL）
                for n in tqdm(acc_notes, disable=not verbose, desc=f"{conf}-{year} filter"):
                    title, abstract = normalize_note(n)
                    if not filter_func(title, abstract):  # 使用动态过滤函数
                        continue

                    paper_id = f"{conf}_{year}_{n.get('id', len(all_papers_for_classification))}"
                    all_papers_for_classification.append({
                        'id': paper_id,
                        'conf': conf,
                        'year': year,
                        'title': title,
                        'abstract': abstract
                    })

                if verbose:
                    topic_count = sum(1 for p in all_papers_for_classification
                                   if p['conf'] == conf and p['year'] == year)
                    print(f"[{conf} {year}] {topic_name} matched: {topic_count}", flush=True)

            except Exception as e:
                meta_rows.append({
                    "conf": conf, "year": year,
                    "baseurl": "", "invitation": "",
                    "submissions": 0, "accepted": 0,
                    "error": str(e)[:400]
                })
                print(f"[{conf} {year}] ❌ FAILED: {e}", file=sys.stderr, flush=True)

    # 保存抓取元数据
    meta = pd.DataFrame(meta_rows)
    meta_path = os.path.join(args.outdir, "fetch_meta.csv")
    meta.to_csv(meta_path, index=False, encoding="utf-8-sig")

    if not all_papers_for_classification:
        print(f"\nNo papers matched topic '{topic_name}' criteria.", flush=True)
        print(f"See: {meta_path}", flush=True)
        return

    # 4. 分类
    if verbose:
        print(f"\n[Classification] Starting classification for {len(all_papers_for_classification)} papers...")

    checkpoint_file = args.checkpoint_file
    if checkpoint_file:
        checkpoint_file = os.path.join(args.outdir, checkpoint_file)

    classification_results = hybrid_classifier.classify_papers(
        papers=all_papers_for_classification,
        batch_size=args.llm_batch_size,
        checkpoint_file=checkpoint_file,
        verbose=verbose
    )

    # 4. 整理结果
    df = pd.DataFrame(classification_results)

    # 保存详细结果（包含所有字段）
    output_prefix = "custom" if use_custom_mode else "icl"
    df_detailed_path = os.path.join(args.outdir, f"{output_prefix}_papers_classified_detailed.csv")
    df.to_csv(df_detailed_path, index=False, encoding="utf-8-sig")

    # 保存简化版本（兼容原有格式）
    df_simple = df[['conf', 'year', 'title', 'abstract', 'category_label']].copy()
    df_simple.rename(columns={'category_label': 'category'}, inplace=True)
    df_path = os.path.join(args.outdir, f"{output_prefix}_papers_filtered.csv")
    df_simple.to_csv(df_path, index=False, encoding="utf-8-sig")

    # 5. 输出分类统计
    if verbose:
        print("\n[Classification] Category distribution:")
        print(df['category_label'].value_counts().to_string())
        if 'method' in df.columns:
            print("\n[Classification] Method distribution:")
            print(df['method'].value_counts().to_string())

    # 6. 绘图
    pie_out = os.path.join(args.outdir, f"{output_prefix}_pie_donut_refined.png")
    trend_out = os.path.join(args.outdir, f"{output_prefix}_trend_lines_refined.png")

    confs_str = " & ".join(args.confs)
    year_min, year_max = min(args.years), max(args.years)
    method_desc = "LLM+规则混合" if args.use_llm and llm_classifier else "规则匹配"

    # 根据模式设置标题和标签
    if use_custom_mode:
        main_title = f"{confs_str}（{year_min}–{year_max}）{topic_name} 相关论文"
        subtitle = f"口径：OpenReview title+abstract；主题：{topic_name}；分类：{method_desc}"
        center_text = topic_name  # 饼图中心显示主题名
        legend_title = f"{topic_name}研究方向分类"  # 图例标题
        other_label = "其他（长尾）"  # 折线图其他类别标签
    else:
        main_title = f"{confs_str}（{year_min}–{year_max}）ICL 相关论文"
        subtitle = f"口径：OpenReview title+abstract；分类：{method_desc}（可复现）"
        center_text = "ICL相关"  # 饼图中心显示"ICL相关"
        legend_title = "研究方向（规则分类）"  # 保持原有标题
        other_label = "🧺 其他（长尾）"  # 保持原有emoji标签

    # 使用简化的df绘图
    plot_donut_pie(
        df_simple, pie_out,
        title=f"{main_title}：研究方向占比（细分 taxonomy）",
        subtitle=subtitle,
        center_text=center_text,
        legend_title=legend_title
    )
    plot_trend(
        df_simple, trend_out,
        title=f"{main_title}：研究方向发文趋势（细分 taxonomy）",
        topk=args.topk,
        years=args.years,
        other_label=other_label
    )

    print("\nSaved:", flush=True)
    print(" -", df_path)
    print(" -", df_detailed_path)
    print(" -", meta_path)
    print(" -", pie_out)
    print(" -", trend_out)


if __name__ == "__main__":
    main()
