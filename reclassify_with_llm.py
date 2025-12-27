#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 LLM 对已抓取的论文进行重新分类

这个脚本读取之前抓取的论文数据，使用 LLM 进行智能分类，
并将结果与规则分类进行对比。
"""

import pandas as pd
import sys
import os

from icl_taxonomy import RuleClassifier
from icl_classifier import LLMClassifier, HybridClassifier
from config_loader import load_config

# 从配置文件加载 API 信息
config = load_config()
API_CONFIG = {
    "api_base": config.api_base,
    "api_key": config.api_key,
    "model": config.model
}

# 检查API密钥是否配置
if not API_CONFIG["api_key"]:
    print("\n❌ 错误: 未配置 API 密钥")
    print("请编辑 config.json 文件，在 'api.api_key' 字段中填入您的 API 密钥")
    sys.exit(1)


def main():
    # 读取已有数据
    csv_path = "out/icl_papers_filtered.csv"
    if not os.path.exists(csv_path):
        print(f"❌ 错误: 找不到文件 {csv_path}")
        print("请先运行主程序抓取数据：")
        print("  python3 openreview_icl_crawl_and_plot.py")
        return

    print("=" * 70)
    print("LLM 论文重分类工具")
    print("=" * 70)
    print()

    # 读取数据
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"✓ 读取到 {len(df)} 篇论文")
    print()

    # 询问用户要处理多少篇
    print("选项:")
    print("  1. 测试模式 (前5篇)")
    print("  2. 小规模 (前20篇)")
    print("  3. 中等规模 (前50篇)")
    print("  4. 全部论文 ({}篇)".format(len(df)))
    print()

    choice = input("请选择 [1-4，默认1]: ").strip() or "1"

    if choice == "1":
        num_papers = 5
    elif choice == "2":
        num_papers = 20
    elif choice == "3":
        num_papers = 50
    else:
        num_papers = len(df)

    df_sample = df.head(num_papers).copy()
    print(f"\n处理 {len(df_sample)} 篇论文...")
    print()

    # 准备数据
    papers = []
    for idx, row in df_sample.iterrows():
        papers.append({
            'id': f"paper_{idx}",
            'title': row['title'],
            'abstract': row['abstract'],
            'original_category': row['category']  # 原始规则分类结果
        })

    # 初始化分类器
    print("[1] 初始化分类器...")
    rule_classifier = RuleClassifier()
    llm_classifier = LLMClassifier(
        api_base=API_CONFIG['api_base'],
        api_key=API_CONFIG['api_key'],
        model=API_CONFIG['model'],
        cache_file="reclassify_cache.json",
        max_rpm=20  # 限制速率
    )

    hybrid_classifier = HybridClassifier(
        llm_classifier=llm_classifier,
        rule_classifier=rule_classifier,
        confidence_threshold=0.6
    )
    print("    ✓ 初始化完成")
    print()

    # 执行分类
    print(f"[2] 使用 LLM 分类 (模型: {API_CONFIG['model']})...")
    print("    这可能需要一些时间，请耐心等待...")
    print()

    results = hybrid_classifier.classify_papers(
        papers=[{k: v for k, v in p.items() if k != 'original_category'} for p in papers],
        batch_size=5,  # 小批量处理
        checkpoint_file="reclassify_checkpoint.json",
        verbose=True
    )

    # 对比结果
    print("\n[3] 对比分类结果:")
    print("=" * 70)

    changes = 0
    llm_used = 0

    for paper, result in zip(papers, results):
        original = paper['original_category']
        new = result['category_label']
        method = result.get('method', 'rule')

        if method == 'llm':
            llm_used += 1

        if original != new:
            changes += 1
            print(f"\n📄 {paper['title'][:60]}...")
            print(f"   原分类: {original}")
            print(f"   新分类: {new}")
            print(f"   置信度: {result['confidence']:.2f}")
            print(f"   方法: {method}")
            if 'reasoning' in result:
                print(f"   理由: {result['reasoning'][:100]}...")

    # 统计
    print("\n" + "=" * 70)
    print("统计:")
    print(f"  总计处理: {len(papers)} 篇")
    print(f"  LLM 分类: {llm_used} 篇")
    print(f"  规则分类: {len(papers) - llm_used} 篇")
    print(f"  分类改变: {changes} 篇 ({changes/len(papers)*100:.1f}%)")

    # 保存结果
    output_df = pd.DataFrame(results)
    output_path = "out/icl_papers_llm_reclassified.csv"
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ 结果已保存到: {output_path}")

    # 分类分布
    print("\n新分类分布:")
    print(output_df['category_label'].value_counts().to_string())

    print("\n" + "=" * 70)
    print("完成！")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
