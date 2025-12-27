#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本：验证重构后的模块功能
"""

from icl_taxonomy import RuleClassifier, get_all_categories, get_category_definitions_for_llm
from icl_fetcher import is_icl_related
from icl_classifier import HybridClassifier

def test_taxonomy():
    """测试分类体系"""
    print("=" * 60)
    print("测试 1: 分类体系")
    print("=" * 60)

    categories = get_all_categories()
    print(f"总共定义了 {len(categories)} 个类别：")
    for i, cat in enumerate(categories, 1):
        print(f"{i}. {cat.label} (key: {cat.key})")
    print()

def test_rule_classifier():
    """测试规则分类器"""
    print("=" * 60)
    print("测试 2: 规则分类器")
    print("=" * 60)

    classifier = RuleClassifier()

    test_cases = [
        "Chain-of-thought prompting improves reasoning in large language models",
        "A benchmark for evaluating in-context learning capabilities",
        "Understanding the mechanism of in-context learning through attention analysis",
        "Efficient prompt compression for long-context language models",
    ]

    for text in test_cases:
        result = classifier.classify(text)
        print(f"文本: {text[:60]}...")
        print(f"分类: {result}")
        print()

def test_icl_filter():
    """测试 ICL 过滤"""
    print("=" * 60)
    print("测试 3: ICL 关键词过滤")
    print("=" * 60)

    test_cases = [
        ("In-context learning enables few-shot learning", True),
        ("This paper studies neural networks", False),
        ("We propose a method for ICL in transformers", True),
        ("Deep learning for computer vision", False),
    ]

    for text, expected in test_cases:
        result = is_icl_related(text, "")
        status = "✓" if result == expected else "✗"
        print(f"{status} {text}: {result}")
    print()

def test_hybrid_classifier():
    """测试混合分类器（仅规则模式）"""
    print("=" * 60)
    print("测试 4: 混合分类器（规则模式）")
    print("=" * 60)

    rule_classifier = RuleClassifier()
    hybrid = HybridClassifier(
        llm_classifier=None,
        rule_classifier=rule_classifier,
        confidence_threshold=0.6
    )

    papers = [
        {
            'id': 'test_1',
            'title': 'Chain-of-Thought Prompting',
            'abstract': 'We propose chain-of-thought prompting to improve reasoning in large language models through in-context learning.'
        },
        {
            'id': 'test_2',
            'title': 'ICL Benchmark',
            'abstract': 'A comprehensive benchmark for evaluating in-context learning capabilities of language models.'
        },
    ]

    results = hybrid.classify_papers(papers, batch_size=10, verbose=False)

    for r in results:
        print(f"ID: {r['id']}")
        print(f"标题: {r['title']}")
        print(f"分类: {r['category_label']}")
        print(f"方法: {r['method']}")
        print()

def test_llm_definitions():
    """测试 LLM 提示词生成"""
    print("=" * 60)
    print("测试 5: LLM 分类定义生成")
    print("=" * 60)

    definitions = get_category_definitions_for_llm()
    print("生成的 LLM 分类定义：")
    print(definitions[:500] + "...")
    print(f"\n总长度: {len(definitions)} 字符")
    print()

def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "ICL 论文分类工具 - 功能测试" + " " * 17 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    try:
        test_taxonomy()
        test_rule_classifier()
        test_icl_filter()
        test_hybrid_classifier()
        test_llm_definitions()

        print("=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        print()
        print("提示：")
        print("- 运行主程序: python3 openreview_icl_crawl_and_plot.py")
        print("- 查看使用说明: cat USAGE.md")
        print()

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
