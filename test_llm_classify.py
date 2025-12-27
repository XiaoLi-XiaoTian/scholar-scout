#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 LLM 分类功能（从配置文件读取 API 配置）
"""

import os
import sys

try:
    from icl_taxonomy import RuleClassifier
    from icl_classifier import LLMClassifier, HybridClassifier
    from config_loader import load_config
    print("✓ 模块导入成功")
except ImportError as e:
    print(f"✗ 模块导入失败: {e}")
    sys.exit(1)

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


def test_llm_classifier():
    """测试 LLM 分类器"""
    print("\n" + "=" * 70)
    print("测试 LLM 分类器")
    print("=" * 70)

    try:
        # 初始化 LLM 分类器
        print(f"\n[1] 初始化 LLM 分类器...")
        print(f"    API Base: {API_CONFIG['api_base']}")
        print(f"    Model: {API_CONFIG['model']}")

        llm_classifier = LLMClassifier(
            api_base=API_CONFIG['api_base'],
            api_key=API_CONFIG['api_key'],
            model=API_CONFIG['model'],
            cache_file="test_llm_cache.json",
            max_rpm=10  # 降低速率以避免触发限制
        )
        print("    ✓ 分类器初始化成功")

        # 准备测试论文
        print("\n[2] 准备测试论文...")
        test_papers = [
            {
                'id': 'paper_1',
                'title': 'Chain-of-Thought Prompting Elicits Reasoning in Large Language Models',
                'abstract': 'We explore how generating a chain of thought—a series of intermediate reasoning steps—significantly improves the ability of large language models to perform complex reasoning. We show that chain-of-thought prompting is a simple and broadly applicable technique that can be used with any language model via few-shot in-context learning.'
            },
            {
                'id': 'paper_2',
                'title': 'A Comprehensive Benchmark for In-Context Learning',
                'abstract': 'We present a comprehensive benchmark for evaluating in-context learning capabilities of language models across diverse tasks. Our benchmark includes multiple evaluation metrics and provides detailed analysis of model performance under different in-context learning settings.'
            }
        ]

        for i, paper in enumerate(test_papers, 1):
            print(f"    Paper {i}: {paper['title'][:50]}...")

        # 执行分类
        print("\n[3] 执行 LLM 分类...")
        print("    (这可能需要几秒钟...)")

        results = llm_classifier.classify_batch(test_papers, batch_size=2)

        print("    ✓ 分类完成")

        # 显示结果
        print("\n[4] 分类结果:")
        print("-" * 70)

        for paper, result in zip(test_papers, results):
            print(f"\n📄 论文: {paper['title']}")
            print(f"   分类: {result['category_label']}")
            print(f"   置信度: {result['confidence']:.2f}")
            print(f"   理由: {result['reasoning']}")
            print(f"   来源: {'缓存' if result.get('from_cache', False) else 'API调用'}")

        print("\n" + "=" * 70)
        print("✅ LLM 分类器测试成功！")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_classifier():
    """测试混合分类器"""
    print("\n" + "=" * 70)
    print("测试混合分类器（LLM + 规则回退）")
    print("=" * 70)

    try:
        # 初始化分类器
        print("\n[1] 初始化混合分类器...")

        rule_classifier = RuleClassifier()
        llm_classifier = LLMClassifier(
            api_base=API_CONFIG['api_base'],
            api_key=API_CONFIG['api_key'],
            model=API_CONFIG['model'],
            cache_file="test_llm_cache.json",
            max_rpm=10
        )

        hybrid_classifier = HybridClassifier(
            llm_classifier=llm_classifier,
            rule_classifier=rule_classifier,
            confidence_threshold=0.6
        )
        print("    ✓ 混合分类器初始化成功")

        # 准备测试论文
        print("\n[2] 准备测试论文...")
        test_papers = [
            {
                'id': 'hybrid_1',
                'title': 'Understanding In-Context Learning Mechanisms',
                'abstract': 'This paper investigates the underlying mechanisms of in-context learning in transformer models through theoretical analysis and empirical experiments.'
            }
        ]

        # 执行分类
        print("\n[3] 执行混合分类...")
        results = hybrid_classifier.classify_papers(
            test_papers,
            batch_size=1,
            verbose=False
        )

        # 显示结果
        print("\n[4] 分类结果:")
        print("-" * 70)

        for r in results:
            print(f"\n📄 论文: {r['title']}")
            print(f"   分类: {r['category_label']}")
            print(f"   置信度: {r['confidence']:.2f}")
            print(f"   方法: {r['method']}")
            if 'reasoning' in r:
                print(f"   理由: {r['reasoning']}")

        print("\n" + "=" * 70)
        print("✅ 混合分类器测试成功！")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "ICL 论文 LLM 分类功能测试" + " " * 24 + "║")
    print("╚" + "=" * 68 + "╝")

    # 检查是否安装了 openai 包
    try:
        import openai
        print(f"✓ OpenAI SDK 版本: {openai.__version__}")
    except ImportError:
        print("❌ 错误: 未安装 openai 包")
        print("   请运行: pip install openai")
        sys.exit(1)

    # 运行测试
    success = True

    # 测试 1: LLM 分类器
    if not test_llm_classifier():
        success = False

    # 测试 2: 混合分类器
    if success:
        if not test_hybrid_classifier():
            success = False

    # 最终结果
    print("\n")
    if success:
        print("🎉 所有测试通过！")
        print("\n提示:")
        print("  - 缓存文件已保存到: test_llm_cache.json")
        print("  - 下次运行相同论文将直接使用缓存")
        print("\n使用主程序:")
        print(f"  python3 openreview_icl_crawl_and_plot.py --use_llm")
    else:
        print("❌ 部分测试失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
