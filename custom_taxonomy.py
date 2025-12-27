#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义分类体系模块

支持用户自定义主题和类别，自动生成匹配规则。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Dict
from icl_taxonomy import Category, RuleClassifier


def normalize_keyword(keyword: str) -> str:
    """标准化关键词（去除空格、转小写）"""
    return keyword.strip().lower()


def generate_keyword_variants(keyword: str) -> List[str]:
    """
    自动生成关键词的变体形式

    Args:
        keyword: 原始关键词

    Returns:
        关键词变体列表（包含原词）
    """
    variants = [keyword]
    keyword_lower = keyword.lower()

    # 1. 添加复数形式（简单规则）
    if not keyword_lower.endswith('s'):
        variants.append(keyword + 's')
    if keyword_lower.endswith('y') and len(keyword) > 2:
        variants.append(keyword[:-1] + 'ies')

    # 2. 添加ing形式
    if len(keyword) > 3:
        if keyword_lower.endswith('e'):
            variants.append(keyword[:-1] + 'ing')
        else:
            variants.append(keyword + 'ing')

    # 3. 添加ed形式
    if len(keyword) > 3:
        if keyword_lower.endswith('e'):
            variants.append(keyword + 'd')
        else:
            variants.append(keyword + 'ed')

    # 4. 添加连字符和空格变体
    if '-' in keyword:
        variants.append(keyword.replace('-', ' '))
        variants.append(keyword.replace('-', ''))
    if ' ' in keyword:
        variants.append(keyword.replace(' ', '-'))
        variants.append(keyword.replace(' ', ''))

    return list(set(variants))


def extract_keywords_from_label(label: str) -> List[str]:
    """
    从类别标签中提取关键词

    Args:
        label: 类别标签（如"视觉语言融合"、"跨模态检索"）

    Returns:
        提取的关键词列表
    """
    keywords = []

    # 中英文映射词典（基础版本）
    cn_to_en = {
        '视觉': ['visual', 'vision', 'image'],
        '语言': ['language', 'linguistic', 'text', 'nlp'],
        '融合': ['fusion', 'integration', 'multimodal'],
        '检索': ['retrieval', 'search', 'matching'],
        '跨模态': ['cross-modal', 'multimodal'],
        '生成': ['generation', 'generative', 'generate'],
        '图像': ['image', 'visual', 'picture'],
        '文本': ['text', 'language', 'textual'],
        '推理': ['reasoning', 'inference'],
        '模型': ['model', 'network'],
        '训练': ['training', 'learning'],
        '优化': ['optimization', 'efficient'],
        '评测': ['evaluation', 'benchmark', 'assessment'],
        '安全': ['safety', 'secure', 'robust'],
        '应用': ['application', 'applied'],
    }

    # 提取中文关键词对应的英文
    for cn_word, en_words in cn_to_en.items():
        if cn_word in label:
            keywords.extend(en_words)

    # 分词（简单按常见分隔符）
    words = re.split(r'[、，,：:；;\s]+', label)
    for word in words:
        if word:
            keywords.append(word)
            # 对英文词添加变体
            if re.match(r'^[a-zA-Z-]+$', word):
                keywords.extend(generate_keyword_variants(word))

    return list(set(keywords))


def parse_categories_string(categories_str: str) -> Dict[str, List[str]]:
    """
    解析用户输入的类别字符串

    格式: "类别1:关键词1,关键词2;类别2:关键词A,关键词B"

    Args:
        categories_str: 类别定义字符串

    Returns:
        类别字典 {类别名: [关键词列表]}
    """
    if not categories_str or not categories_str.strip():
        raise ValueError("Categories string cannot be empty")

    categories = {}

    # 按分号分割各个类别
    category_parts = categories_str.split(';')

    for part in category_parts:
        part = part.strip()
        if not part:
            continue

        # 按冒号分割类别名和关键词
        if ':' not in part:
            raise ValueError(f"Invalid category format: '{part}'. Expected format: '类别名:关键词1,关键词2'")

        label, keywords_str = part.split(':', 1)
        label = label.strip()

        if not label:
            raise ValueError("Category label cannot be empty")

        # 解析关键词
        user_keywords = [normalize_keyword(kw) for kw in keywords_str.split(',') if kw.strip()]

        # 自动补充关键词
        auto_keywords = extract_keywords_from_label(label)

        # 合并用户关键词和自动生成的关键词
        all_keywords = list(set(user_keywords + auto_keywords))

        categories[label] = all_keywords

    if len(categories) == 0:
        raise ValueError("At least one category must be defined")

    return categories


def create_custom_taxonomy(categories_dict: Dict[str, List[str]]) -> Tuple[List[Category], str, str]:
    """
    创建自定义分类体系

    Args:
        categories_dict: 类别字典 {类别名: [关键词列表]}

    Returns:
        (Category列表, default_key, default_label)
    """
    custom_categories = []

    for idx, (label, keywords) in enumerate(categories_dict.items(), 1):
        # 生成category key（使用拼音或简化标识符）
        key = f"custom_{idx}"

        # 为每个关键词生成正则表达式模式
        patterns = []
        for keyword in keywords:
            # 转义特殊字符
            escaped_kw = re.escape(keyword)
            # 添加词边界匹配
            pattern = rf"\b{escaped_kw}\b"
            patterns.append(pattern)

        # 生成描述
        description = f"包含关键词：{', '.join(keywords[:10])}" + ("..." if len(keywords) > 10 else "")

        category = Category(
            key=key,
            label=label,
            description=description,
            patterns=tuple(patterns)
        )

        custom_categories.append(category)

    # 默认类别
    default_key = "other"
    default_label = "其他/未归类"

    return custom_categories, default_key, default_label


class CustomRuleClassifier(RuleClassifier):
    """自定义规则分类器"""

    def __init__(self, categories: List[Category], default_key: str, default_label: str):
        """
        初始化自定义分类器

        Args:
            categories: 自定义类别列表
            default_key: 默认类别key
            default_label: 默认类别label
        """
        self.categories = categories
        self.default_label = default_label
        self.default_key = default_key

    def classify(self, text: str) -> str:
        """分类文本，返回类别标签"""
        for cat in self.categories:
            for pattern in cat.patterns:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    return cat.label
        return self.default_label

    def classify_with_key(self, text: str) -> Tuple[str, str]:
        """分类文本，返回(key, label)"""
        for cat in self.categories:
            for pattern in cat.patterns:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    return cat.key, cat.label
        return self.default_key, self.default_label

    def get_category_by_key(self, key: str) -> Category | None:
        """根据key获取类别"""
        for cat in self.categories:
            if cat.key == key:
                return cat
        return None

    def get_all_categories(self) -> List[Category]:
        """获取所有类别"""
        return self.categories

    def get_category_definitions_for_llm(self) -> str:
        """生成给LLM的分类定义"""
        lines = []
        for i, cat in enumerate(self.categories, 1):
            lines.append(f"{i}. **{cat.label}** (key: {cat.key})")
            lines.append(f"   {cat.description}")
            lines.append("")

        # 添加"其他"类别
        lines.append(f"{len(self.categories) + 1}. **{self.default_label}** (key: {self.default_key})")
        lines.append("   无法归入以上任何类别的论文")

        return "\n".join(lines)


def create_topic_filter(topic: str) -> tuple:
    """
    创建主题过滤器

    Args:
        topic: 主题关键词（如 "multimodal learning"）

    Returns:
        (topic_regex, topic_description)
    """
    # 提取关键词
    keywords = [kw.strip() for kw in re.split(r'[,\s]+', topic) if kw.strip()]

    # 自动补充变体
    all_keywords = []
    for kw in keywords:
        all_keywords.append(kw)
        all_keywords.extend(generate_keyword_variants(kw))

    # 去重
    all_keywords = list(set(all_keywords))

    # 生成正则表达式
    patterns = [rf"\b{re.escape(kw)}\b" for kw in all_keywords]
    topic_regex = re.compile("|".join(patterns), flags=re.IGNORECASE)

    # 描述
    topic_description = f"Topic: {topic} (keywords: {', '.join(keywords)})"

    return topic_regex, topic_description
