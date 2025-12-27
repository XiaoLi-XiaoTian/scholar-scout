#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICL论文分类器模块

提供基于LLM的分类器、规则分类器和混合分类器。
"""

from __future__ import annotations

import json
import time
import hashlib
import os
from typing import Dict, List, Optional
from pathlib import Path

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from icl_taxonomy import RuleClassifier, get_category_definitions_for_llm, get_category_map_for_llm, get_category_by_key, DEFAULT_KEY, DEFAULT_LABEL


class RateLimiter:
    """速率限制器"""

    def __init__(self, max_requests_per_minute: int = 60):
        self.max_rpm = max_requests_per_minute
        self.requests = []

    def wait_if_needed(self):
        """如果超过速率限制则等待"""
        now = time.time()
        # 移除1分钟前的请求记录
        self.requests = [t for t in self.requests if now - t < 60]

        if len(self.requests) >= self.max_rpm:
            sleep_time = 60 - (now - self.requests[0])
            if sleep_time > 0:
                print(f"[Rate Limit] Sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                self.requests = []

        self.requests.append(time.time())


class LLMClassifier:
    """基于LLM的分类器"""

    def __init__(self, api_base: str, api_key: str, model: str = "gpt-3.5-turbo",
                 cache_file: str = "llm_cache.json", max_rpm: int = 60):
        if not HAS_OPENAI:
            raise ImportError("Please install openai package: pip install openai")

        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.rate_limiter = RateLimiter(max_rpm)

    def _load_cache(self) -> Dict:
        """加载缓存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        """保存缓存"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _get_paper_hash(self, title: str, abstract: str) -> str:
        """计算论文的哈希值（用于缓存key）"""
        content = f"{title}|||{abstract}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _get_cached_result(self, paper_hash: str) -> Optional[Dict]:
        """获取缓存结果"""
        return self.cache.get(paper_hash)

    def _set_cached_result(self, paper_hash: str, result: Dict):
        """设置缓存结果"""
        self.cache[paper_hash] = result
        self._save_cache()

    def classify_batch(self, papers: List[Dict], batch_size: int = 10) -> List[Dict]:
        """
        批量分类论文

        Args:
            papers: 论文列表，每个包含 {id, title, abstract}
            batch_size: 批处理大小

        Returns:
            分类结果列表，每个包含 {id, category_key, category_label, confidence, reasoning, from_cache}
        """
        results = []

        # 先检查缓存
        uncached_papers = []
        for paper in papers:
            paper_hash = self._get_paper_hash(paper['title'], paper['abstract'])
            cached = self._get_cached_result(paper_hash)
            if cached:
                results.append({
                    'id': paper['id'],
                    'category_key': cached['category_key'],
                    'category_label': cached['category_label'],
                    'confidence': cached['confidence'],
                    'reasoning': cached.get('reasoning', ''),
                    'from_cache': True
                })
            else:
                uncached_papers.append(paper)

        # 批量处理未缓存的论文
        for i in range(0, len(uncached_papers), batch_size):
            batch = uncached_papers[i:i + batch_size]
            batch_results = self._call_api_with_retry(batch)

            for paper, result in zip(batch, batch_results):
                paper_hash = self._get_paper_hash(paper['title'], paper['abstract'])
                self._set_cached_result(paper_hash, result)
                results.append({
                    'id': paper['id'],
                    **result,
                    'from_cache': False
                })

        return results

    def _call_api_with_retry(self, papers: List[Dict], max_retries: int = 3) -> List[Dict]:
        """调用API with retry logic"""
        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                return self._call_api(papers)
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[LLM API] Failed after {max_retries} attempts: {e}")
                    # 返回默认结果
                    return [{'category_key': DEFAULT_KEY, 'category_label': DEFAULT_LABEL,
                            'confidence': 0.0, 'reasoning': f'API Error: {str(e)[:100]}'} for _ in papers]
                print(f"[LLM API] Attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(2 ** attempt)

    def _call_api(self, papers: List[Dict]) -> List[Dict]:
        """调用LLM API进行分类"""
        # 构建prompt
        category_defs = get_category_definitions_for_llm()
        papers_text = "\n\n".join([
            f"Paper ID: {p['id']}\nTitle: {p['title']}\nAbstract: {p['abstract'][:500]}..."
            for p in papers
        ])

        prompt = f"""你是一个专业的AI论文分类助手。请根据论文的标题和摘要，将其分类到最合适的类别中。

分类标准：
{category_defs}

待分类论文：
{papers_text}

请以JSON格式返回分类结果，格式如下：
{{
  "classifications": [
    {{
      "id": "paper_id",
      "category_key": "category_key",
      "confidence": 0.0-1.0,
      "reasoning": "简短说明分类理由（1-2句话）"
    }}
  ]
}}

注意：
1. 如果无法确定分类（confidence < 0.6），请使用 category_key: "{DEFAULT_KEY}"
2. 每篇论文只选择一个最合适的类别
3. confidence表示分类置信度（0-1之间）
"""

        # 调用API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        # 解析响应
        content = response.choices[0].message.content
        data = json.loads(content)

        # 提取结果
        results = []
        classifications = data.get('classifications', [])
        for i, paper in enumerate(papers):
            if i < len(classifications):
                cls = classifications[i]
                cat_key = cls.get('category_key', DEFAULT_KEY)
                cat = get_category_by_key(cat_key)
                cat_label = cat.label if cat else DEFAULT_LABEL

                results.append({
                    'category_key': cat_key,
                    'category_label': cat_label,
                    'confidence': cls.get('confidence', 0.5),
                    'reasoning': cls.get('reasoning', '')
                })
            else:
                results.append({
                    'category_key': DEFAULT_KEY,
                    'category_label': DEFAULT_LABEL,
                    'confidence': 0.0,
                    'reasoning': 'No classification returned'
                })

        return results


class HybridClassifier:
    """混合分类器：LLM优先，规则补充"""

    def __init__(self, llm_classifier: Optional[LLMClassifier], rule_classifier: RuleClassifier,
                 confidence_threshold: float = 0.6):
        self.llm = llm_classifier
        self.rule = rule_classifier
        self.threshold = confidence_threshold
        self.use_llm = llm_classifier is not None

    def classify_papers(self, papers: List[Dict], batch_size: int = 10,
                       checkpoint_file: Optional[str] = None, verbose: bool = True) -> List[Dict]:
        """
        分类论文列表，支持断点续传

        Args:
            papers: 论文列表，每个包含 {id, title, abstract}
            batch_size: 批处理大小
            checkpoint_file: checkpoint文件路径
            verbose: 是否输出详细日志

        Returns:
            分类结果列表
        """
        # 加载checkpoint
        processed_ids = set()
        results = []
        if checkpoint_file and os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data.get('results', [])
                processed_ids = {r['id'] for r in results}
            if verbose:
                print(f"[Checkpoint] Loaded {len(processed_ids)} processed papers")

        # 过滤未处理的论文
        unprocessed = [p for p in papers if p['id'] not in processed_ids]
        if verbose and unprocessed:
            print(f"[Classifier] Processing {len(unprocessed)} papers (total {len(papers)})")

        # 分类
        if self.use_llm and unprocessed:
            # 使用LLM分类
            llm_results = self.llm.classify_batch(unprocessed, batch_size=batch_size)

            for paper, llm_result in zip(unprocessed, llm_results):
                if llm_result['confidence'] < self.threshold:
                    # 置信度低，使用规则分类
                    rule_key, rule_label = self.rule.classify_with_key(f"{paper['title']}\n{paper['abstract']}")
                    result = {
                        **paper,  # 保留输入的所有字段
                        'category_key': rule_key,
                        'category_label': rule_label,
                        'confidence': 0.5,
                        'reasoning': f'LLM low confidence ({llm_result["confidence"]:.2f}), used rule',
                        'method': 'rule_fallback'
                    }
                else:
                    result = {
                        **paper,  # 保留输入的所有字段
                        'category_key': llm_result['category_key'],
                        'category_label': llm_result['category_label'],
                        'confidence': llm_result['confidence'],
                        'reasoning': llm_result['reasoning'],
                        'method': 'llm'
                    }
                results.append(result)

                # 保存checkpoint
                if checkpoint_file and len(results) % 10 == 0:
                    self._save_checkpoint(checkpoint_file, results)
        else:
            # 仅使用规则分类
            for paper in unprocessed:
                rule_key, rule_label = self.rule.classify_with_key(f"{paper['title']}\n{paper['abstract']}")
                results.append({
                    **paper,  # 保留输入的所有字段
                    'category_key': rule_key,
                    'category_label': rule_label,
                    'confidence': 1.0,
                    'reasoning': 'Rule-based classification',
                    'method': 'rule'
                })

        # 最终保存checkpoint
        if checkpoint_file:
            self._save_checkpoint(checkpoint_file, results)

        return results

    def _save_checkpoint(self, checkpoint_file: str, results: List[Dict]):
        """保存checkpoint"""
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({'results': results, 'timestamp': time.time()}, f, ensure_ascii=False, indent=2)
