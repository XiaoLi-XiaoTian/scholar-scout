#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件加载模块

支持从config.json读取配置,命令行参数优先级高于配置文件。
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


DEFAULT_CONFIG_PATH = "config.json"


class Config:
    """配置管理类"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置

        Args:
            config_path: 配置文件路径,默认为 config.json
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            print(f"[警告] 配置文件 {self.config_path} 不存在,使用默认配置")
            return self._get_default_config()

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"[配置] 成功加载配置文件: {self.config_path}")
            return config
        except Exception as e:
            print(f"[错误] 加载配置文件失败: {e}")
            print("[配置] 使用默认配置")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "api": {
                "api_base": "https://api.openai.com/v1",
                "api_key": "",
                "model": "gpt-3.5-turbo",
                "max_rpm": 60,
                "timeout": 60
            },
            "data_fetch": {
                "years": [2023, 2024, 2025],
                "conferences": ["ICLR", "ICML"],
                "timeout": 60
            },
            "classification": {
                "use_llm": False,
                "llm_batch_size": 10,
                "llm_confidence_threshold": 0.6,
                "cache_file": "llm_cache.json",
                "checkpoint_file": "classification_checkpoint.json"
            },
            "custom_taxonomy": {
                "topic": "",
                "categories": ""
            },
            "output": {
                "output_dir": "out",
                "topk_trends": 12
            },
            "visualization": {
                "font": "",
                "dpi": 300
            },
            "runtime": {
                "quiet": False,
                "plot_only": False,
                "data_csv": ""
            }
        }

    def get(self, *keys, default=None):
        """
        获取配置值

        Args:
            *keys: 配置键路径,例如 get("api", "api_key")
            default: 默认值

        Returns:
            配置值
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, *keys, value):
        """
        设置配置值

        Args:
            *keys: 配置键路径,例如 set("api", "api_key", value="xxx")
            value: 配置值
        """
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value

    def save(self, path: Optional[str] = None):
        """
        保存配置到文件

        Args:
            path: 保存路径,默认为原配置文件路径
        """
        save_path = path or self.config_path
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            print(f"[配置] 配置已保存到: {save_path}")
        except Exception as e:
            print(f"[错误] 保存配置文件失败: {e}")

    # API配置
    @property
    def api_base(self) -> str:
        return self.get("api", "api_base", default="https://api.openai.com/v1")

    @property
    def api_key(self) -> str:
        return self.get("api", "api_key", default="")

    @property
    def model(self) -> str:
        return self.get("api", "model", default="gpt-3.5-turbo")

    @property
    def max_rpm(self) -> int:
        return self.get("api", "max_rpm", default=60)

    @property
    def api_timeout(self) -> int:
        return self.get("api", "timeout", default=60)

    # 数据抓取配置
    @property
    def years(self) -> list:
        return self.get("data_fetch", "years", default=[2023, 2024, 2025])

    @property
    def conferences(self) -> list:
        return self.get("data_fetch", "conferences", default=["ICLR", "ICML"])

    @property
    def fetch_timeout(self) -> int:
        return self.get("data_fetch", "timeout", default=60)

    # 分类配置
    @property
    def use_llm(self) -> bool:
        return self.get("classification", "use_llm", default=False)

    @property
    def llm_batch_size(self) -> int:
        return self.get("classification", "llm_batch_size", default=10)

    @property
    def llm_confidence_threshold(self) -> float:
        return self.get("classification", "llm_confidence_threshold", default=0.6)

    @property
    def cache_file(self) -> str:
        return self.get("classification", "cache_file", default="llm_cache.json")

    @property
    def checkpoint_file(self) -> str:
        return self.get("classification", "checkpoint_file", default="classification_checkpoint.json")

    # 自定义分类
    @property
    def custom_topic(self) -> str:
        return self.get("custom_taxonomy", "topic", default="")

    @property
    def custom_categories(self) -> str:
        return self.get("custom_taxonomy", "categories", default="")

    # 输出配置
    @property
    def output_dir(self) -> str:
        return self.get("output", "output_dir", default="out")

    @property
    def topk_trends(self) -> int:
        return self.get("output", "topk_trends", default=12)

    # 可视化配置
    @property
    def font(self) -> str:
        return self.get("visualization", "font", default="")

    @property
    def dpi(self) -> int:
        return self.get("visualization", "dpi", default=300)

    # 运行时配置
    @property
    def quiet(self) -> bool:
        return self.get("runtime", "quiet", default=False)

    @property
    def plot_only(self) -> bool:
        return self.get("runtime", "plot_only", default=False)

    @property
    def data_csv(self) -> str:
        return self.get("runtime", "data_csv", default="")

    def __repr__(self):
        """字符串表示"""
        return f"Config(path={self.config_path})"

    def print_summary(self):
        """打印配置摘要"""
        print("\n" + "="*60)
        print("当前配置摘要:")
        print("="*60)
        print(f"API Base:     {self.api_base}")
        print(f"API Key:      {'***已设置***' if self.api_key else '未设置'}")
        print(f"Model:        {self.model}")
        print(f"Years:        {self.years}")
        print(f"Conferences:  {self.conferences}")
        print(f"Use LLM:      {self.use_llm}")
        print(f"Output Dir:   {self.output_dir}")

        if self.custom_topic:
            print(f"Custom Topic: {self.custom_topic}")
        if self.custom_categories:
            print(f"Categories:   {self.custom_categories[:50]}...")

        print("="*60 + "\n")


def load_config(config_path: Optional[str] = None) -> Config:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径,默认为 config.json

    Returns:
        Config对象
    """
    return Config(config_path)


if __name__ == "__main__":
    # 测试配置加载
    config = load_config()
    config.print_summary()

    print("\n测试配置访问:")
    print(f"API Base: {config.api_base}")
    print(f"API Key: {config.api_key}")
    print(f"Model: {config.model}")
    print(f"Years: {config.years}")
    print(f"Output Dir: {config.output_dir}")
