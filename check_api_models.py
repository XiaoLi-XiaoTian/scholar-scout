#!/usr/bin/env python3
"""检查 API 支持的模型列表（从配置文件读取 API 信息）"""

import requests
import sys

try:
    from config_loader import load_config
    config = load_config()

    API_KEY = config.api_key
    API_BASE = config.api_base

    if not API_KEY:
        print("\n❌ 错误: 未配置 API 密钥")
        print("请编辑 config.json 文件，在 'api.api_key' 字段中填入您的 API 密钥")
        sys.exit(1)

except ImportError:
    print("❌ 错误: 无法导入配置模块")
    print("请确保 config_loader.py 存在且 config.json 已正确配置")
    sys.exit(1)

# 尝试获取模型列表
url = f"{API_BASE}/models"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

print(f"正在查询: {url}")
print()

try:
    response = requests.get(url, headers=headers, timeout=10)
    print(f"状态码: {response.status_code}")
    print()

    if response.status_code == 200:
        data = response.json()
        print("返回数据:")
        import json
        print(json.dumps(data, indent=2, ensure_ascii=False))

        # 尝试提取模型列表
        if 'data' in data:
            print("\n可用模型:")
            for model in data['data']:
                if isinstance(model, dict) and 'id' in model:
                    print(f"  - {model['id']}")
                else:
                    print(f"  - {model}")
    else:
        print("响应内容:")
        print(response.text)

except Exception as e:
    print(f"错误: {e}")

# 尝试测试聊天 API
print("\n" + "=" * 70)
print("尝试测试聊天 API（使用几个常见模型名称）")
print("=" * 70)

test_models = [
    "gpt-3.5-turbo",
    "gpt-4",
    "claude-3-sonnet",
    "claude-3-5-sonnet",
    "claude-sonnet-3.5",
    "anthropic/claude-3-5-sonnet",
]

for model_name in test_models:
    url = f"{API_BASE}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            print(f"✓ {model_name:40s} - 可用")
            break
        else:
            error_msg = response.json().get('error', response.text)[:100]
            print(f"✗ {model_name:40s} - {error_msg}")
    except Exception as e:
        print(f"✗ {model_name:40s} - {str(e)[:100]}")
