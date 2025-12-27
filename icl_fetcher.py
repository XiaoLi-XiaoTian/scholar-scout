#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenReview数据抓取模块

提供从OpenReview API抓取ICLR/ICML会议论文的功能。
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple, Union

import requests


def safe_json(resp: requests.Response) -> Union[dict, list]:
    """安全解析JSON响应"""
    try:
        return resp.json()
    except Exception:
        snippet = resp.text[:400].replace("\n", " ")
        raise RuntimeError(f"Non-JSON response (status={resp.status_code}). Head: {snippet}")


def extract_notes(payload: Union[dict, list]) -> List[dict]:
    """从API响应中提取notes列表"""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("notes", "data", "results"):
            if k in payload and isinstance(payload[k], list):
                return payload[k]
    return []


def http_get(baseurl: str, path: str, params: Dict, timeout: int, max_retries: int = 3) -> Union[dict, list]:
    """
    执行HTTP GET请求，带重试机制

    Args:
        baseurl: API基础URL
        path: 请求路径
        params: 查询参数
        timeout: 超时时间
        max_retries: 最大重试次数

    Returns:
        JSON响应（dict或list）
    """
    url = f"{baseurl.rstrip('/')}{path}"
    headers = {
        "User-Agent": "ICL-survey-bot/4.2 (requests)",
        "Accept": "application/json",
        "Connection": "close"  # 避免连接复用导致的问题
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            # 每次重试增加超时时间
            actual_timeout = timeout * (attempt + 1)
            r = requests.get(url, params=params, headers=headers, timeout=actual_timeout)
            r.raise_for_status()
            return safe_json(r)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            last_error = e
            if attempt < max_retries - 1:
                import time
                wait_time = (attempt + 1) * 2  # 指数退避
                time.sleep(wait_time)
            continue
        except Exception as e:
            # 其他错误直接抛出
            raise

    # 所有重试都失败
    raise last_error if last_error else RuntimeError("Request failed")


def fetch_notes_paginated(baseurl: str, invitation: str, extra_params: Dict,
                         limit: int = 1000, timeout: int = 60, verbose: bool = True) -> List[dict]:
    """
    分页抓取notes

    Args:
        baseurl: API基础URL
        invitation: 邀请ID
        extra_params: 额外的查询参数
        limit: 每页数量
        timeout: 超时时间（秒）
        verbose: 是否输出详细日志

    Returns:
        所有notes的列表
    """
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
    """
    从note中提取标准化的标题和摘要

    处理API v1和v2的格式差异

    Args:
        note: OpenReview note字典

    Returns:
        (title, abstract)元组
    """
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
    """
    生成可能的invitation候选列表

    Args:
        conf: 会议名称（ICLR/ICML）
        year: 年份

    Returns:
        invitation字符串列表
    """
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
    """
    尝试抓取会议的accepted papers

    Args:
        conf: 会议名称
        year: 年份
        verbose: 是否输出详细日志
        timeout: 超时时间

    Returns:
        (baseurl, invitation, total_submissions, accepted_count, accepted_notes)

    Raises:
        RuntimeError: 如果无法抓取数据
    """
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
                        venue_field = content.get("venue", "")

                        # 提取venue字符串值（处理dict格式）
                        if isinstance(venue_field, dict):
                            venue_str = str(venue_field.get("value", ""))
                        else:
                            venue_str = str(venue_field)

                        # 检查是否为accepted paper
                        if (note_venueid == venueid or
                            (venue_str and ("accept" in venue_str.lower() or f"{conf} {year}" in venue_str))):
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


# ICL关键词正则表达式
ICL_TERMS = [
    r"\bin[- ]context\b",
    r"\bin[- ]context learning\b",
    r"\bICL\b",
    r"\b(in[- ]context) (reason|learn|adapt|generaliz)\w*",
    r"\bmany[- ]shot\b",
    r"\bfew[- ]shot\b",
]
ICL_REGEX = re.compile("|".join(ICL_TERMS), flags=re.IGNORECASE)


def is_icl_related(title: str, abstract: str) -> bool:
    """
    判断论文是否与ICL相关

    Args:
        title: 论文标题
        abstract: 论文摘要

    Returns:
        是否ICL相关
    """
    text = f"{title}\n{abstract}"
    return ICL_REGEX.search(text) is not None
