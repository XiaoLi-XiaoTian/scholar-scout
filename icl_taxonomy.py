#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICL论文分类体系定义

基于ICL研究核心问题的分类体系，包含9个核心类别。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Category:
    """分类类别定义"""
    key: str
    label: str
    description: str  # 给LLM的详细描述
    patterns: Tuple[str, ...]  # 规则匹配模式


# 1. Prompt工程与优化
PROMPT_ENGINEERING = Category(
    "prompt_eng",
    "📚 Prompt工程与优化",
    "研究如何设计和优化prompt来提升ICL效果，包括：示例选择、示例排序、prompt模板设计、指令优化等。",
    (
        r"\bprompt (engineering|design|optimization|learning|tuning)\b",
        r"\bexample selection\b|\bdemonstration selection\b|\bexemplar selection\b",
        r"\bselect(ing)? (examples|demonstrations|exemplars)\b",
        r"\bprompt ordering\b|\border(ing)? demonstrations\b|\bpermutation\b",
        r"\bcompose(d)? demonstrations\b|\bstructure(d)? prompt\b",
        r"\bretrieve demonstrations\b|\bfew[- ]shot (example|prompt)\b",
        r"\btemplate\b.*\b(design|optimization)\b|\binstruction (following|tuning)\b",
    ),
)

# 2. 推理与思维链
REASONING_COT = Category(
    "reasoning_cot",
    "🧠 推理与思维链",
    "研究利用ICL进行复杂推理的方法，包括：思维链(CoT)、多步推理、自洽性、思维树等推理增强技术。",
    (
        r"\bchain[- ]of[- ]thought\b|\bCoT\b|\bscratchpad\b",
        r"\bself[- ]consistency\b|\btree[- ]of[- ]thought\b|\bgraph[- ]of[- ]thought\b",
        r"\bmultistep\b|\bmulti[- ]step\b|\bmultiple step\b",
        r"\breason(er|ing)\b.*\b(trace|path|step|chain|process)\b",
        r"\bdeliberat(e|ion)\b|\bthought (generation|process)\b",
        r"\bmany[- ]shot\b|\bmany[- ]step\b",
        r"\bintermediate (reasoning|step|output)\b|\bstep[- ]by[- ]step\b",
        r"\bcomplex reasoning\b|\blogical reasoning\b|\bmathematical reasoning\b",
    ),
)

# 3. 机理理解与可解释性
MECHANISM_THEORY = Category(
    "mechanism_theory",
    "🔬 机理理解与可解释性",
    "研究ICL的工作原理和理论基础，包括：机制分析、理论证明、可解释性研究、注意力分析、诱导头、电路分析等。",
    (
        r"\btheor(y|etical)\b.*\b(ICL|in[- ]context)\b",
        r"\bmechanis\w*\b.*\b(ICL|in[- ]context)\b",
        r"\binduction head(s)?\b|\bcircuit(s)?\b.*\b(analysis|discover)\b",
        r"\binterpretab\w*|\bexplainab\w*|\bunderstanding\b.*\b(ICL|in[- ]context)\b",
        r"\bassociative memory\b|\bhopfield\b|\bmeta[- ]learn\w*",
        r"\bimplicit (learning|gradient)\b|\bin[- ]weights\b",
        r"\bprovab\w*|\bconvergence\b|\blearning dynamics\b",
        r"\battribution\b|\bprobe\b|\bdiagnostic\b.*\bICL\b",
    ),
)

# 4. 模型训练与架构
MODEL_TRAINING = Category(
    "model_training",
    "🏗️ 模型训练与架构",
    "研究如何通过模型训练和架构设计来增强ICL能力，包括：预训练方法、架构变体、注意力机制、位置编码、模型缩放等。",
    (
        r"\bpretrain\w*\b|\bfine[- ]tun\w*\b|\btraining\b",
        r"\barchitecture\b|\bmodel design\b|\bneural architecture\b",
        r"\bstate space model\b|\bxLSTM\b|\bmamba\b|\bretention\b",
        r"\bsequence model(ing)?\b|\bmixture of experts\b|\bMoE\b",
        r"\btransformer (variant|architecture|model)\b",
        r"\battention (mechanism|variant|pattern|head)\b",
        r"\bposition(al)? (encoding|embedding|interpolation)\b",
        r"\blayer (normalization|norm)\b|\bactivation function\b",
        r"\bmodel (scaling|size|capacity|parameter)\b",
        r"\bbackbone\b|\bfoundation model\b|\blarge language model\b.*\barchitecture\b",
    ),
)

# 5. 效率优化
EFFICIENCY = Category(
    "efficiency",
    "⚡ 效率优化",
    "研究如何提升ICL的计算效率，包括：上下文压缩、KV缓存优化、高效注意力机制、长度外推等。",
    (
        r"\bcontext compression\b|\bprompt compression\b",
        r"\bcompress(ion|ing)?\b.*\b(ICL|context|prompt)\b",
        r"\bdistill(at|ation)\w*\b.*\b(ICL|context|in[- ]context)\b",
        r"\b(in[- ]context )?autoencoder\b|\bcontext distillation\b",
        r"\bkv cache\b|\bkey[- ]value cache\b|\bcache\b.*\boptimization\b",
        r"\bprefill\b|\bthroughput\b|\blatency\b.*\b(optimization|reduction)\b",
        r"\befficient (attention|inference)\b|\blinear attention\b|\bflash[- ]?attention\b",
        r"\blength generaliz\w*\b|\blength extrapolat\w*\b",
        r"\btrain short.*infer long\b|\blong[- ]short\b",
        r"\bcontext length\b.*\b(generaliz\w*|extrapolat\w*|extension)\b",
        r"\bpositional extrapolat\w*\b|\bRoPE\b.*\bscaling\b",
    ),
)

# 6. 评测基准与数据集
EVALUATION = Category(
    "evaluation",
    "📊 评测基准与数据集",
    "研究ICL的评测方法和基准数据集，包括：基准构建、评测方法、消融实验、诊断工具、综述等。",
    (
        r"\bbenchmark\b.*\b(ICL|in[- ]context|few[- ]shot)\b",
        r"\b(evaluation|testbed|dataset)\b.*\b(ICL|in[- ]context|few[- ]shot)\b",
        r"\bnew (benchmark|dataset|task)\b",
        r"\bmeasure\b|\bmetric\b.*\b(ICL|in[- ]context)\b",
        r"\bablation (study|experiment)\b|\bempirical (study|analysis)\b",
        r"\bsurvey\b|\bliterature review\b",
    ),
)

# 7. 应用：Agent与工具使用
APPLICATION_AGENT = Category(
    "application_agent",
    "🤖 应用：Agent与工具使用",
    "研究ICL在Agent和工具使用场景中的应用，包括：规划、工具调用、函数调用、动作序列、轨迹学习等。",
    (
        r"\bagent(s)?\b.*\b(ICL|in[- ]context|few[- ]shot)\b",
        r"\bplanning\b.*\b(agent|ICL|in[- ]context)\b",
        r"\btool (use|usage|calling|learning)\b",
        r"\bfunction calling\b|\bAPI (call|usage)\b",
        r"\baction (sequence|selection)\b|\btrajectory\b",
        r"\breasoning and acting\b|\bReAct\b",
        r"\baudited reasoning\b|\bemergent abilit\w*\b",
    ),
)

# 8. 可靠性与安全
RELIABILITY_SAFETY = Category(
    "reliability_safety",
    "🛡️ 可靠性与安全",
    "研究ICL的可靠性和安全性问题，包括：校准、不确定性估计、鲁棒性、隐私、遗忘、攻击防御、幻觉等。",
    (
        r"\bcalibrat\w*|\buncertaint\w*|\bconfidence (estimation|calibration)\b",
        r"\breliabilit\w*\b|\brobust\w*\b.*\b(ICL|in[- ]context)\b",
        r"\bselective prediction\b|\babstain\b|\breject option\b",
        r"\bunlearning\b|\bforget(ting)?\b|\bmachine unlearning\b",
        r"\bprivacy\b.*\b(ICL|in[- ]context)\b|\bdata leakage\b",
        r"\battack\b.*\b(ICL|prompt)\b|\bbackdoor\b|\badversarial\b",
        r"\bjailbreak\b|\bprompt injection\b",
        r"\bwatermark\b|\bsafety\b|\brefusal\b",
        r"\bhallucination\b|\bfaithful\w*\b",
    ),
)

# 9. 特定技术方法
SPECIFIC_METHODS = Category(
    "specific_methods",
    "🎯 特定技术方法",
    "特定的ICL技术方法，包括：kNN-ICL、非参数方法、从错误学习、原则归纳、自我修正、对比学习等。",
    (
        r"\bnearest neighbor\b|\b(k[- ]?nn|kNN)\b.*\b(ICL|in[- ]context)\b",
        r"\bnonparametric\b.*\b(ICL|learning)\b|\bprototype(s)?\b",
        r"\bcalibration[- ]free\b|\bembedding[- ]based inference\b",
        r"\bvector database\b|\bretrieval[- ]augmented\b",
        r"\bmistake(s)?\b.*\b(learning|correction)\b",
        r"\berror(s)?\b.*\b(analysis|learning|feedback)\b",
        r"\bcounterexample(s)?\b|\bfrom mistakes\b",
        r"\bprinciple learning\b|\brule induction\b",
        r"\bself[- ]correction\b|\bself[- ]refinement\b|\bself[- ]improvement\b",
        r"\bcontrastive\b.*\b(ICL|learning)\b|\bsymbol tuning\b",
    ),
)

# 分类优先级（按优先级排序）
CATEGORY_PRIORITY: List[Category] = [
    EVALUATION,           # 优先识别评测类（避免被其他类误判）
    APPLICATION_AGENT,    # Agent应用（特征明显）
    REASONING_COT,        # 推理与CoT（特征明显）
    PROMPT_ENGINEERING,   # Prompt工程
    SPECIFIC_METHODS,     # 特定方法（避免被大类吸收）
    EFFICIENCY,           # 效率优化
    RELIABILITY_SAFETY,   # 可靠性与安全
    MECHANISM_THEORY,     # 机理理论
    MODEL_TRAINING,       # 模型训练（最后，避免过度匹配）
]

DEFAULT_LABEL = "🧺 其他/未归类"
DEFAULT_KEY = "other"


# =============================================================================
# 导出接口
# =============================================================================

def get_all_categories() -> List[Category]:
    """获取所有类别定义"""
    return CATEGORY_PRIORITY


def get_category_by_key(key: str) -> Category | None:
    """根据key获取类别"""
    for cat in CATEGORY_PRIORITY:
        if cat.key == key:
            return cat
    return None


def get_category_definitions_for_llm() -> str:
    """生成给LLM的分类定义文本"""
    lines = []
    for i, cat in enumerate(CATEGORY_PRIORITY, 1):
        lines.append(f"{i}. **{cat.label}** (key: {cat.key})")
        lines.append(f"   {cat.description}")
        lines.append("")

    # 添加"其他/未归类"类别
    lines.append(f"{len(CATEGORY_PRIORITY) + 1}. **{DEFAULT_LABEL}** (key: {DEFAULT_KEY})")
    lines.append("   无法归入以上任何类别的论文")

    return "\n".join(lines)


def get_category_map_for_llm() -> dict:
    """生成给LLM的类别映射（用于JSON格式）"""
    categories = []
    for cat in CATEGORY_PRIORITY:
        categories.append({
            "key": cat.key,
            "label": cat.label,
            "description": cat.description
        })

    # 添加"其他/未归类"类别
    categories.append({
        "key": DEFAULT_KEY,
        "label": DEFAULT_LABEL,
        "description": "无法归入以上任何类别的论文"
    })

    return categories


class RuleClassifier:
    """基于规则的分类器"""

    def __init__(self):
        self.categories = CATEGORY_PRIORITY
        self.default_label = DEFAULT_LABEL

    def classify(self, text: str) -> str:
        """
        基于规则分类文本

        Args:
            text: 要分类的文本（通常是 title + abstract）

        Returns:
            分类标签
        """
        for cat in self.categories:
            for pattern in cat.patterns:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    return cat.label
        return self.default_label

    def classify_with_key(self, text: str) -> tuple[str, str]:
        """
        基于规则分类文本，返回 (key, label)

        Args:
            text: 要分类的文本

        Returns:
            (category_key, category_label)
        """
        for cat in self.categories:
            for pattern in cat.patterns:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    return cat.key, cat.label
        return DEFAULT_KEY, self.default_label
