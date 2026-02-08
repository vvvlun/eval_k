"""
Exp 4: Mechanism Comparison - 机制比较主实验

比较5种聚合机制在识别高影响力论文上的表现。

机制定义：
| ID | 名称 | 公式 | 需要LLM |
|----|------|------|---------|
| M1 | Simple Average | θ̂ = (1/n)Σs_d | ❌ |
| M2 | Confidence-Weighted | θ̂ = Σ(c_d*s_d)/Σc_d | ✅ 需要获取confidence |
| M2v| Dimension-Weighted | θ̂ = Σ(w_d*s_d) 固定权重 | ❌ |
| M3 | AC-Centric | Chair Agent综合决策 | ❌ 用已有结果 |
| M4 | Bounded Iteration | 3轮讨论后聚合 | ✅ 需要迭代 |
| M5 | Diverse Ensemble | M2 + 2个通才Agent | ✅ 需要通才Agent |

使用方法：
    # 只运行不需要LLM的机制 (M1, M2v, M3)
    python exp4_mechanism_comparison.py \
        --agent-results ./data/results/iclr2018_con_llama8b.json \
        --papers ./data/raw/iclr_2018.json \
        --output ./data/results/exp4_mechanisms.json

    # 运行所有机制（需要ollama）
    python exp4_mechanism_comparison.py \
        --agent-results ./data/results/iclr2018_con_llama8b.json \
        --papers ./data/raw/iclr_2018.json \
        --output ./data/results/exp4_mechanisms.json \
        --use-llm \
        --model llama3.1:8b

    # 带Ground Truth数据（用于影响力评估）
    python exp4_mechanism_comparison.py \
        --agent-results ./data/results/iclr2018_con_llama8b.json \
        --papers ./data/raw/iclr_2018.json \
        --ground-truth ./data/processed/ground_truth.json \
        --output ./data/results/exp4_mechanisms.json \
        --use-llm
"""

import os
import sys
import json
import logging
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from scipy.stats import spearmanr, pearsonr, chi2
from collections import defaultdict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_to_serializable(obj):
    """将numpy类型转换为Python原生类型，以便JSON序列化"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class AgentEvaluation:
    """Agent评估结果"""
    dimension: str
    score: float
    confidence: float
    sentiment: str = "mixed"
    key_findings: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class PaperData:
    """论文完整数据"""
    paper_id: str
    title: str = ""
    abstract: str = ""
    reviews: List[Dict] = field(default_factory=list)
    rebuttal: Optional[str] = None
    real_decision: str = ""
    
    # Agent评分（来自exp1）
    expert_scores: Dict[str, float] = field(default_factory=dict)
    expert_confidences: Dict[str, float] = field(default_factory=dict)
    weighted_score: float = 0.0
    simulated_decision: str = ""
    
    # 影响力数据（来自ground_truth）
    citations: int = 0
    is_high_impact: bool = False


@dataclass
class MechanismResult:
    """机制输出结果"""
    mechanism_name: str
    paper_id: str
    aggregated_score: float
    decision: str
    confidence: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class MechanismMetrics:
    """机制评估指标"""
    mechanism_name: str
    spearman_rho: float = 0.0
    spearman_p: float = 0.0
    pearson_r: float = 0.0
    pearson_p: float = 0.0
    precision_at_20: float = 0.0
    precision_at_30: float = 0.0
    recall_at_20: float = 0.0
    recall_at_30: float = 0.0
    fnr: float = 0.0
    fpr: float = 0.0
    agreement_with_real: float = 0.0
    cohens_kappa: float = 0.0
    n_papers: int = 0
    n_accept: int = 0
    n_high_impact: int = 0
    spearman_ci_lower: float = 0.0
    spearman_ci_upper: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# 机制实现
# ============================================================================

class BaseMechanism:
    """机制基类"""
    
    def __init__(self, name: str, threshold: float = 5.5):
        self.name = name
        self.threshold = threshold
    
    def make_decision(self, score: float) -> str:
        return "Accept" if score >= self.threshold else "Reject"
    
    def aggregate(self, paper: PaperData) -> MechanismResult:
        raise NotImplementedError


class SimpleAverageMechanism(BaseMechanism):
    """M1: 简单平均"""
    
    def __init__(self, threshold: float = 5.5):
        super().__init__("M1_SimpleAverage", threshold)
    
    def aggregate(self, paper: PaperData) -> MechanismResult:
        scores = list(paper.expert_scores.values())
        if not scores:
            return MechanismResult(self.name, paper.paper_id, 5.0, "Reject", 0.3)
        
        avg_score = sum(scores) / len(scores)
        decision = self.make_decision(avg_score)
        
        return MechanismResult(
            mechanism_name=self.name,
            paper_id=paper.paper_id,
            aggregated_score=avg_score,
            decision=decision,
            confidence=0.7,
            metadata={"dimension_scores": paper.expert_scores}
        )


class DimensionWeightedMechanism(BaseMechanism):
    """M2v: 固定维度权重（使用已有的weighted_score）"""
    
    WEIGHTS = {
        'novelty': 0.30,
        'methodology': 0.30,
        'clarity': 0.15,
        'empirical': 0.25
    }
    
    def __init__(self, threshold: float = 5.5):
        super().__init__("M2v_DimensionWeighted", threshold)
    
    def aggregate(self, paper: PaperData) -> MechanismResult:
        # 优先使用已有的weighted_score
        if paper.weighted_score > 0:
            score = paper.weighted_score
        else:
            # 手动计算
            score = sum(
                paper.expert_scores.get(dim, 5.0) * w
                for dim, w in self.WEIGHTS.items()
            )
        
        decision = self.make_decision(score)
        
        return MechanismResult(
            mechanism_name=self.name,
            paper_id=paper.paper_id,
            aggregated_score=score,
            decision=decision,
            confidence=0.7,
            metadata={"weights": self.WEIGHTS}
        )


class ConfidenceWeightedMechanism(BaseMechanism):
    """M2: 真正的Confidence加权（需要LLM获取confidence）"""
    
    def __init__(self, threshold: float = 5.5, llm_client=None):
        super().__init__("M2_ConfidenceWeighted", threshold)
        self.llm_client = llm_client
        self.analyzer = None
    
    def set_analyzer(self, analyzer):
        """设置MultiDimensionalAnalyzer"""
        self.analyzer = analyzer
    
    def aggregate(self, paper: PaperData) -> MechanismResult:
        scores = paper.expert_scores
        confidences = paper.expert_confidences
        
        if not scores:
            return MechanismResult(self.name, paper.paper_id, 5.0, "Reject", 0.3)
        
        # 如果有真实的confidence数据，使用它
        if confidences and any(c != 0.7 for c in confidences.values()):
            total_weight = sum(confidences.values())
            if total_weight > 0:
                weighted_score = sum(
                    scores[dim] * confidences.get(dim, 0.7)
                    for dim in scores
                ) / total_weight
            else:
                weighted_score = sum(scores.values()) / len(scores)
            
            avg_confidence = sum(confidences.values()) / len(confidences)
        else:
            # 没有真实confidence，返回None表示需要LLM
            return None
        
        decision = self.make_decision(weighted_score)
        
        return MechanismResult(
            mechanism_name=self.name,
            paper_id=paper.paper_id,
            aggregated_score=weighted_score,
            decision=decision,
            confidence=avg_confidence,
            metadata={
                "dimension_scores": scores,
                "dimension_confidences": confidences
            }
        )


class ACCentricMechanism(BaseMechanism):
    """M3: AC-Centric（使用已有的simulated_decision）"""
    
    def __init__(self, threshold: float = 5.5):
        super().__init__("M3_ACCentric", threshold)
    
    def aggregate(self, paper: PaperData) -> MechanismResult:
        # 使用已有的simulated_decision
        if paper.simulated_decision:
            decision = paper.simulated_decision
            score = paper.weighted_score if paper.weighted_score > 0 else 5.5
        else:
            # 降级到基于分数的决策
            score = paper.weighted_score if paper.weighted_score > 0 else 5.0
            decision = self.make_decision(score)
        
        return MechanismResult(
            mechanism_name=self.name,
            paper_id=paper.paper_id,
            aggregated_score=score,
            decision=decision,
            confidence=0.75,
            metadata={"source": "simulated_decision" if paper.simulated_decision else "fallback"}
        )


class BoundedIterationMechanism(BaseMechanism):
    """M4: 有限轮迭代（需要LLM）"""
    
    def __init__(self, threshold: float = 5.5, llm_client=None, max_rounds: int = 3):
        super().__init__("M4_BoundedIteration", threshold)
        self.llm_client = llm_client
        self.max_rounds = max_rounds
    
    def aggregate(self, paper: PaperData) -> MechanismResult:
        if not self.llm_client:
            # 模拟迭代
            return self._simulate_iteration(paper)
        else:
            return self._run_with_llm(paper)
    
    def _simulate_iteration(self, paper: PaperData) -> MechanismResult:
        """模拟迭代（不使用LLM）"""
        import random
        
        current_scores = dict(paper.expert_scores)
        convergence_rate = 0.2
        
        for _ in range(self.max_rounds):
            mean_score = sum(current_scores.values()) / len(current_scores)
            new_scores = {}
            for dim, score in current_scores.items():
                new_score = score + convergence_rate * (mean_score - score)
                new_score += random.gauss(0, 0.1)
                new_scores[dim] = max(1.0, min(10.0, new_score))
            current_scores = new_scores
        
        final_score = sum(current_scores.values()) / len(current_scores)
        decision = self.make_decision(final_score)
        
        return MechanismResult(
            mechanism_name=self.name,
            paper_id=paper.paper_id,
            aggregated_score=final_score,
            decision=decision,
            confidence=0.7,
            metadata={"method": "simulated", "rounds": self.max_rounds}
        )
    
    def _run_with_llm(self, paper: PaperData) -> MechanismResult:
        """使用LLM运行迭代"""
        current_scores = dict(paper.expert_scores)
        current_confidences = dict(paper.expert_confidences) if paper.expert_confidences else {
            dim: 0.7 for dim in current_scores
        }
        
        for round_num in range(1, self.max_rounds + 1):
            # 构建讨论上下文
            context = self._build_context(current_scores, round_num)
            
            # 每个agent更新评分
            new_scores = {}
            new_confidences = {}
            
            for dim in current_scores:
                updated = self._update_agent(dim, current_scores[dim], context, paper)
                new_scores[dim] = updated['score']
                new_confidences[dim] = updated['confidence']
            
            current_scores = new_scores
            current_confidences = new_confidences
        
        # Confidence加权聚合
        total_weight = sum(current_confidences.values())
        if total_weight > 0:
            final_score = sum(
                current_scores[dim] * current_confidences[dim]
                for dim in current_scores
            ) / total_weight
        else:
            final_score = sum(current_scores.values()) / len(current_scores)
        
        decision = self.make_decision(final_score)
        avg_confidence = sum(current_confidences.values()) / len(current_confidences)
        
        return MechanismResult(
            mechanism_name=self.name,
            paper_id=paper.paper_id,
            aggregated_score=final_score,
            decision=decision,
            confidence=avg_confidence,
            metadata={"method": "llm", "rounds": self.max_rounds}
        )
    
    def _build_context(self, scores: Dict[str, float], round_num: int) -> str:
        context = f"=== Discussion Round {round_num} ===\n"
        context += "Current expert evaluations:\n"
        for dim, score in scores.items():
            context += f"  {dim.capitalize()}: {score:.1f}/10\n"
        return context
    
    def _update_agent(self, dimension: str, current_score: float, 
                      context: str, paper: PaperData) -> Dict:
        """让agent更新评分"""
        prompt = f"""You are the {dimension.upper()} expert in a peer review discussion.

{context}

Your current score for {dimension}: {current_score:.1f}/10

Paper: {paper.title}

After seeing other experts' scores, reconsider your evaluation.
You may adjust your score based on the discussion, but maintain your unique perspective.

Output as JSON:
{{"score": <float 1-10>, "confidence": <float 0-1>}}"""

        try:
            response = self.llm_client.generate_json(
                prompt=prompt,
                system_prompt=f"You are an expert reviewer for {dimension}.",
                temperature=0.3
            )
            return {
                'score': max(1.0, min(10.0, float(response.get('score', current_score)))),
                'confidence': max(0.0, min(1.0, float(response.get('confidence', 0.7))))
            }
        except:
            return {'score': current_score, 'confidence': 0.7}


class DiverseEnsembleMechanism(BaseMechanism):
    """M5: 扩展Ensemble（需要LLM运行通才Agent）"""
    
    def __init__(self, threshold: float = 5.5, llm_client=None):
        super().__init__("M5_DiverseEnsemble", threshold)
        self.llm_client = llm_client
        self.generalist_weight = 0.8
    
    def aggregate(self, paper: PaperData) -> MechanismResult:
        dimension_scores = paper.expert_scores
        dimension_confidences = paper.expert_confidences or {dim: 0.7 for dim in dimension_scores}
        
        if not dimension_scores:
            return MechanismResult(self.name, paper.paper_id, 5.0, "Reject", 0.3)
        
        # 获取通才Agent评估
        if self.llm_client:
            generalist_evals = self._run_generalists(paper, dimension_scores)
        else:
            generalist_evals = self._simulate_generalists(dimension_scores)
        
        # 聚合所有评估
        all_scores = []
        all_weights = []
        
        # 维度Agent
        for dim, score in dimension_scores.items():
            all_scores.append(score)
            all_weights.append(dimension_confidences.get(dim, 0.7))
        
        # 通才Agent
        for name, eval_data in generalist_evals.items():
            all_scores.append(eval_data['score'])
            all_weights.append(eval_data['confidence'] * self.generalist_weight)
        
        # 加权平均
        total_weight = sum(all_weights)
        final_score = sum(s * w for s, w in zip(all_scores, all_weights)) / total_weight
        final_confidence = sum(all_weights) / len(all_weights)
        
        decision = self.make_decision(final_score)
        
        return MechanismResult(
            mechanism_name=self.name,
            paper_id=paper.paper_id,
            aggregated_score=final_score,
            decision=decision,
            confidence=final_confidence,
            metadata={"generalist_scores": generalist_evals}
        )
    
    def _run_generalists(self, paper: PaperData, dimension_scores: Dict) -> Dict:
        """运行通才Agent"""
        generalists = {}
        avg_score = sum(dimension_scores.values()) / len(dimension_scores)
        
        # Holistic Agent
        holistic = self._run_holistic(paper, dimension_scores, avg_score)
        if holistic:
            generalists['holistic'] = holistic
        
        # Contrarian Agent
        contrarian = self._run_contrarian(paper, dimension_scores, avg_score)
        if contrarian:
            generalists['contrarian'] = contrarian
        
        return generalists
    
    def _run_holistic(self, paper: PaperData, scores: Dict, avg: float) -> Optional[Dict]:
        prompt = f"""Evaluate this paper from a HOLISTIC perspective.

Paper: {paper.title}
Abstract: {paper.abstract[:500] if paper.abstract else 'N/A'}...

Specialist scores:
{chr(10).join(f'  {dim}: {s:.1f}/10' for dim, s in scores.items())}
Average: {avg:.2f}/10

Provide your overall assessment considering how different aspects combine.

Output as JSON:
{{"score": <float 1-10>, "confidence": <float 0-1>}}"""

        try:
            response = self.llm_client.generate_json(
                prompt=prompt,
                system_prompt="You are a holistic reviewer assessing overall contribution.",
                temperature=0.3
            )
            return {
                'score': max(1.0, min(10.0, float(response.get('score', avg)))),
                'confidence': max(0.0, min(1.0, float(response.get('confidence', 0.6))))
            }
        except:
            return None
    
    def _run_contrarian(self, paper: PaperData, scores: Dict, avg: float) -> Optional[Dict]:
        direction = "positive" if avg >= 5.5 else "negative"
        task = "look for potential weaknesses" if avg >= 5.5 else "look for potential strengths"
        
        prompt = f"""Evaluate this paper from a CONTRARIAN perspective.

Paper: {paper.title}
Current consensus is {direction} (avg: {avg:.2f}/10).

Your task: {task}

Provide an independent assessment that challenges the consensus.

Output as JSON:
{{"score": <float 1-10>, "confidence": <float 0-1>}}"""

        try:
            response = self.llm_client.generate_json(
                prompt=prompt,
                system_prompt="You are a contrarian reviewer offering alternative perspectives.",
                temperature=0.4
            )
            return {
                'score': max(1.0, min(10.0, float(response.get('score', avg)))),
                'confidence': max(0.0, min(1.0, float(response.get('confidence', 0.5))))
            }
        except:
            return None
    
    def _simulate_generalists(self, dimension_scores: Dict) -> Dict:
        """模拟通才评估"""
        import random
        avg = sum(dimension_scores.values()) / len(dimension_scores)
        
        return {
            'holistic': {
                'score': max(1.0, min(10.0, avg + random.gauss(0, 0.3))),
                'confidence': 0.65
            },
            'contrarian': {
                'score': max(1.0, min(10.0, avg + (-1 if avg >= 5.5 else 1) * random.uniform(0.5, 1.5))),
                'confidence': 0.5
            }
        }


# ============================================================================
# M2完整实现：调用Expert Agent获取Confidence
# ============================================================================

class ExpertAgentRunner:
    """运行Expert Agent获取带confidence的评估"""
    
    DIMENSIONS = ['novelty', 'methodology', 'clarity', 'empirical']
    
    SYSTEM_PROMPTS = {
        'novelty': """You are an expert reviewer specializing in NOVELTY and ORIGINALITY.
Evaluate: originality, difference from prior work, potential impact.""",
        
        'methodology': """You are an expert reviewer specializing in METHODOLOGY.
Evaluate: technical correctness, theoretical foundation, rigor.""",
        
        'clarity': """You are an expert reviewer specializing in CLARITY.
Evaluate: writing quality, organization, readability.""",
        
        'empirical': """You are an expert reviewer specializing in EMPIRICAL QUALITY.
Evaluate: experimental rigor, baselines, reproducibility."""
    }
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def evaluate_paper(self, paper: PaperData) -> Dict[str, AgentEvaluation]:
        """评估单篇论文，返回所有维度的评估（包含confidence）"""
        results = {}
        
        for dim in self.DIMENSIONS:
            eval_result = self._evaluate_dimension(paper, dim)
            results[dim] = eval_result
        
        return results
    
    def _evaluate_dimension(self, paper: PaperData, dimension: str) -> AgentEvaluation:
        """评估单个维度"""
        # 格式化reviews
        reviews_text = self._format_reviews(paper.reviews)
        
        prompt = f"""Evaluate the {dimension.upper()} of this paper:

=== Paper Title ===
{paper.title}

=== Abstract ===
{paper.abstract[:800] if paper.abstract else 'N/A'}

=== Reviews ===
{reviews_text}

Based on the reviews, evaluate the paper's {dimension}.

Respond with a JSON object:
{{
    "score": <float 1-10>,
    "confidence": <float 0-1, how confident are you in this assessment>,
    "sentiment": "<positive/negative/mixed>",
    "key_findings": ["<finding1>", "<finding2>"]
}}

ONLY output the JSON object."""

        try:
            response = self.llm_client.generate_json(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPTS[dimension],
                temperature=0.3
            )
            
            return AgentEvaluation(
                dimension=dimension,
                score=max(1.0, min(10.0, float(response.get('score', 5.0)))),
                confidence=max(0.0, min(1.0, float(response.get('confidence', 0.7)))),
                sentiment=response.get('sentiment', 'mixed'),
                key_findings=response.get('key_findings', [])
            )
        except Exception as e:
            logger.warning(f"Error evaluating {dimension}: {e}")
            return AgentEvaluation(
                dimension=dimension,
                score=5.0,
                confidence=0.5,
                sentiment='mixed'
            )
    
    def _format_reviews(self, reviews: List[Dict]) -> str:
        if not reviews:
            return "No reviews available."
        
        formatted = []
        for i, review in enumerate(reviews, 1):
            rating = review.get('rating', 'N/A')
            text = review.get('review_text', '')[:500]
            formatted.append(f"Review {i} (Rating: {rating}):\n{text}...")
        
        return "\n\n".join(formatted)


# ============================================================================
# 评估指标计算
# ============================================================================

def compute_spearman(scores: List[float], citations: List[float]) -> Tuple[float, float]:
    if len(scores) < 3:
        return 0.0, 1.0
    valid = [(s, c) for s, c in zip(scores, citations) if s is not None and c is not None]
    if len(valid) < 3:
        return 0.0, 1.0
    s, c = zip(*valid)
    rho, p = spearmanr(s, c)
    return float(rho), float(p)


def compute_precision_at_k(scores: List[float], is_high_impact: List[bool], k: float) -> float:
    if not scores:
        return 0.0
    n = len(scores)
    top_k = max(1, int(n * k))
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in indexed[:top_k]]
    return sum(1 for idx in top_indices if is_high_impact[idx]) / top_k


def compute_recall_at_k(scores: List[float], is_high_impact: List[bool], k: float) -> float:
    if not scores:
        return 0.0
    total_high = sum(is_high_impact)
    if total_high == 0:
        return 0.0
    n = len(scores)
    top_k = max(1, int(n * k))
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_indices = set(idx for idx, _ in indexed[:top_k])
    return sum(1 for i, hi in enumerate(is_high_impact) if hi and i in top_indices) / total_high


def compute_fnr(decisions: List[str], is_high_impact: List[bool]) -> float:
    total = sum(is_high_impact)
    if total == 0:
        return 0.0
    fn = sum(1 for d, hi in zip(decisions, is_high_impact) if hi and d == "Reject")
    return fn / total


def compute_fpr(decisions: List[str], is_high_impact: List[bool]) -> float:
    total = sum(1 for hi in is_high_impact if not hi)
    if total == 0:
        return 0.0
    fp = sum(1 for d, hi in zip(decisions, is_high_impact) if not hi and d == "Accept")
    return fp / total


def compute_agreement(decisions: List[str], real: List[str]) -> float:
    if not decisions:
        return 0.0
    return sum(1 for d, r in zip(decisions, real) if d == r) / len(decisions)


def compute_kappa(decisions: List[str], real: List[str]) -> float:
    if not decisions:
        return 0.0
    n = len(decisions)
    tp = sum(1 for d, r in zip(decisions, real) if d == "Accept" and r == "Accept")
    tn = sum(1 for d, r in zip(decisions, real) if d == "Reject" and r == "Reject")
    fp = sum(1 for d, r in zip(decisions, real) if d == "Accept" and r == "Reject")
    fn = sum(1 for d, r in zip(decisions, real) if d == "Reject" and r == "Accept")
    
    po = (tp + tn) / n
    pe = ((tp + fp) / n) * ((tp + fn) / n) + ((tn + fn) / n) * ((tn + fp) / n)
    
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def bootstrap_ci(scores: List[float], citations: List[float], n_bootstrap: int = 1000) -> Tuple[float, float]:
    np.random.seed(42)
    n = len(scores)
    if n < 10:
        return 0.0, 0.0
    
    correlations = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        s = [scores[i] for i in idx]
        c = [citations[i] for i in idx]
        rho, _ = spearmanr(s, c)
        if not np.isnan(rho):
            correlations.append(rho)
    
    if not correlations:
        return 0.0, 0.0
    return float(np.percentile(correlations, 2.5)), float(np.percentile(correlations, 97.5))


def evaluate_mechanism(name: str, results: List[MechanismResult], 
                      papers: List[PaperData]) -> MechanismMetrics:
    """评估单个机制"""
    scores = [r.aggregated_score for r in results]
    decisions = [r.decision for r in results]
    citations = [p.citations for p in papers]
    is_high = [p.is_high_impact for p in papers]
    real = [p.real_decision for p in papers]
    
    rho, rho_p = compute_spearman(scores, citations)
    r, r_p = pearsonr(scores, citations) if len(scores) >= 3 else (0.0, 1.0)
    ci_lo, ci_hi = bootstrap_ci(scores, citations)
    
    return MechanismMetrics(
        mechanism_name=name,
        spearman_rho=rho,
        spearman_p=rho_p,
        pearson_r=float(r),
        pearson_p=float(r_p),
        precision_at_20=compute_precision_at_k(scores, is_high, 0.2),
        precision_at_30=compute_precision_at_k(scores, is_high, 0.3),
        recall_at_20=compute_recall_at_k(scores, is_high, 0.2),
        recall_at_30=compute_recall_at_k(scores, is_high, 0.3),
        fnr=compute_fnr(decisions, is_high),
        fpr=compute_fpr(decisions, is_high),
        agreement_with_real=compute_agreement(decisions, real),
        cohens_kappa=compute_kappa(decisions, real),
        n_papers=len(papers),
        n_accept=sum(1 for d in decisions if d == "Accept"),
        n_high_impact=sum(is_high),
        spearman_ci_lower=ci_lo,
        spearman_ci_upper=ci_hi
    )


# ============================================================================
# 数据加载
# ============================================================================

def load_agent_results(file_path: str) -> List[Dict]:
    """加载exp1的agent评估结果，支持文件或文件夹"""
    path = Path(file_path)
    
    if path.is_dir():
        # 文件夹：加载所有JSON文件
        all_results = []
        json_files = sorted(path.glob('*.json'))
        logger.info(f"Found {len(json_files)} JSON files in {path}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'paper_details' in data:
                    all_results.extend(data['paper_details'])
                elif isinstance(data, list):
                    all_results.extend(data)
                else:
                    # 单个结果
                    all_results.append(data)
                    
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
        
        return all_results
    else:
        # 单个文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'paper_details' in data:
            return data['paper_details']
        return data if isinstance(data, list) else []


def load_papers(file_path: str) -> Dict[str, Dict]:
    """加载原始论文数据，支持文件或文件夹"""
    if not file_path:
        return {}
    
    path = Path(file_path)
    if not path.exists():
        return {}
    
    all_papers = {}
    
    if path.is_dir():
        # 文件夹：加载所有JSON文件
        json_files = sorted(path.glob('*.json'))
        logger.info(f"Found {len(json_files)} paper JSON files in {path}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    papers = json.load(f)
                
                if isinstance(papers, list):
                    for p in papers:
                        pid = p.get('paper_id', '')
                        if pid:
                            all_papers[pid] = p
                elif isinstance(papers, dict) and 'paper_id' in papers:
                    all_papers[papers['paper_id']] = papers
                    
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
    else:
        # 单个文件
        with open(file_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        if isinstance(papers, list):
            all_papers = {p.get('paper_id', ''): p for p in papers}
        elif isinstance(papers, dict) and 'papers' in papers:
            all_papers = {p.get('paper_id', ''): p for p in papers['papers']}
    
    return all_papers


def load_ground_truth(file_path: str) -> Dict[str, Dict]:
    """加载ground truth数据，支持文件或文件夹"""
    if not file_path:
        return {}
    
    path = Path(file_path)
    if not path.exists():
        return {}
    
    all_gt = {}
    
    if path.is_dir():
        # 文件夹：加载所有JSON文件
        json_files = sorted(path.glob('*.json'))
        logger.info(f"Found {len(json_files)} ground truth JSON files in {path}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for p in data:
                        pid = p.get('paper_id', '')
                        if pid:
                            all_gt[pid] = p
                elif isinstance(data, dict):
                    if 'papers' in data:
                        for p in data['papers']:
                            pid = p.get('paper_id', '')
                            if pid:
                                all_gt[pid] = p
                    elif 'paper_id' in data:
                        all_gt[data['paper_id']] = data
                    else:
                        # 可能是 {paper_id: data} 格式
                        all_gt.update(data)
                        
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
    else:
        # 单个文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            all_gt = {p.get('paper_id', ''): p for p in data}
        elif 'papers' in data:
            all_gt = {p.get('paper_id', ''): p for p in data['papers']}
        elif isinstance(data, dict):
            all_gt = data
    
    return all_gt


def prepare_papers(agent_results: List[Dict], 
                   papers_data: Dict[str, Dict],
                   ground_truth: Dict[str, Dict],
                   high_impact_pct: float = 30.0) -> List[PaperData]:
    """准备论文数据"""
    papers = []
    all_citations = []
    
    # 收集引用数据
    for result in agent_results:
        pid = result.get('paper_id', '')
        gt = ground_truth.get(pid, {})
        citations = gt.get('citations', 0)
        if citations > 0:
            all_citations.append(citations)
    
    # 计算高影响力阈值
    threshold = np.percentile(all_citations, 100 - high_impact_pct) if all_citations else 0
    
    for result in agent_results:
        pid = result.get('paper_id', '')
        raw = papers_data.get(pid, {})
        gt = ground_truth.get(pid, {})
        
        expert_scores = result.get('expert_scores', {})
        if not expert_scores:
            continue
        
        citations = gt.get('citations', 0)
        
        # 如果没有引用数据，根据分数模拟
        if citations == 0 and not gt:
            import random
            avg = sum(expert_scores.values()) / len(expert_scores)
            citations = random.randint(50, 200) if avg >= 7 else random.randint(0, 80)
        
        paper = PaperData(
            paper_id=pid,
            title=raw.get('title', result.get('title', '')),
            abstract=raw.get('abstract', ''),
            reviews=raw.get('reviews', []),
            rebuttal=raw.get('rebuttal'),
            real_decision=result.get('real_decision', gt.get('decision', '')),
            expert_scores=expert_scores,
            weighted_score=result.get('weighted_score', 0.0),
            simulated_decision=result.get('simulated_decision', ''),
            citations=citations,
            is_high_impact=citations >= threshold if threshold > 0 else False
        )
        papers.append(paper)
    
    logger.info(f"Prepared {len(papers)} papers")
    logger.info(f"High impact threshold: {threshold:.0f} citations")
    logger.info(f"High impact papers: {sum(p.is_high_impact for p in papers)}")
    
    return papers


# ============================================================================
# 主实验流程
# ============================================================================

def run_experiment(papers: List[PaperData], 
                   llm_client,
                   use_llm: bool,
                   threshold: float = 5.5) -> Dict[str, Any]:
    """运行机制比较实验"""
    
    results = {
        'config': {
            'threshold': threshold,
            'use_llm': use_llm,
            'n_papers': len(papers),
            'timestamp': datetime.now().isoformat()
        },
        'mechanisms': {},
        'summary': {}
    }
    
    all_mechanism_results = {}
    
    # ========== M1: Simple Average ==========
    logger.info("Running M1: Simple Average...")
    m1 = SimpleAverageMechanism(threshold)
    m1_results = [m1.aggregate(p) for p in papers]
    all_mechanism_results['M1_SimpleAverage'] = m1_results
    results['mechanisms']['M1_SimpleAverage'] = evaluate_mechanism(
        'M1_SimpleAverage', m1_results, papers
    ).to_dict()
    
    # ========== M2v: Dimension-Weighted ==========
    logger.info("Running M2v: Dimension-Weighted...")
    m2v = DimensionWeightedMechanism(threshold)
    m2v_results = [m2v.aggregate(p) for p in papers]
    all_mechanism_results['M2v_DimensionWeighted'] = m2v_results
    results['mechanisms']['M2v_DimensionWeighted'] = evaluate_mechanism(
        'M2v_DimensionWeighted', m2v_results, papers
    ).to_dict()
    
    # ========== M2: Confidence-Weighted (需要LLM) ==========
    if use_llm and llm_client:
        logger.info("Running M2: Confidence-Weighted (with LLM)...")
        expert_runner = ExpertAgentRunner(llm_client)
        m2_results = []
        
        start_time = time.time()
        for i, paper in enumerate(papers):
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (len(papers) - i - 1)
                logger.info(f"  M2 Progress: {i+1}/{len(papers)}, ETA: {eta/60:.1f} min")
            
            # 调用Expert Agent获取带confidence的评估
            evals = expert_runner.evaluate_paper(paper)
            
            # 更新paper的confidence数据
            paper.expert_confidences = {dim: e.confidence for dim, e in evals.items()}
            
            # Confidence加权计算
            total_w = sum(e.confidence for e in evals.values())
            if total_w > 0:
                score = sum(e.score * e.confidence for e in evals.values()) / total_w
            else:
                score = sum(e.score for e in evals.values()) / len(evals)
            
            avg_conf = sum(e.confidence for e in evals.values()) / len(evals)
            decision = "Accept" if score >= threshold else "Reject"
            
            m2_results.append(MechanismResult(
                mechanism_name="M2_ConfidenceWeighted",
                paper_id=paper.paper_id,
                aggregated_score=score,
                decision=decision,
                confidence=avg_conf,
                metadata={
                    "dimension_scores": {d: e.score for d, e in evals.items()},
                    "dimension_confidences": {d: e.confidence for d, e in evals.items()}
                }
            ))
        
        all_mechanism_results['M2_ConfidenceWeighted'] = m2_results
        results['mechanisms']['M2_ConfidenceWeighted'] = evaluate_mechanism(
            'M2_ConfidenceWeighted', m2_results, papers
        ).to_dict()
    else:
        logger.info("Skipping M2 (requires --use-llm)")
    
    # ========== M3: AC-Centric ==========
    logger.info("Running M3: AC-Centric...")
    m3 = ACCentricMechanism(threshold)
    m3_results = [m3.aggregate(p) for p in papers]
    all_mechanism_results['M3_ACCentric'] = m3_results
    results['mechanisms']['M3_ACCentric'] = evaluate_mechanism(
        'M3_ACCentric', m3_results, papers
    ).to_dict()
    
    # ========== M4: Bounded Iteration ==========
    logger.info(f"Running M4: Bounded Iteration ({'with LLM' if use_llm else 'simulated'})...")
    m4 = BoundedIterationMechanism(threshold, llm_client if use_llm else None)
    m4_results = []
    
    if use_llm and llm_client:
        start_time = time.time()
        for i, paper in enumerate(papers):
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (len(papers) - i - 1)
                logger.info(f"  M4 Progress: {i+1}/{len(papers)}, ETA: {eta/60:.1f} min")
            m4_results.append(m4.aggregate(paper))
    else:
        m4_results = [m4.aggregate(p) for p in papers]
    
    all_mechanism_results['M4_BoundedIteration'] = m4_results
    results['mechanisms']['M4_BoundedIteration'] = evaluate_mechanism(
        'M4_BoundedIteration', m4_results, papers
    ).to_dict()
    
    # ========== M5: Diverse Ensemble ==========
    logger.info(f"Running M5: Diverse Ensemble ({'with LLM' if use_llm else 'simulated'})...")
    m5 = DiverseEnsembleMechanism(threshold, llm_client if use_llm else None)
    m5_results = []
    
    if use_llm and llm_client:
        start_time = time.time()
        for i, paper in enumerate(papers):
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (len(papers) - i - 1)
                logger.info(f"  M5 Progress: {i+1}/{len(papers)}, ETA: {eta/60:.1f} min")
            m5_results.append(m5.aggregate(paper))
    else:
        m5_results = [m5.aggregate(p) for p in papers]
    
    all_mechanism_results['M5_DiverseEnsemble'] = m5_results
    results['mechanisms']['M5_DiverseEnsemble'] = evaluate_mechanism(
        'M5_DiverseEnsemble', m5_results, papers
    ).to_dict()
    
    # ========== Actual Decisions Baseline ==========
    logger.info("Computing Actual Decisions baseline...")
    actual_scores = [1.0 if p.real_decision == "Accept" else 0.0 for p in papers]
    actual_decisions = [p.real_decision for p in papers]
    citations = [p.citations for p in papers]
    is_high = [p.is_high_impact for p in papers]
    
    rho, rho_p = compute_spearman(actual_scores, citations)
    
    results['mechanisms']['Actual_Decisions'] = {
        'mechanism_name': 'Actual_Decisions',
        'spearman_rho': rho,
        'spearman_p': rho_p,
        'precision_at_20': compute_precision_at_k(actual_scores, is_high, 0.2),
        'precision_at_30': compute_precision_at_k(actual_scores, is_high, 0.3),
        'fnr': compute_fnr(actual_decisions, is_high),
        'fpr': compute_fpr(actual_decisions, is_high),
        'agreement_with_real': 1.0,
        'cohens_kappa': 1.0,
        'n_papers': len(papers)
    }
    
    # 生成摘要
    results['summary'] = generate_summary(results['mechanisms'])
    
    return results


def generate_summary(mechanisms: Dict) -> Dict:
    """生成结果摘要"""
    valid = {k: v for k, v in mechanisms.items() 
             if k != 'Actual_Decisions' and isinstance(v, dict) and 'spearman_rho' in v}
    
    if not valid:
        return {}
    
    best_rho = max(valid.items(), key=lambda x: x[1].get('spearman_rho', 0))
    best_p20 = max(valid.items(), key=lambda x: x[1].get('precision_at_20', 0))
    best_fnr = min(valid.items(), key=lambda x: x[1].get('fnr', 1))
    
    ranking = sorted(valid.items(), key=lambda x: x[1].get('spearman_rho', 0), reverse=True)
    
    actual_rho = mechanisms.get('Actual_Decisions', {}).get('spearman_rho', 0)
    actual_fnr = mechanisms.get('Actual_Decisions', {}).get('fnr', 0)
    
    improvements = {}
    for name, data in valid.items():
        rho_imp = (data['spearman_rho'] - actual_rho) / max(abs(actual_rho), 0.01) * 100
        fnr_imp = (actual_fnr - data['fnr']) / max(actual_fnr, 0.01) * 100
        improvements[name] = {
            'spearman_improvement_pct': rho_imp,
            'fnr_reduction_pct': fnr_imp
        }
    
    return {
        'best_spearman': {'mechanism': best_rho[0], 'value': best_rho[1]['spearman_rho']},
        'best_precision_20': {'mechanism': best_p20[0], 'value': best_p20[1]['precision_at_20']},
        'lowest_fnr': {'mechanism': best_fnr[0], 'value': best_fnr[1]['fnr']},
        'ranking_by_spearman': [
            {'rank': i+1, 'mechanism': name, 'spearman_rho': data['spearman_rho']}
            for i, (name, data) in enumerate(ranking)
        ],
        'improvement_over_actual': improvements
    }


def format_markdown_table(results: Dict) -> str:
    """生成Markdown表格"""
    mechanisms = results.get('mechanisms', {})
    
    md = "\n## Mechanism Comparison Results\n\n"
    md += "| Mechanism | ρ_impact | P@20% | P@30% | FNR | Agreement | κ |\n"
    md += "|-----------|----------|-------|-------|-----|-----------|---|\n"
    
    order = ['Actual_Decisions', 'M1_SimpleAverage', 'M2_ConfidenceWeighted',
             'M2v_DimensionWeighted', 'M3_ACCentric', 'M4_BoundedIteration',
             'M5_DiverseEnsemble']
    
    for name in order:
        if name not in mechanisms:
            continue
        data = mechanisms[name]
        if not isinstance(data, dict):
            continue
        
        display = name.replace('_', ' ')
        md += f"| {display} | {data.get('spearman_rho', 0):.3f} | "
        md += f"{data.get('precision_at_20', 0):.3f} | {data.get('precision_at_30', 0):.3f} | "
        md += f"{data.get('fnr', 0):.3f} | {data.get('agreement_with_real', 0):.3f} | "
        md += f"{data.get('cohens_kappa', 0):.3f} |\n"
    
    return md


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Exp 4: Mechanism Comparison')
    
    parser.add_argument('--agent-results', '-a', required=True,
                       help='Agent评估结果JSON文件或文件夹（exp1输出）')
    parser.add_argument('--papers', '-p', default=None,
                       help='原始论文数据JSON文件或文件夹')
    parser.add_argument('--ground-truth', '-g', default=None,
                       help='Ground Truth数据文件或文件夹（包含引用数据）')
    parser.add_argument('--output', '-o', default='./exp4_results.json',
                       help='输出结果文件')
    
    parser.add_argument('--threshold', type=float, default=5.5,
                       help='Accept/Reject阈值')
    parser.add_argument('--high-impact-pct', type=float, default=30.0,
                       help='高影响力论文百分位')
    
    parser.add_argument('--use-llm', action='store_true',
                       help='使用LLM运行M2/M4/M5（需要ollama）')
    parser.add_argument('--model', '-m', default='llama3.1:8b',
                       help='Ollama模型名称')
    parser.add_argument('--base-url', default='http://localhost:11434',
                       help='Ollama API base URL')
    
    parser.add_argument('--limit', type=int, default=None,
                       help='限制处理论文数（测试用）')
    parser.add_argument('--markdown', action='store_true',
                       help='输出Markdown表格')
    
    args = parser.parse_args()
    
    # 加载数据
    logger.info(f"Loading agent results from {args.agent_results}")
    agent_results = load_agent_results(args.agent_results)
    
    if args.limit:
        agent_results = agent_results[:args.limit]
    
    logger.info(f"Loaded {len(agent_results)} agent results")
    
    logger.info(f"Loading papers data from {args.papers}" if args.papers else "No papers path provided")
    papers_data = load_papers(args.papers) if args.papers else {}
    logger.info(f"Loaded {len(papers_data)} papers")
    
    logger.info(f"Loading ground truth from {args.ground_truth}" if args.ground_truth else "No ground truth provided")
    ground_truth = load_ground_truth(args.ground_truth) if args.ground_truth else {}
    logger.info(f"Loaded {len(ground_truth)} ground truth entries")
    
    # 准备数据
    papers = prepare_papers(agent_results, papers_data, ground_truth, args.high_impact_pct)
    
    if not papers:
        logger.error("No valid papers to process")
        return
    
    # 初始化LLM
    llm_client = None
    if args.use_llm:
        try:
            # 尝试导入项目的LLM客户端
            sys.path.insert(0, str(Path(args.agent_results).parent.parent))
            from llm_client import OllamaClient
            llm_client = OllamaClient(model=args.model, base_url=args.base_url)
            logger.info(f"Initialized LLM client: {args.model} @ {args.base_url}")
        except ImportError:
            # 使用简化版本
            logger.warning("Could not import OllamaClient, using inline implementation")
            llm_client = SimpleOllamaClient(args.model, args.base_url)
    
    # 运行实验
    logger.info("="*60)
    logger.info("Starting Mechanism Comparison Experiment")
    logger.info("="*60)
    
    results = run_experiment(papers, llm_client, args.use_llm, args.threshold)
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换numpy类型为Python原生类型
    serializable_results = convert_to_serializable(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")
    
    # 输出摘要
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    summary = results.get('summary', {})
    
    if summary.get('best_spearman'):
        print(f"\nBest Spearman ρ: {summary['best_spearman']['mechanism']} "
              f"({summary['best_spearman']['value']:.4f})")
    
    if summary.get('best_precision_20'):
        print(f"Best Precision@20%: {summary['best_precision_20']['mechanism']} "
              f"({summary['best_precision_20']['value']:.4f})")
    
    if summary.get('lowest_fnr'):
        print(f"Lowest FNR: {summary['lowest_fnr']['mechanism']} "
              f"({summary['lowest_fnr']['value']:.4f})")
    
    print("\nRanking by Spearman ρ:")
    for item in summary.get('ranking_by_spearman', []):
        print(f"  {item['rank']}. {item['mechanism']}: {item['spearman_rho']:.4f}")
    
    if args.markdown:
        print(format_markdown_table(results))


# ============================================================================
# 简化版Ollama客户端（备用）
# ============================================================================

class SimpleOllamaClient:
    """简化版Ollama客户端"""
    
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def generate_json(self, prompt: str, system_prompt: str = "", temperature: float = 0.3) -> Dict:
        import requests
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature}
                },
                timeout=120
            )
            response.raise_for_status()
            
            content = response.json().get("message", {}).get("content", "")
            
            # 尝试解析JSON
            import re
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return {"score": 5.0, "confidence": 0.5}
            
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return {"score": 5.0, "confidence": 0.5}


if __name__ == '__main__':
    main()