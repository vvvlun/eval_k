"""
Exp 4 Full: Mechanism Comparison - 优化版本

先运行一次4个Expert Agents评估，然后复用结果计算各机制。

流程：
1. 对每篇论文运行4个Expert Agents（只跑一次）
2. 基于同一份评估结果计算各机制：
   - M1: 简单平均（无需额外LLM）
   - M2: confidence加权（无需额外LLM）
   - M3: 调Chair Agent（+1次LLM）
   - M4: 迭代更新（+12次LLM，3轮×4维度）
   - M5: 调2个generalist（+2次LLM）

LLM调用次数：
- Step 1: 4次/paper (共用)
- M1: 0次额外
- M2: 0次额外  
- M3: 1次额外
- M4: 12次额外
- M5: 2次额外
- 总计: 19次/paper (vs 35次/paper 如果重复跑)

使用方法：
    python exp4_full.py \
        --papers ./data/raw/iclr_papers \
        --ground-truth ./data/processed/ground_truth.json \
        --output ./data/results/exp4_full.json \
        --model llama3.1:70b \
        --limit 100
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
from scipy.stats import spearmanr, pearsonr
from collections import defaultdict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_to_serializable(obj):
    """将numpy类型转换为Python原生类型"""
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
    reasoning: str = ""


@dataclass
class PaperData:
    """论文数据"""
    paper_id: str
    title: str = ""
    abstract: str = ""
    reviews: List[Dict] = field(default_factory=list)
    rebuttal: Optional[str] = None
    real_decision: str = ""
    citations: int = 0
    is_high_impact: bool = False
    
    # Expert评估结果（Step 1后填充）
    expert_evals: Dict[str, AgentEvaluation] = field(default_factory=dict)


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
# LLM客户端
# ============================================================================

class OllamaClient:
    """Ollama客户端"""
    
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def generate_json(self, prompt: str, system_prompt: str = "", 
                      temperature: float = 0.3) -> Dict:
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


# ============================================================================
# Step 1: Expert Agent评估（只运行一次）
# ============================================================================

class ExpertAgentRunner:
    """运行Expert Agent获取带confidence的评估"""
    
    DIMENSIONS = ['novelty', 'methodology', 'clarity', 'empirical']
    
    SYSTEM_PROMPTS = {
        'novelty': """You are an expert reviewer specializing in NOVELTY and ORIGINALITY.
Focus ONLY on: originality of ideas, difference from prior work, potential impact on the field.
Do NOT consider methodology quality, writing clarity, or experimental rigor.""",
        
        'methodology': """You are an expert reviewer specializing in METHODOLOGY.
Focus ONLY on: technical correctness, theoretical foundation, algorithmic rigor.
Do NOT consider novelty, writing quality, or experimental completeness.""",
        
        'clarity': """You are an expert reviewer specializing in CLARITY and PRESENTATION.
Focus ONLY on: writing quality, organization, figure quality, readability.
Do NOT consider novelty, technical correctness, or experimental quality.""",
        
        'empirical': """You are an expert reviewer specializing in EMPIRICAL EVALUATION.
Focus ONLY on: experimental design, baselines, reproducibility, statistical rigor.
Do NOT consider novelty, theoretical contributions, or writing quality."""
    }
    
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client
    
    def evaluate_paper(self, paper: PaperData) -> Dict[str, AgentEvaluation]:
        """评估单篇论文，返回所有维度的评估"""
        results = {}
        for dim in self.DIMENSIONS:
            eval_result = self._evaluate_dimension(paper, dim)
            results[dim] = eval_result
        return results
    
    def _evaluate_dimension(self, paper: PaperData, dimension: str) -> AgentEvaluation:
        """评估单个维度"""
        reviews_text = self._format_reviews(paper.reviews)
        
        prompt = f"""Evaluate the {dimension.upper()} of this paper based on the reviews.

=== Paper Title ===
{paper.title}

=== Abstract ===
{paper.abstract[:1000] if paper.abstract else 'N/A'}

=== Reviews ===
{reviews_text}

Based on the reviews, evaluate the paper's {dimension}.

Respond with a JSON object:
{{
    "score": <float 1-10>,
    "confidence": <float 0-1, how confident you are>,
    "reasoning": "<brief explanation>"
}}

ONLY output the JSON object, nothing else."""

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
                reasoning=response.get('reasoning', '')
            )
        except Exception as e:
            logger.warning(f"Error evaluating {dimension}: {e}")
            return AgentEvaluation(dimension=dimension, score=5.0, confidence=0.5)
    
    def _format_reviews(self, reviews: List[Dict]) -> str:
        if not reviews:
            return "No reviews available."
        
        formatted = []
        for i, review in enumerate(reviews, 1):
            rating = review.get('rating', 'N/A')
            text = review.get('review_text', review.get('comment', ''))[:600]
            formatted.append(f"Review {i} (Rating: {rating}):\n{text}...")
        
        return "\n\n".join(formatted[:3])


def run_expert_evaluations(papers: List[PaperData], llm_client: OllamaClient) -> List[PaperData]:
    """Step 1: 对所有论文运行Expert评估（只运行一次）"""
    logger.info("="*60)
    logger.info("Step 1: Running Expert Agent Evaluations (shared by all mechanisms)")
    logger.info("="*60)
    
    expert_runner = ExpertAgentRunner(llm_client)
    start_time = time.time()
    
    for i, paper in enumerate(papers):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(papers) - i - 1)
            logger.info(f"  Expert Eval Progress: {i+1}/{len(papers)}, ETA: {eta/60:.1f} min")
        
        # 运行4个expert评估并保存到paper对象
        paper.expert_evals = expert_runner.evaluate_paper(paper)
    
    elapsed = time.time() - start_time
    logger.info(f"Step 1 completed in {elapsed/60:.1f} min ({len(papers)} papers × 4 dimensions)")
    
    return papers


# ============================================================================
# Step 2: 各机制计算（复用Expert评估结果）
# ============================================================================

class M1_SimpleAverage:
    """M1: 简单平均 - 直接用expert评分，无需额外LLM"""
    
    def __init__(self, threshold: float = 5.5):
        self.name = "M1_SimpleAverage"
        self.threshold = threshold
    
    def evaluate(self, paper: PaperData) -> MechanismResult:
        evals = paper.expert_evals
        
        # 简单平均
        scores = [e.score for e in evals.values()]
        avg_score = sum(scores) / len(scores)
        
        decision = "Accept" if avg_score >= self.threshold else "Reject"
        
        return MechanismResult(
            mechanism_name=self.name,
            paper_id=paper.paper_id,
            aggregated_score=avg_score,
            decision=decision,
            confidence=0.7,
            metadata={
                "dimension_scores": {d: e.score for d, e in evals.items()},
                "dimension_confidences": {d: e.confidence for d, e in evals.items()}
            }
        )


class M2_ConfidenceWeighted:
    """M2: Confidence加权 - 直接用expert评分，无需额外LLM"""
    
    def __init__(self, threshold: float = 5.5):
        self.name = "M2_ConfidenceWeighted"
        self.threshold = threshold
    
    def evaluate(self, paper: PaperData) -> MechanismResult:
        evals = paper.expert_evals
        
        # Confidence加权平均
        total_weight = sum(e.confidence for e in evals.values())
        if total_weight > 0:
            weighted_score = sum(e.score * e.confidence for e in evals.values()) / total_weight
        else:
            weighted_score = sum(e.score for e in evals.values()) / len(evals)
        
        avg_confidence = sum(e.confidence for e in evals.values()) / len(evals)
        decision = "Accept" if weighted_score >= self.threshold else "Reject"
        
        return MechanismResult(
            mechanism_name=self.name,
            paper_id=paper.paper_id,
            aggregated_score=weighted_score,
            decision=decision,
            confidence=avg_confidence,
            metadata={
                "dimension_scores": {d: e.score for d, e in evals.items()},
                "dimension_confidences": {d: e.confidence for d, e in evals.items()}
            }
        )


class M3_ACCentric:
    """M3: AC-Centric - 用expert评分 + 调Chair Agent（+1次LLM/paper）"""
    
    CHAIR_SYSTEM_PROMPT = """You are an experienced Area Chair (AC) at a top AI conference.
You will see evaluations from 4 expert reviewers on different dimensions.
Your job is to make a holistic assessment considering all dimensions together.
You have the final authority to accept or reject papers."""
    
    def __init__(self, llm_client: OllamaClient, threshold: float = 5.5):
        self.name = "M3_ACCentric"
        self.threshold = threshold
        self.llm_client = llm_client
    
    def evaluate(self, paper: PaperData) -> MechanismResult:
        evals = paper.expert_evals
        
        # 构建给Chair的输入
        expert_summary = "\n".join([
            f"- {dim.upper()}: {e.score:.1f}/10 (confidence: {e.confidence:.2f})\n  Reasoning: {e.reasoning[:100]}..."
            for dim, e in evals.items()
        ])
        
        prompt = f"""As Area Chair, make a final decision on this paper.

=== Paper Title ===
{paper.title}

=== Expert Evaluations ===
{expert_summary}

Based on these expert evaluations, provide your holistic assessment.
Consider:
1. Overall quality across all dimensions
2. Relative importance of each dimension for this paper
3. Whether any critical weaknesses should be disqualifying

Respond with a JSON object:
{{
    "score": <float 1-10, your overall assessment>,
    "confidence": <float 0-1>,
    "decision": "<Accept or Reject>",
    "reasoning": "<your rationale>"
}}

ONLY output the JSON object."""

        try:
            response = self.llm_client.generate_json(
                prompt=prompt,
                system_prompt=self.CHAIR_SYSTEM_PROMPT,
                temperature=0.3
            )
            
            score = max(1.0, min(10.0, float(response.get('score', 5.0))))
            confidence = max(0.0, min(1.0, float(response.get('confidence', 0.7))))
            decision = response.get('decision', 'Reject')
            if decision not in ['Accept', 'Reject']:
                decision = "Accept" if score >= self.threshold else "Reject"
            
        except Exception as e:
            logger.warning(f"Chair evaluation failed: {e}")
            score = sum(e.score for e in evals.values()) / len(evals)
            confidence = 0.5
            decision = "Accept" if score >= self.threshold else "Reject"
        
        return MechanismResult(
            mechanism_name=self.name,
            paper_id=paper.paper_id,
            aggregated_score=score,
            decision=decision,
            confidence=confidence,
            metadata={
                "dimension_scores": {d: e.score for d, e in evals.items()},
                "dimension_confidences": {d: e.confidence for d, e in evals.items()},
                "chair_decision": decision
            }
        )


class M4_BoundedIteration:
    """M4: 有限轮迭代 - 用expert评分作为初始值，然后迭代（+12次LLM/paper）"""
    
    def __init__(self, llm_client: OllamaClient, threshold: float = 5.5, max_rounds: int = 3):
        self.name = "M4_BoundedIteration"
        self.threshold = threshold
        self.llm_client = llm_client
        self.max_rounds = max_rounds
    
    def evaluate(self, paper: PaperData) -> MechanismResult:
        evals = paper.expert_evals
        
        # 用Step 1的评估作为Round 0
        current_scores = {d: e.score for d, e in evals.items()}
        current_confidences = {d: e.confidence for d, e in evals.items()}
        
        round_history = [{
            'round': 0,
            'scores': dict(current_scores),
            'confidences': dict(current_confidences)
        }]
        
        # Rounds 1-3: 迭代更新
        for round_num in range(1, self.max_rounds + 1):
            context = self._build_context(current_scores, round_num)
            
            new_scores = {}
            new_confidences = {}
            
            for dim in current_scores:
                updated = self._update_agent(dim, current_scores[dim], context, paper)
                new_scores[dim] = updated['score']
                new_confidences[dim] = updated['confidence']
            
            current_scores = new_scores
            current_confidences = new_confidences
            
            round_history.append({
                'round': round_num,
                'scores': dict(current_scores),
                'confidences': dict(current_confidences)
            })
        
        # 最终聚合
        total_weight = sum(current_confidences.values())
        if total_weight > 0:
            final_score = sum(
                current_scores[dim] * current_confidences[dim]
                for dim in current_scores
            ) / total_weight
        else:
            final_score = sum(current_scores.values()) / len(current_scores)
        
        avg_confidence = sum(current_confidences.values()) / len(current_confidences)
        decision = "Accept" if final_score >= self.threshold else "Reject"
        
        return MechanismResult(
            mechanism_name=self.name,
            paper_id=paper.paper_id,
            aggregated_score=final_score,
            decision=decision,
            confidence=avg_confidence,
            metadata={
                "initial_scores": {d: e.score for d, e in evals.items()},
                "final_scores": current_scores,
                "final_confidences": current_confidences,
                "round_history": round_history
            }
        )
    
    def _build_context(self, scores: Dict[str, float], round_num: int) -> str:
        context = f"=== Discussion Round {round_num} ===\n"
        context += "Current expert evaluations:\n"
        for dim, score in scores.items():
            context += f"  {dim.capitalize()}: {score:.1f}/10\n"
        return context
    
    def _update_agent(self, dimension: str, current_score: float, 
                      context: str, paper: PaperData) -> Dict:
        prompt = f"""You are the {dimension.upper()} expert in a peer review discussion.

{context}

Your current score for {dimension}: {current_score:.1f}/10

Paper: {paper.title}

After seeing other experts' scores, reconsider your evaluation.
You may adjust your score slightly based on the discussion, but maintain your unique perspective on {dimension}.

Respond with JSON:
{{"score": <float 1-10>, "confidence": <float 0-1>}}"""

        try:
            response = self.llm_client.generate_json(
                prompt=prompt,
                system_prompt=f"You are an expert reviewer for {dimension}. Maintain your perspective.",
                temperature=0.3
            )
            return {
                'score': max(1.0, min(10.0, float(response.get('score', current_score)))),
                'confidence': max(0.0, min(1.0, float(response.get('confidence', 0.7))))
            }
        except:
            return {'score': current_score, 'confidence': 0.7}


class M5_DiverseEnsemble:
    """M5: 扩展Ensemble - 用expert评分 + 2个generalist（+2次LLM/paper）"""
    
    def __init__(self, llm_client: OllamaClient, threshold: float = 5.5):
        self.name = "M5_DiverseEnsemble"
        self.threshold = threshold
        self.llm_client = llm_client
    
    def evaluate(self, paper: PaperData) -> MechanismResult:
        evals = paper.expert_evals
        
        # 获取2个通才评估
        generalist_evals = self._run_generalists(paper, evals)
        
        # 合并所有评估
        all_scores = {d: e.score for d, e in evals.items()}
        all_confidences = {d: e.confidence for d, e in evals.items()}
        
        for name, gen_eval in generalist_evals.items():
            all_scores[name] = gen_eval['score']
            all_confidences[name] = gen_eval['confidence']
        
        # Confidence加权聚合
        total_weight = sum(all_confidences.values())
        if total_weight > 0:
            final_score = sum(
                all_scores[k] * all_confidences[k] for k in all_scores
            ) / total_weight
        else:
            final_score = sum(all_scores.values()) / len(all_scores)
        
        avg_confidence = sum(all_confidences.values()) / len(all_confidences)
        decision = "Accept" if final_score >= self.threshold else "Reject"
        
        return MechanismResult(
            mechanism_name=self.name,
            paper_id=paper.paper_id,
            aggregated_score=final_score,
            decision=decision,
            confidence=avg_confidence,
            metadata={
                "expert_scores": {d: e.score for d, e in evals.items()},
                "generalist_scores": generalist_evals,
                "all_scores": all_scores,
                "all_confidences": all_confidences
            }
        )
    
    def _run_generalists(self, paper: PaperData, expert_evals: Dict) -> Dict:
        expert_summary = "\n".join([
            f"- {dim.upper()}: {e.score:.1f}/10"
            for dim, e in expert_evals.items()
        ])
        
        results = {}
        
        # Holistic Agent
        holistic_prompt = f"""As a holistic reviewer, evaluate this paper considering all aspects together.

Paper: {paper.title}

Expert Evaluations:
{expert_summary}

Provide your overall assessment:
{{"score": <float 1-10>, "confidence": <float 0-1>}}"""

        try:
            response = self.llm_client.generate_json(
                prompt=holistic_prompt,
                system_prompt="You are a holistic reviewer who considers overall paper quality.",
                temperature=0.3
            )
            results['holistic'] = {
                'score': max(1.0, min(10.0, float(response.get('score', 5.0)))),
                'confidence': max(0.0, min(1.0, float(response.get('confidence', 0.6))))
            }
        except:
            avg = sum(e.score for e in expert_evals.values()) / len(expert_evals)
            results['holistic'] = {'score': avg, 'confidence': 0.5}
        
        # Contrarian Agent
        avg_score = sum(e.score for e in expert_evals.values()) / len(expert_evals)
        contrarian_prompt = f"""As a contrarian reviewer, critically examine whether the consensus is correct.

Paper: {paper.title}

Expert Evaluations:
{expert_summary}

Average score: {avg_score:.1f}/10

Challenge the consensus if appropriate. Are there overlooked strengths or weaknesses?
{{"score": <float 1-10>, "confidence": <float 0-1>}}"""

        try:
            response = self.llm_client.generate_json(
                prompt=contrarian_prompt,
                system_prompt="You are a contrarian reviewer who challenges consensus views.",
                temperature=0.5
            )
            results['contrarian'] = {
                'score': max(1.0, min(10.0, float(response.get('score', 5.0)))),
                'confidence': max(0.0, min(1.0, float(response.get('confidence', 0.5))))
            }
        except:
            contrarian_score = avg_score + (1.0 if avg_score < 5.5 else -1.0)
            results['contrarian'] = {'score': contrarian_score, 'confidence': 0.4}
        
        return results


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

def load_papers(file_path: str) -> List[Dict]:
    """加载论文数据"""
    path = Path(file_path)
    all_papers = []
    
    if path.is_dir():
        json_files = sorted(path.glob('*.json'))
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    all_papers.extend(data)
                elif 'papers' in data:
                    all_papers.extend(data['papers'])
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            all_papers = data
        elif 'papers' in data:
            all_papers = data['papers']
    
    return all_papers


def load_ground_truth(file_path: str) -> Dict[str, Dict]:
    """加载ground truth数据"""
    if not file_path:
        return {}
    
    path = Path(file_path)
    if not path.exists():
        return {}
    
    all_gt = {}
    
    if path.is_dir():
        json_files = sorted(path.glob('*.json'))
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
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            all_gt = {p.get('paper_id', ''): p for p in data}
        elif 'papers' in data:
            all_gt = {p.get('paper_id', ''): p for p in data['papers']}
    
    return all_gt


def prepare_papers(raw_papers: List[Dict], ground_truth: Dict[str, Dict],
                   high_impact_pct: float = 30.0) -> List[PaperData]:
    """准备论文数据"""
    all_citations = []
    for p in raw_papers:
        pid = p.get('paper_id', '')
        gt = ground_truth.get(pid, {})
        citations = gt.get('citations', p.get('citations', 0))
        if citations > 0:
            all_citations.append(citations)
    
    threshold = np.percentile(all_citations, 100 - high_impact_pct) if all_citations else 0
    logger.info(f"High impact threshold: {threshold:.0f} citations (top {high_impact_pct}%)")
    
    papers = []
    for p in raw_papers:
        pid = p.get('paper_id', '')
        gt = ground_truth.get(pid, {})
        
        citations = gt.get('citations', p.get('citations', 0))
        decision = gt.get('decision', p.get('decision', ''))
        
        paper = PaperData(
            paper_id=pid,
            title=p.get('title', ''),
            abstract=p.get('abstract', ''),
            reviews=p.get('reviews', []),
            rebuttal=p.get('rebuttal'),
            real_decision=decision,
            citations=citations,
            is_high_impact=citations >= threshold if threshold > 0 else False
        )
        papers.append(paper)
    
    logger.info(f"Prepared {len(papers)} papers")
    logger.info(f"High impact papers: {sum(p.is_high_impact for p in papers)}")
    
    return papers


# ============================================================================
# 主实验流程
# ============================================================================

def run_experiment(papers: List[PaperData], llm_client: OllamaClient,
                   threshold: float = 5.5,
                   mechanisms_to_run: List[str] = None) -> Dict[str, Any]:
    """运行完整实验"""
    
    all_mechanisms = ['M1', 'M2', 'M3', 'M4', 'M5']
    if mechanisms_to_run:
        all_mechanisms = [m for m in all_mechanisms if m in mechanisms_to_run]
    
    results = {
        'config': {
            'threshold': threshold,
            'n_papers': len(papers),
            'mechanisms': all_mechanisms,
            'timestamp': datetime.now().isoformat()
        },
        'mechanisms': {},
        'expert_evaluations': {},
        'summary': {}
    }
    
    # ========== Step 1: 运行Expert评估（共用） ==========
    papers = run_expert_evaluations(papers, llm_client)
    
    # 保存expert评估结果
    for paper in papers:
        results['expert_evaluations'][paper.paper_id] = {
            d: {'score': e.score, 'confidence': e.confidence, 'reasoning': e.reasoning}
            for d, e in paper.expert_evals.items()
        }
    
    # ========== Step 2: 计算各机制 ==========
    
    # M1: Simple Average
    if 'M1' in all_mechanisms:
        logger.info("="*50)
        logger.info("Step 2a: Computing M1 (Simple Average) - no extra LLM calls")
        m1 = M1_SimpleAverage(threshold)
        m1_results = [m1.evaluate(p) for p in papers]
        results['mechanisms']['M1_SimpleAverage'] = evaluate_mechanism(
            'M1_SimpleAverage', m1_results, papers
        ).to_dict()
        logger.info(f"  M1 Spearman ρ: {results['mechanisms']['M1_SimpleAverage']['spearman_rho']:.4f}")
    
    # M2: Confidence-Weighted
    if 'M2' in all_mechanisms:
        logger.info("="*50)
        logger.info("Step 2b: Computing M2 (Confidence-Weighted) - no extra LLM calls")
        m2 = M2_ConfidenceWeighted(threshold)
        m2_results = [m2.evaluate(p) for p in papers]
        results['mechanisms']['M2_ConfidenceWeighted'] = evaluate_mechanism(
            'M2_ConfidenceWeighted', m2_results, papers
        ).to_dict()
        logger.info(f"  M2 Spearman ρ: {results['mechanisms']['M2_ConfidenceWeighted']['spearman_rho']:.4f}")
    
    # M3: AC-Centric
    if 'M3' in all_mechanisms:
        logger.info("="*50)
        logger.info("Step 2c: Computing M3 (AC-Centric) - +1 LLM call per paper")
        m3 = M3_ACCentric(llm_client, threshold)
        m3_results = []
        start_time = time.time()
        for i, paper in enumerate(papers):
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (len(papers) - i - 1)
                logger.info(f"  M3 Progress: {i+1}/{len(papers)}, ETA: {eta/60:.1f} min")
            m3_results.append(m3.evaluate(paper))
        results['mechanisms']['M3_ACCentric'] = evaluate_mechanism(
            'M3_ACCentric', m3_results, papers
        ).to_dict()
        logger.info(f"  M3 Spearman ρ: {results['mechanisms']['M3_ACCentric']['spearman_rho']:.4f}")
    
    # M4: Bounded Iteration
    if 'M4' in all_mechanisms:
        logger.info("="*50)
        logger.info("Step 2d: Computing M4 (Bounded Iteration) - +12 LLM calls per paper")
        m4 = M4_BoundedIteration(llm_client, threshold, max_rounds=3)
        m4_results = []
        start_time = time.time()
        for i, paper in enumerate(papers):
            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (len(papers) - i - 1)
                logger.info(f"  M4 Progress: {i+1}/{len(papers)}, ETA: {eta/60:.1f} min")
            m4_results.append(m4.evaluate(paper))
        results['mechanisms']['M4_BoundedIteration'] = evaluate_mechanism(
            'M4_BoundedIteration', m4_results, papers
        ).to_dict()
        logger.info(f"  M4 Spearman ρ: {results['mechanisms']['M4_BoundedIteration']['spearman_rho']:.4f}")
    
    # M5: Diverse Ensemble
    if 'M5' in all_mechanisms:
        logger.info("="*50)
        logger.info("Step 2e: Computing M5 (Diverse Ensemble) - +2 LLM calls per paper")
        m5 = M5_DiverseEnsemble(llm_client, threshold)
        m5_results = []
        start_time = time.time()
        for i, paper in enumerate(papers):
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (len(papers) - i - 1)
                logger.info(f"  M5 Progress: {i+1}/{len(papers)}, ETA: {eta/60:.1f} min")
            m5_results.append(m5.evaluate(paper))
        results['mechanisms']['M5_DiverseEnsemble'] = evaluate_mechanism(
            'M5_DiverseEnsemble', m5_results, papers
        ).to_dict()
        logger.info(f"  M5 Spearman ρ: {results['mechanisms']['M5_DiverseEnsemble']['spearman_rho']:.4f}")
    
    # Actual Decisions Baseline
    logger.info("="*50)
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
    best_p30 = max(valid.items(), key=lambda x: x[1].get('precision_at_30', 0))
    
    ranking = sorted(valid.items(), key=lambda x: x[1].get('spearman_rho', 0), reverse=True)
    
    return {
        'best_spearman': {'mechanism': best_rho[0], 'value': best_rho[1]['spearman_rho']},
        'best_precision_20': {'mechanism': best_p20[0], 'value': best_p20[1]['precision_at_20']},
        'best_precision_30': {'mechanism': best_p30[0], 'value': best_p30[1]['precision_at_30']},
        'ranking_by_spearman': [
            {'rank': i+1, 'mechanism': name, 'spearman_rho': data['spearman_rho']}
            for i, (name, data) in enumerate(ranking)
        ]
    }


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Exp 4 Full: Optimized mechanism comparison')
    
    parser.add_argument('--papers', '-p', required=True,
                       help='论文数据JSON文件或文件夹')
    parser.add_argument('--ground-truth', '-g', default=None,
                       help='Ground Truth数据（包含引用数据）')
    parser.add_argument('--output', '-o', default='./exp4_full_results.json',
                       help='输出结果文件')
    
    parser.add_argument('--threshold', type=float, default=5.5,
                       help='Accept/Reject阈值')
    parser.add_argument('--high-impact-pct', type=float, default=30.0,
                       help='高影响力论文百分位')
    
    parser.add_argument('--model', '-m', default='llama3.1:70b',
                       help='Ollama模型名称')
    parser.add_argument('--base-url', default='http://localhost:11434',
                       help='Ollama API base URL')
    
    parser.add_argument('--mechanisms', nargs='+', default=None,
                       help='指定要运行的机制，如 M1 M2 M3')
    parser.add_argument('--limit', type=int, default=None,
                       help='限制处理论文数（测试用）')
    
    args = parser.parse_args()
    
    # 加载数据
    logger.info(f"Loading papers from {args.papers}")
    raw_papers = load_papers(args.papers)
    logger.info(f"Loaded {len(raw_papers)} papers")
    
    logger.info(f"Loading ground truth from {args.ground_truth}")
    ground_truth = load_ground_truth(args.ground_truth) if args.ground_truth else {}
    logger.info(f"Loaded {len(ground_truth)} ground truth entries")
    
    # 准备数据
    papers = prepare_papers(raw_papers, ground_truth, args.high_impact_pct)
    
    if args.limit:
        papers = papers[:args.limit]
        logger.info(f"Limited to {len(papers)} papers")
    
    if not papers:
        logger.error("No valid papers to process")
        return
    
    # 初始化LLM客户端
    llm_client = OllamaClient(args.model, args.base_url)
    logger.info(f"Initialized LLM client: {args.model} @ {args.base_url}")
    
    # 运行实验
    logger.info("="*60)
    logger.info("Starting Optimized Mechanism Comparison Experiment")
    logger.info("="*60)
    
    results = run_experiment(papers, llm_client, args.threshold, args.mechanisms)
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
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
    
    if summary.get('best_precision_30'):
        print(f"Best Precision@30%: {summary['best_precision_30']['mechanism']} "
              f"({summary['best_precision_30']['value']:.4f})")
    
    print("\nRanking by Spearman ρ:")
    for item in summary.get('ranking_by_spearman', []):
        print(f"  {item['rank']}. {item['mechanism']}: {item['spearman_rho']:.4f}")
    
    # 打印详细表格
    print("\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)
    print(f"{'Mechanism':<25} {'Spearman ρ':>12} {'P@20%':>10} {'P@30%':>10}")
    print("-"*60)
    
    for name, data in results['mechanisms'].items():
        if isinstance(data, dict) and 'spearman_rho' in data:
            print(f"{name:<25} {data['spearman_rho']:>12.4f} "
                  f"{data.get('precision_at_20', 0):>10.4f} "
                  f"{data.get('precision_at_30', 0):>10.4f}")


if __name__ == '__main__':
    main()