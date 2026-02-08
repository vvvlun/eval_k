"""
Exp 4 Multi-LLM: Mechanism Comparison with Heterogeneous LLMs

核心改进：每个维度的Expert Agent可以使用不同的LLM，测试异构LLM是否能减轻Diversity Erosion。

假设：如果不同LLM有不同的"思维方式"，迭代讨论时可能不会那么容易趋同。

Agent配置：
- Novelty Expert: LLM_1
- Methodology Expert: LLM_2  
- Clarity Expert: LLM_3
- Empirical Expert: LLM_4
- Chair Agent (M3): LLM_chair
- Generalist Agents (M5): LLM_holistic, LLM_contrarian

使用方法：
    # 方式1：使用配置文件
    python exp4_multi_llm.py \
        --papers ./data/raw/iclr_papers \
        --ground-truth ./data/processed/ground_truth.json \
        --config ./llm_config.json \
        --output ./exp4_multi_llm_results.json

    # 方式2：命令行指定（简化版，4个expert用同一个，其他用另一个）
    python exp4_multi_llm.py \
        --papers ./data/raw/iclr_papers \
        --ground-truth ./data/processed/ground_truth.json \
        --expert-models llama3.1:70b,qwen2.5:72b,llama3.1:70b,qwen2.5:72b \
        --expert-urls http://localhost:11434,http://localhost:11435,http://localhost:11434,http://localhost:11435 \
        --output ./exp4_multi_llm_results.json

配置文件格式 (llm_config.json):
{
    "experts": {
        "novelty": {"model": "llama3.1:70b", "base_url": "http://localhost:11434"},
        "methodology": {"model": "qwen2.5:72b", "base_url": "http://localhost:11435"},
        "clarity": {"model": "llama3.1:8b", "base_url": "http://localhost:11434"},
        "empirical": {"model": "qwen2.5:32b", "base_url": "http://localhost:11435"}
    },
    "chair": {"model": "deepseek-r1:70b", "base_url": "http://localhost:11436"},
    "holistic": {"model": "llama3.1:70b", "base_url": "http://localhost:11434"},
    "contrarian": {"model": "qwen2.5:72b", "base_url": "http://localhost:11435"}
}
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
    model_used: str = ""  # 记录使用的模型


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
# 多LLM客户端管理
# ============================================================================

class OllamaClient:
    """单个Ollama客户端"""
    
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.name = f"{model}@{base_url.split('/')[-1]}"
    
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
                timeout=180
            )
            response.raise_for_status()
            
            content = response.json().get("message", {}).get("content", "")
            
            import re
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return {"score": 5.0, "confidence": 0.5}
            
        except Exception as e:
            logger.warning(f"LLM call failed ({self.name}): {e}")
            return {"score": 5.0, "confidence": 0.5}


class MultiLLMManager:
    """管理多个LLM客户端"""
    
    DIMENSIONS = ['novelty', 'methodology', 'clarity', 'empirical']
    
    def __init__(self, config: Dict):
        """
        config格式:
        {
            "experts": {
                "novelty": {"model": "...", "base_url": "..."},
                "methodology": {"model": "...", "base_url": "..."},
                ...
            },
            "chair": {"model": "...", "base_url": "..."},
            "holistic": {"model": "...", "base_url": "..."},
            "contrarian": {"model": "...", "base_url": "..."}
        }
        """
        self.config = config
        self.clients = {}
        
        # 创建expert clients
        for dim in self.DIMENSIONS:
            expert_config = config.get('experts', {}).get(dim, config.get('default', {}))
            self.clients[dim] = OllamaClient(
                model=expert_config.get('model', 'llama3.1:8b'),
                base_url=expert_config.get('base_url', 'http://localhost:11438')
            )
        
        # 创建其他clients
        for role in ['chair', 'holistic', 'contrarian']:
            role_config = config.get(role, config.get('default', {}))
            self.clients[role] = OllamaClient(
                model=role_config.get('model', 'llama3.1:8b'),
                base_url=role_config.get('base_url', 'http://localhost:11438')
            )
        
        # 打印配置
        logger.info("Multi-LLM Configuration:")
        for name, client in self.clients.items():
            logger.info(f"  {name}: {client.name}")
    
    def get_expert_client(self, dimension: str) -> OllamaClient:
        return self.clients.get(dimension, self.clients.get('novelty'))
    
    def get_chair_client(self) -> OllamaClient:
        return self.clients.get('chair', self.clients.get('novelty'))
    
    def get_holistic_client(self) -> OllamaClient:
        return self.clients.get('holistic', self.clients.get('novelty'))
    
    def get_contrarian_client(self) -> OllamaClient:
        return self.clients.get('contrarian', self.clients.get('novelty'))
    
    def get_model_diversity_info(self) -> Dict:
        """返回模型多样性信息"""
        models = set()
        urls = set()
        for client in self.clients.values():
            models.add(client.model)
            urls.add(client.base_url)
        return {
            'unique_models': list(models),
            'unique_urls': list(urls),
            'n_unique_models': len(models),
            'n_unique_urls': len(urls),
            'is_heterogeneous': len(models) > 1
        }


# ============================================================================
# Expert Agent评估（使用异构LLM）
# ============================================================================

class HeterogeneousExpertRunner:
    """使用异构LLM的Expert Agent Runner"""
    
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
    
    def __init__(self, llm_manager: MultiLLMManager):
        self.llm_manager = llm_manager
    
    def evaluate_paper(self, paper: PaperData) -> Dict[str, AgentEvaluation]:
        """评估单篇论文，每个维度使用不同的LLM"""
        results = {}
        for dim in self.DIMENSIONS:
            client = self.llm_manager.get_expert_client(dim)
            eval_result = self._evaluate_dimension(paper, dim, client)
            results[dim] = eval_result
        return results
    
    def _evaluate_dimension(self, paper: PaperData, dimension: str, 
                           client: OllamaClient) -> AgentEvaluation:
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
            response = client.generate_json(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPTS[dimension],
                temperature=0.3
            )
            
            return AgentEvaluation(
                dimension=dimension,
                score=max(1.0, min(10.0, float(response.get('score', 5.0)))),
                confidence=max(0.0, min(1.0, float(response.get('confidence', 0.7)))),
                reasoning=response.get('reasoning', ''),
                model_used=client.name
            )
        except Exception as e:
            logger.warning(f"Error evaluating {dimension} with {client.name}: {e}")
            return AgentEvaluation(
                dimension=dimension, score=5.0, confidence=0.5, model_used=client.name
            )
    
    def _format_reviews(self, reviews: List[Dict]) -> str:
        if not reviews:
            return "No reviews available."
        
        formatted = []
        for i, review in enumerate(reviews, 1):
            rating = review.get('rating', 'N/A')
            text = review.get('review_text', review.get('comment', ''))[:600]
            formatted.append(f"Review {i} (Rating: {rating}):\n{text}...")
        
        return "\n\n".join(formatted[:3])


def run_expert_evaluations(papers: List[PaperData], 
                          llm_manager: MultiLLMManager) -> List[PaperData]:
    """Step 1: 使用异构LLM进行Expert评估"""
    logger.info("="*60)
    logger.info("Step 1: Running Heterogeneous Expert Agent Evaluations")
    logger.info("="*60)
    
    expert_runner = HeterogeneousExpertRunner(llm_manager)
    start_time = time.time()
    
    for i, paper in enumerate(papers):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(papers) - i - 1)
            logger.info(f"  Expert Eval Progress: {i+1}/{len(papers)}, ETA: {eta/60:.1f} min")
        
        paper.expert_evals = expert_runner.evaluate_paper(paper)
    
    elapsed = time.time() - start_time
    logger.info(f"Step 1 completed in {elapsed/60:.1f} min")
    
    return papers


# ============================================================================
# 机制实现（使用异构LLM）
# ============================================================================

class M1_SimpleAverage:
    """M1: 简单平均"""
    
    def __init__(self, threshold: float = 5.5):
        self.name = "M1_SimpleAverage"
        self.threshold = threshold
    
    def evaluate(self, paper: PaperData) -> MechanismResult:
        evals = paper.expert_evals
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
                "models_used": {d: e.model_used for d, e in evals.items()}
            }
        )


class M2_ConfidenceWeighted:
    """M2: Confidence加权"""
    
    def __init__(self, threshold: float = 5.5):
        self.name = "M2_ConfidenceWeighted"
        self.threshold = threshold
    
    def evaluate(self, paper: PaperData) -> MechanismResult:
        evals = paper.expert_evals
        
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
                "dimension_confidences": {d: e.confidence for d, e in evals.items()},
                "models_used": {d: e.model_used for d, e in evals.items()}
            }
        )


class M3_ACCentric:
    """M3: AC-Centric（Chair使用独立LLM）"""
    
    CHAIR_SYSTEM_PROMPT = """You are an experienced Area Chair (AC) at a top AI conference.
You will see evaluations from 4 expert reviewers on different dimensions.
Your job is to make a holistic assessment considering all dimensions together."""
    
    def __init__(self, llm_manager: MultiLLMManager, threshold: float = 5.5):
        self.name = "M3_ACCentric"
        self.threshold = threshold
        self.llm_manager = llm_manager
    
    def evaluate(self, paper: PaperData) -> MechanismResult:
        evals = paper.expert_evals
        chair_client = self.llm_manager.get_chair_client()
        
        expert_summary = "\n".join([
            f"- {dim.upper()}: {e.score:.1f}/10 (confidence: {e.confidence:.2f}, model: {e.model_used})\n  Reasoning: {e.reasoning[:100]}..."
            for dim, e in evals.items()
        ])
        
        prompt = f"""As Area Chair, make a final decision on this paper.

=== Paper Title ===
{paper.title}

=== Expert Evaluations ===
{expert_summary}

Respond with a JSON object:
{{
    "score": <float 1-10>,
    "confidence": <float 0-1>,
    "decision": "<Accept or Reject>",
    "reasoning": "<your rationale>"
}}

ONLY output the JSON object."""

        try:
            response = chair_client.generate_json(
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
                "expert_scores": {d: e.score for d, e in evals.items()},
                "chair_model": chair_client.name,
                "expert_models": {d: e.model_used for d, e in evals.items()}
            }
        )


class M4_BoundedIteration:
    """M4: 有限轮迭代（每个维度保持使用自己的LLM）
    
    关键点：迭代时每个agent使用自己的LLM，测试异构LLM是否能抵抗diversity erosion
    """
    
    def __init__(self, llm_manager: MultiLLMManager, threshold: float = 5.5, max_rounds: int = 3):
        self.name = "M4_BoundedIteration"
        self.threshold = threshold
        self.llm_manager = llm_manager
        self.max_rounds = max_rounds
    
    def evaluate(self, paper: PaperData) -> MechanismResult:
        evals = paper.expert_evals
        
        # Round 0: 使用Step 1的评估
        current_scores = {d: e.score for d, e in evals.items()}
        current_confidences = {d: e.confidence for d, e in evals.items()}
        
        round_history = [{
            'round': 0,
            'scores': dict(current_scores),
            'confidences': dict(current_confidences),
            'score_std': np.std(list(current_scores.values()))  # 跟踪diversity
        }]
        
        # Rounds 1-3: 迭代更新，每个维度使用自己的LLM
        for round_num in range(1, self.max_rounds + 1):
            context = self._build_context(current_scores, round_num)
            
            new_scores = {}
            new_confidences = {}
            
            for dim in current_scores:
                client = self.llm_manager.get_expert_client(dim)
                updated = self._update_agent(dim, current_scores[dim], context, paper, client)
                new_scores[dim] = updated['score']
                new_confidences[dim] = updated['confidence']
            
            current_scores = new_scores
            current_confidences = new_confidences
            
            round_history.append({
                'round': round_num,
                'scores': dict(current_scores),
                'confidences': dict(current_confidences),
                'score_std': np.std(list(current_scores.values()))
            })
        
        # 计算diversity erosion指标
        initial_std = round_history[0]['score_std']
        final_std = round_history[-1]['score_std']
        diversity_erosion = (initial_std - final_std) / initial_std if initial_std > 0 else 0
        
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
                "round_history": round_history,
                "diversity_erosion": diversity_erosion,
                "initial_std": initial_std,
                "final_std": final_std,
                "models_used": {d: self.llm_manager.get_expert_client(d).name 
                               for d in current_scores}
            }
        )
    
    def _build_context(self, scores: Dict[str, float], round_num: int) -> str:
        context = f"=== Discussion Round {round_num} ===\n"
        context += "Current expert evaluations:\n"
        for dim, score in scores.items():
            context += f"  {dim.capitalize()}: {score:.1f}/10\n"
        return context
    
    def _update_agent(self, dimension: str, current_score: float, 
                      context: str, paper: PaperData, client: OllamaClient) -> Dict:
        prompt = f"""You are the {dimension.upper()} expert in a peer review discussion.

{context}

Your current score for {dimension}: {current_score:.1f}/10

Paper: {paper.title}

After seeing other experts' scores, reconsider your evaluation.
You may adjust your score based on the discussion, but maintain your unique perspective on {dimension}.
Don't simply converge to the average - your specialized view matters.

Respond with JSON:
{{"score": <float 1-10>, "confidence": <float 0-1>}}"""

        try:
            response = client.generate_json(
                prompt=prompt,
                system_prompt=f"You are an expert reviewer for {dimension}. Maintain your unique perspective.",
                temperature=0.3
            )
            return {
                'score': max(1.0, min(10.0, float(response.get('score', current_score)))),
                'confidence': max(0.0, min(1.0, float(response.get('confidence', 0.7))))
            }
        except:
            return {'score': current_score, 'confidence': 0.7}


class M5_DiverseEnsemble:
    """M5: 扩展Ensemble（generalist使用不同LLM）"""
    
    def __init__(self, llm_manager: MultiLLMManager, threshold: float = 5.5):
        self.name = "M5_DiverseEnsemble"
        self.threshold = threshold
        self.llm_manager = llm_manager
    
    def evaluate(self, paper: PaperData) -> MechanismResult:
        evals = paper.expert_evals
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
                "expert_models": {d: e.model_used for d, e in evals.items()},
                "generalist_models": {
                    'holistic': self.llm_manager.get_holistic_client().name,
                    'contrarian': self.llm_manager.get_contrarian_client().name
                }
            }
        )
    
    def _run_generalists(self, paper: PaperData, expert_evals: Dict) -> Dict:
        expert_summary = "\n".join([
            f"- {dim.upper()}: {e.score:.1f}/10"
            for dim, e in expert_evals.items()
        ])
        
        results = {}
        
        # Holistic Agent
        holistic_client = self.llm_manager.get_holistic_client()
        holistic_prompt = f"""As a holistic reviewer, evaluate this paper considering all aspects together.

Paper: {paper.title}

Expert Evaluations:
{expert_summary}

Provide your overall assessment:
{{"score": <float 1-10>, "confidence": <float 0-1>}}"""

        try:
            response = holistic_client.generate_json(
                prompt=holistic_prompt,
                system_prompt="You are a holistic reviewer who considers overall paper quality.",
                temperature=0.3
            )
            results['holistic'] = {
                'score': max(1.0, min(10.0, float(response.get('score', 5.0)))),
                'confidence': max(0.0, min(1.0, float(response.get('confidence', 0.6)))),
                'model': holistic_client.name
            }
        except:
            avg = sum(e.score for e in expert_evals.values()) / len(expert_evals)
            results['holistic'] = {'score': avg, 'confidence': 0.5, 'model': holistic_client.name}
        
        # Contrarian Agent
        contrarian_client = self.llm_manager.get_contrarian_client()
        avg_score = sum(e.score for e in expert_evals.values()) / len(expert_evals)
        contrarian_prompt = f"""As a contrarian reviewer, critically examine whether the consensus is correct.

Paper: {paper.title}

Expert Evaluations:
{expert_summary}

Average score: {avg_score:.1f}/10

Challenge the consensus if appropriate. Are there overlooked strengths or weaknesses?
{{"score": <float 1-10>, "confidence": <float 0-1>}}"""

        try:
            response = contrarian_client.generate_json(
                prompt=contrarian_prompt,
                system_prompt="You are a contrarian reviewer who challenges consensus views.",
                temperature=0.5
            )
            results['contrarian'] = {
                'score': max(1.0, min(10.0, float(response.get('score', 5.0)))),
                'confidence': max(0.0, min(1.0, float(response.get('confidence', 0.5)))),
                'model': contrarian_client.name
            }
        except:
            contrarian_score = avg_score + (1.0 if avg_score < 5.5 else -1.0)
            results['contrarian'] = {
                'score': contrarian_score, 'confidence': 0.4, 'model': contrarian_client.name
            }
        
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


def compute_diversity_erosion_stats(m4_results: List[MechanismResult]) -> Dict:
    """计算M4的diversity erosion统计"""
    erosions = []
    initial_stds = []
    final_stds = []
    
    for r in m4_results:
        meta = r.metadata
        if 'diversity_erosion' in meta:
            erosions.append(meta['diversity_erosion'])
            initial_stds.append(meta.get('initial_std', 0))
            final_stds.append(meta.get('final_std', 0))
    
    if not erosions:
        return {}
    
    return {
        'mean_diversity_erosion': np.mean(erosions),
        'std_diversity_erosion': np.std(erosions),
        'mean_initial_std': np.mean(initial_stds),
        'mean_final_std': np.mean(final_stds),
        'pct_papers_with_erosion': sum(1 for e in erosions if e > 0) / len(erosions)
    }


# ============================================================================
# 数据加载
# ============================================================================

def load_papers(file_path: str) -> List[Dict]:
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
                elif isinstance(data, dict) and 'papers' in data:
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
    all_citations = []
    for p in raw_papers:
        pid = p.get('paper_id', '')
        gt = ground_truth.get(pid, {})
        citations = gt.get('citations', p.get('citations', 0))
        if citations > 0:
            all_citations.append(citations)
    
    threshold = np.percentile(all_citations, 100 - high_impact_pct) if all_citations else 0
    logger.info(f"High impact threshold: {threshold:.0f} citations")
    
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
    
    logger.info(f"Prepared {len(papers)} papers, {sum(p.is_high_impact for p in papers)} high impact")
    
    return papers


def load_llm_config(config_path: str) -> Dict:
    """加载LLM配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)


def build_config_from_args(args) -> Dict:
    """从命令行参数构建配置"""
    config = {'experts': {}, 'default': {}}
    
    dimensions = ['novelty', 'methodology', 'clarity', 'empirical']
    
    if args.expert_models and args.expert_urls:
        models = args.expert_models.split(',')
        urls = args.expert_urls.split(',')
        
        for i, dim in enumerate(dimensions):
            config['experts'][dim] = {
                'model': models[i % len(models)],
                'base_url': urls[i % len(urls)]
            }
    else:
        # 使用默认值
        for dim in dimensions:
            config['experts'][dim] = {
                'model': args.model,
                'base_url': args.base_url
            }
    
    # Chair配置
    config['chair'] = {
        'model': args.chair_model or args.model,
        'base_url': args.chair_url or args.base_url
    }
    
    # Generalist配置
    config['holistic'] = {
        'model': args.holistic_model or args.model,
        'base_url': args.holistic_url or args.base_url
    }
    config['contrarian'] = {
        'model': args.contrarian_model or args.model,
        'base_url': args.contrarian_url or args.base_url
    }
    
    return config


# ============================================================================
# 主实验流程
# ============================================================================

def save_checkpoint(results: Dict, output_path: Path, stage: str):
    """保存检查点"""
    checkpoint_path = output_path.parent / f"{output_path.stem}_checkpoint_{stage}.json"
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(results), f, indent=2, ensure_ascii=False)
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # 同时更新主文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(results), f, indent=2, ensure_ascii=False)
    logger.info(f"Main results updated: {output_path}")


def load_checkpoint(output_path: Path) -> Tuple[Optional[Dict], Optional[Dict]]:
    """加载已有的检查点"""
    # 尝试加载主文件
    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"Loaded existing results from {output_path}")
            return results, results.get('expert_evaluations')
        except Exception as e:
            logger.warning(f"Failed to load {output_path}: {e}")
    
    # 尝试加载最新的checkpoint
    checkpoint_files = sorted(output_path.parent.glob(f"{output_path.stem}_checkpoint_*.json"))
    if checkpoint_files:
        latest = checkpoint_files[-1]
        try:
            with open(latest, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"Loaded checkpoint from {latest}")
            return results, results.get('expert_evaluations')
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {latest}: {e}")
    
    return None, None


def restore_expert_evals(papers: List[PaperData], saved_evals: Dict) -> bool:
    """从保存的结果恢复expert评估"""
    if not saved_evals:
        return False
    
    restored_count = 0
    for paper in papers:
        if paper.paper_id in saved_evals:
            evals_data = saved_evals[paper.paper_id]
            paper.expert_evals = {
                dim: AgentEvaluation(
                    dimension=dim,
                    score=data['score'],
                    confidence=data['confidence'],
                    reasoning=data.get('reasoning', ''),
                    model_used=data.get('model_used', '')
                )
                for dim, data in evals_data.items()
            }
            restored_count += 1
    
    logger.info(f"Restored expert evaluations for {restored_count}/{len(papers)} papers")
    return restored_count == len(papers)


def run_experiment(papers: List[PaperData], llm_manager: MultiLLMManager,
                   threshold: float = 5.5,
                   mechanisms_to_run: List[str] = None,
                   output_path: Path = None,
                   resume: bool = True) -> Dict[str, Any]:
    
    all_mechanisms = ['M1', 'M2', 'M3', 'M4', 'M5']
    if mechanisms_to_run:
        all_mechanisms = [m for m in all_mechanisms if m in mechanisms_to_run]
    
    # 尝试恢复已有结果
    existing_results = None
    saved_expert_evals = None
    completed_mechanisms = set()
    
    if resume and output_path:
        existing_results, saved_expert_evals = load_checkpoint(output_path)
        if existing_results:
            completed_mechanisms = set(existing_results.get('mechanisms', {}).keys())
            completed_mechanisms.discard('Actual_Decisions')  # 这个每次都重新算
            logger.info(f"Already completed mechanisms: {completed_mechanisms}")
    
    results = existing_results or {
        'config': {
            'threshold': threshold,
            'n_papers': len(papers),
            'mechanisms': all_mechanisms,
            'llm_diversity': llm_manager.get_model_diversity_info(),
            'timestamp': datetime.now().isoformat()
        },
        'mechanisms': {},
        'expert_evaluations': {},
        'diversity_analysis': {},
        'summary': {}
    }
    
    # Step 1: Expert评估（如果已有则跳过）
    if saved_expert_evals and restore_expert_evals(papers, saved_expert_evals):
        logger.info("="*60)
        logger.info("Step 1: SKIPPED - Using restored expert evaluations")
        logger.info("="*60)
    else:
        papers = run_expert_evaluations(papers, llm_manager)
        
        # 保存expert评估结果
        for paper in papers:
            results['expert_evaluations'][paper.paper_id] = {
                d: {'score': e.score, 'confidence': e.confidence, 
                    'reasoning': e.reasoning, 'model_used': e.model_used}
                for d, e in paper.expert_evals.items()
            }
        
        # 保存Step 1检查点
        if output_path:
            save_checkpoint(results, output_path, "step1_experts")
    
    # M1
    if 'M1' in all_mechanisms:
        if 'M1_SimpleAverage' in completed_mechanisms:
            logger.info("="*50)
            logger.info("M1 (Simple Average) - SKIPPED (already completed)")
        else:
            logger.info("="*50)
            logger.info("Computing M1 (Simple Average)")
            m1 = M1_SimpleAverage(threshold)
            m1_results = [m1.evaluate(p) for p in papers]
            results['mechanisms']['M1_SimpleAverage'] = evaluate_mechanism(
                'M1_SimpleAverage', m1_results, papers
            ).to_dict()
            logger.info(f"  M1 Spearman ρ: {results['mechanisms']['M1_SimpleAverage']['spearman_rho']:.4f}")
            if output_path:
                save_checkpoint(results, output_path, "M1")
    
    # M2
    if 'M2' in all_mechanisms:
        if 'M2_ConfidenceWeighted' in completed_mechanisms:
            logger.info("="*50)
            logger.info("M2 (Confidence-Weighted) - SKIPPED (already completed)")
        else:
            logger.info("="*50)
            logger.info("Computing M2 (Confidence-Weighted)")
            m2 = M2_ConfidenceWeighted(threshold)
            m2_results = [m2.evaluate(p) for p in papers]
            results['mechanisms']['M2_ConfidenceWeighted'] = evaluate_mechanism(
                'M2_ConfidenceWeighted', m2_results, papers
            ).to_dict()
            logger.info(f"  M2 Spearman ρ: {results['mechanisms']['M2_ConfidenceWeighted']['spearman_rho']:.4f}")
            if output_path:
                save_checkpoint(results, output_path, "M2")
    
    # M3
    if 'M3' in all_mechanisms:
        if 'M3_ACCentric' in completed_mechanisms:
            logger.info("="*50)
            logger.info("M3 (AC-Centric) - SKIPPED (already completed)")
        else:
            logger.info("="*50)
            logger.info("Computing M3 (AC-Centric)")
            m3 = M3_ACCentric(llm_manager, threshold)
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
            if output_path:
                save_checkpoint(results, output_path, "M3")
    
    # M4
    if 'M4' in all_mechanisms:
        if 'M4_BoundedIteration' in completed_mechanisms:
            logger.info("="*50)
            logger.info("M4 (Bounded Iteration) - SKIPPED (already completed)")
        else:
            logger.info("="*50)
            logger.info("Computing M4 (Bounded Iteration with Heterogeneous LLMs)")
            m4 = M4_BoundedIteration(llm_manager, threshold, max_rounds=3)
            m4_results = []
            start_time = time.time()
            
            # M4每100篇保存一次中间结果
            for i, paper in enumerate(papers):
                if (i + 1) % 20 == 0:
                    elapsed = time.time() - start_time
                    eta = elapsed / (i + 1) * (len(papers) - i - 1)
                    logger.info(f"  M4 Progress: {i+1}/{len(papers)}, ETA: {eta/60:.1f} min")
                
                m4_results.append(m4.evaluate(paper))
                
                # 每500篇保存一次中间进度
                if output_path and (i + 1) % 500 == 0:
                    # 计算当前进度的metrics
                    partial_metrics = evaluate_mechanism(
                        'M4_BoundedIteration', m4_results, papers[:i+1]
                    ).to_dict()
                    partial_metrics['_partial'] = True
                    partial_metrics['_progress'] = f"{i+1}/{len(papers)}"
                    results['mechanisms']['M4_BoundedIteration_partial'] = partial_metrics
                    save_checkpoint(results, output_path, f"M4_progress_{i+1}")
                    logger.info(f"  M4 intermediate save at {i+1} papers")
            
            # 清理partial结果
            results['mechanisms'].pop('M4_BoundedIteration_partial', None)
            
            results['mechanisms']['M4_BoundedIteration'] = evaluate_mechanism(
                'M4_BoundedIteration', m4_results, papers
            ).to_dict()
            
            # 计算diversity erosion统计
            results['diversity_analysis']['M4'] = compute_diversity_erosion_stats(m4_results)
            
            logger.info(f"  M4 Spearman ρ: {results['mechanisms']['M4_BoundedIteration']['spearman_rho']:.4f}")
            if results['diversity_analysis']['M4']:
                logger.info(f"  M4 Mean Diversity Erosion: {results['diversity_analysis']['M4']['mean_diversity_erosion']:.2%}")
            if output_path:
                save_checkpoint(results, output_path, "M4")
    
    # M5
    if 'M5' in all_mechanisms:
        if 'M5_DiverseEnsemble' in completed_mechanisms:
            logger.info("="*50)
            logger.info("M5 (Diverse Ensemble) - SKIPPED (already completed)")
        else:
            logger.info("="*50)
            logger.info("Computing M5 (Diverse Ensemble)")
            m5 = M5_DiverseEnsemble(llm_manager, threshold)
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
            if output_path:
                save_checkpoint(results, output_path, "M5")
    
    # Actual Decisions
    logger.info("="*50)
    logger.info("Computing Actual Decisions baseline")
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
    
    # Summary
    results['summary'] = generate_summary(results['mechanisms'])
    
    return results


def generate_summary(mechanisms: Dict) -> Dict:
    valid = {k: v for k, v in mechanisms.items() 
             if k != 'Actual_Decisions' and isinstance(v, dict) and 'spearman_rho' in v}
    
    if not valid:
        return {}
    
    best_rho = max(valid.items(), key=lambda x: x[1].get('spearman_rho', 0))
    ranking = sorted(valid.items(), key=lambda x: x[1].get('spearman_rho', 0), reverse=True)
    
    return {
        'best_spearman': {'mechanism': best_rho[0], 'value': best_rho[1]['spearman_rho']},
        'ranking_by_spearman': [
            {'rank': i+1, 'mechanism': name, 'spearman_rho': data['spearman_rho']}
            for i, (name, data) in enumerate(ranking)
        ]
    }


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Exp 4 Multi-LLM: Mechanism comparison with heterogeneous LLMs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用配置文件
  python exp4_multi_llm.py -p ./data/papers -g ./data/gt.json --config llm_config.json

  # 命令行指定不同模型
  python exp4_multi_llm.py -p ./data/papers -g ./data/gt.json \\
      --expert-models llama3.1:70b,qwen2.5:72b,llama3.1:70b,qwen2.5:72b \\
      --expert-urls http://localhost:11434,http://localhost:11435,http://localhost:11434,http://localhost:11435

  # 简单模式：所有expert用一个模型，chair用另一个
  python exp4_multi_llm.py -p ./data/papers -g ./data/gt.json \\
      --model llama3.1:70b --base-url http://localhost:11434 \\
      --chair-model qwen2.5:72b --chair-url http://localhost:11435
        """
    )
    
    # 数据参数
    parser.add_argument('--papers', '-p', required=True, help='论文数据')
    parser.add_argument('--ground-truth', '-g', default=None, help='Ground Truth数据')
    parser.add_argument('--output', '-o', default='./exp4_multi_llm_results.json', help='输出文件')
    
    # LLM配置 - 方式1：配置文件
    parser.add_argument('--config', '-c', default=None, help='LLM配置文件路径')
    
    # LLM配置 - 方式2：命令行参数
    parser.add_argument('--model', '-m', default='llama3.1:8b', help='默认模型')
    parser.add_argument('--base-url', default='http://localhost:11438', help='默认Ollama URL')
    
    parser.add_argument('--expert-models', default=None, 
                       help='Expert模型，逗号分隔：novelty,methodology,clarity,empirical')
    parser.add_argument('--expert-urls', default=None,
                       help='Expert URLs，逗号分隔')
    
    parser.add_argument('--chair-model', default=None, help='Chair模型')
    parser.add_argument('--chair-url', default=None, help='Chair URL')
    
    parser.add_argument('--holistic-model', default=None, help='Holistic Agent模型')
    parser.add_argument('--holistic-url', default=None, help='Holistic Agent URL')
    
    parser.add_argument('--contrarian-model', default=None, help='Contrarian Agent模型')
    parser.add_argument('--contrarian-url', default=None, help='Contrarian Agent URL')
    
    # 实验参数
    parser.add_argument('--threshold', type=float, default=5.5, help='Accept阈值')
    parser.add_argument('--high-impact-pct', type=float, default=30.0, help='高影响力百分位')
    parser.add_argument('--mechanisms', nargs='+', default=None, help='要运行的机制')
    parser.add_argument('--limit', type=int, default=None, help='限制论文数')
    parser.add_argument('--no-resume', action='store_true', help='不从检查点恢复，重新开始')
    
    args = parser.parse_args()
    
    # 加载数据
    logger.info(f"Loading papers from {args.papers}")
    raw_papers = load_papers(args.papers)
    logger.info(f"Loaded {len(raw_papers)} papers")
    
    ground_truth = load_ground_truth(args.ground_truth) if args.ground_truth else {}
    logger.info(f"Loaded {len(ground_truth)} ground truth entries")
    
    papers = prepare_papers(raw_papers, ground_truth, args.high_impact_pct)
    
    if args.limit:
        papers = papers[:args.limit]
        logger.info(f"Limited to {len(papers)} papers")
    
    if not papers:
        logger.error("No valid papers")
        return
    
    # 构建LLM配置
    if args.config:
        config = load_llm_config(args.config)
    else:
        config = build_config_from_args(args)
    
    # 创建LLM管理器
    llm_manager = MultiLLMManager(config)
    
    # 运行实验
    logger.info("="*60)
    logger.info("Starting Multi-LLM Mechanism Comparison")
    logger.info("="*60)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = run_experiment(
        papers, llm_manager, args.threshold, args.mechanisms,
        output_path=output_path,
        resume=not args.no_resume
    )
    
    # 最终保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(results), f, indent=2, ensure_ascii=False)
    
    logger.info(f"Final results saved to {output_path}")
    
    # 打印结果
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"\nLLM Configuration:")
    diversity_info = results['config']['llm_diversity']
    print(f"  Unique models: {diversity_info['n_unique_models']}")
    print(f"  Models: {', '.join(diversity_info['unique_models'])}")
    print(f"  Heterogeneous: {diversity_info['is_heterogeneous']}")
    
    print(f"\nRanking by Spearman ρ:")
    for item in results['summary'].get('ranking_by_spearman', []):
        print(f"  {item['rank']}. {item['mechanism']}: {item['spearman_rho']:.4f}")
    
    if 'M4' in results.get('diversity_analysis', {}):
        da = results['diversity_analysis']['M4']
        print(f"\nM4 Diversity Analysis:")
        print(f"  Mean Diversity Erosion: {da['mean_diversity_erosion']:.2%}")
        print(f"  Initial Score STD: {da['mean_initial_std']:.3f}")
        print(f"  Final Score STD: {da['mean_final_std']:.3f}")
        print(f"  Papers with Erosion: {da['pct_papers_with_erosion']:.1%}")
    
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