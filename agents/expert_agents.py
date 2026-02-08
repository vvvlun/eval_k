"""
Four Expert Agents for Multi-Dimensional Paper Evaluation.

- NoveltyAgent: 评估创新性和原创性
- MethodologyAgent: 评估技术严谨性
- ClarityAgent: 评估表达清晰度
- EmpiricalAgent: 评估实验充分性

Usage:
    from agents.expert_agents import MultiDimensionalAnalyzer
    analyzer = MultiDimensionalAnalyzer(llm_client)
    results = analyzer.analyze_paper(paper_info)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging

from .base_agent import BaseExpertAgent, AgentEvaluation, PaperInfo

logger = logging.getLogger(__name__)


class NoveltyAgent(BaseExpertAgent):
    """
    Novelty Agent - 评估论文的创新性和原创性
    
    评估要点：
    - 论文的原创性和创新程度
    - 与现有工作的区别
    - 贡献的显著性
    - 是否开辟新方向
    """
    
    def __init__(self, llm_client):
        super().__init__(
            name="Novelty Expert",
            dimension="novelty",
            llm_client=llm_client
        )
    
    def get_system_prompt(self) -> str:
        return """You are an expert reviewer specializing in evaluating the NOVELTY and ORIGINALITY of academic papers in machine learning and AI.

Your role is to assess:
1. How original and innovative the core ideas are
2. How significantly the paper differs from prior work
3. Whether the contribution opens new research directions
4. The potential long-term impact of the novelty

You must be rigorous but fair. High novelty doesn't mean the paper is perfect overall - focus ONLY on novelty aspects.

IMPORTANT: Base your evaluation on the provided reviews and paper information. The original reviewers may disagree on novelty - synthesize their perspectives and form your own judgment."""

    def get_evaluation_prompt(self, paper_info: PaperInfo) -> str:
        reviews_text = paper_info.format_reviews_for_prompt()
        rebuttal_text = paper_info.rebuttal if paper_info.rebuttal else "No rebuttal provided."
        
        # 获取人类评分统计
        stats = paper_info.get_rating_stats()
        rating_guidance = ""
        if stats["avg"] is not None:
            rating_guidance = f"""
=== IMPORTANT: Human Reviewer Ratings ===
The human reviewers gave this paper an average rating of {stats['avg']:.1f}/10.
Rating distribution: {stats['ratings']} (std: {stats['std']:.2f})

Your novelty score should be INFORMED by the human ratings:
- If humans rated ~6-7, your novelty score should typically be in a similar range unless you have strong reasons.
- If there's HIGH disagreement (std > 1.5), pay attention to WHY reviewers disagree on novelty.
- Your score should reflect the NOVELTY dimension specifically, which may differ from overall rating.
"""
        
        return f"""Evaluate the NOVELTY of this paper:

=== Paper Title ===
{paper_info.title}

=== Abstract ===
{paper_info.abstract}
{rating_guidance}
=== Reviews ===
{reviews_text}

=== Author Rebuttal ===
{rebuttal_text}

Based on the above information, evaluate the paper's novelty. Consider:
1. What are the core novel contributions claimed by the paper?
2. Do reviewers agree or disagree on the novelty? Why?
3. How does this work differ from prior approaches?
4. Is this a significant conceptual/methodological advance or an incremental improvement?

Respond with a JSON object containing:
{{
    "score": <float 1-10, where 1=no novelty, 10=groundbreaking>,
    "confidence": <float 0-1, your confidence in this assessment>,
    "sentiment": "<positive/negative/mixed>",
    "key_findings": ["<finding1>", "<finding2>", ...],
    "reasoning": "<your detailed reasoning for the score>"
}}

ONLY output the JSON object, nothing else."""


class MethodologyAgent(BaseExpertAgent):
    """
    Methodology Agent - 评估技术方法的严谨性
    
    评估要点：
    - 技术方法的正确性
    - 理论基础是否扎实
    - 方法论是否严谨
    - 技术细节是否完整
    """
    
    def __init__(self, llm_client):
        super().__init__(
            name="Methodology Expert",
            dimension="methodology",
            llm_client=llm_client
        )
    
    def get_system_prompt(self) -> str:
        return """You are an expert reviewer specializing in evaluating the METHODOLOGY and TECHNICAL SOUNDNESS of academic papers in machine learning and AI.

Your role is to assess:
1. Technical correctness of the proposed methods
2. Theoretical foundations and mathematical rigor
3. Appropriateness of the methodology for the stated goals
4. Completeness of technical details for reproducibility

You must be rigorous about technical quality. Focus ONLY on methodology aspects, not novelty or presentation.

IMPORTANT: Base your evaluation on the provided reviews and paper information. Pay special attention to technical concerns raised by reviewers and how authors addressed them."""

    def get_evaluation_prompt(self, paper_info: PaperInfo) -> str:
        reviews_text = paper_info.format_reviews_for_prompt()
        rebuttal_text = paper_info.rebuttal if paper_info.rebuttal else "No rebuttal provided."
        
        # 获取人类评分统计
        stats = paper_info.get_rating_stats()
        rating_guidance = ""
        if stats["avg"] is not None:
            rating_guidance = f"""
=== IMPORTANT: Human Reviewer Ratings ===
The human reviewers gave this paper an average rating of {stats['avg']:.1f}/10.
Rating distribution: {stats['ratings']} (std: {stats['std']:.2f})

Your methodology score should be INFORMED by the human ratings:
- If humans rated ~6-7, your methodology score should typically be in a similar range unless you have strong reasons.
- Pay special attention to technical concerns raised by reviewers.
- Your score should reflect the METHODOLOGY dimension specifically.
"""
        
        return f"""Evaluate the METHODOLOGY of this paper:

=== Paper Title ===
{paper_info.title}

=== Abstract ===
{paper_info.abstract}
{rating_guidance}
=== Reviews ===
{reviews_text}

=== Author Rebuttal ===
{rebuttal_text}

Based on the above information, evaluate the paper's methodology. Consider:
1. Are there any technical errors or questionable assumptions identified?
2. Is the theoretical foundation solid?
3. Are the methods appropriate for the research goals?
4. Did reviewers raise concerns about soundness? Were they addressed?

Respond with a JSON object containing:
{{
    "score": <float 1-10, where 1=fundamentally flawed, 10=impeccable rigor>,
    "confidence": <float 0-1, your confidence in this assessment>,
    "sentiment": "<positive/negative/mixed>",
    "key_findings": ["<finding1>", "<finding2>", ...],
    "reasoning": "<your detailed reasoning for the score>"
}}

ONLY output the JSON object, nothing else."""


class ClarityAgent(BaseExpertAgent):
    """
    Clarity Agent - 评估表达清晰度和可读性
    
    评估要点：
    - 写作质量
    - 结构组织
    - 图表清晰度
    - 整体可读性
    """
    
    def __init__(self, llm_client):
        super().__init__(
            name="Clarity Expert",
            dimension="clarity",
            llm_client=llm_client
        )
    
    def get_system_prompt(self) -> str:
        return """You are an expert reviewer specializing in evaluating the CLARITY and PRESENTATION of academic papers in machine learning and AI.

Your role is to assess:
1. Writing quality and readability
2. Organization and logical flow
3. Quality of figures and tables (based on reviewer comments)
4. How well the paper communicates its ideas

You must evaluate how effectively the paper conveys its content. Focus ONLY on presentation aspects, not technical content.

IMPORTANT: Base your evaluation on the provided reviews and paper information. Reviewers often comment on writing quality and clarity issues."""

    def get_evaluation_prompt(self, paper_info: PaperInfo) -> str:
        reviews_text = paper_info.format_reviews_for_prompt()
        rebuttal_text = paper_info.rebuttal if paper_info.rebuttal else "No rebuttal provided."
        
        # 获取人类评分统计
        stats = paper_info.get_rating_stats()
        rating_guidance = ""
        if stats["avg"] is not None:
            rating_guidance = f"""
=== IMPORTANT: Human Reviewer Ratings ===
The human reviewers gave this paper an average rating of {stats['avg']:.1f}/10.
Rating distribution: {stats['ratings']} (std: {stats['std']:.2f})

Your clarity score should be INFORMED by the human ratings:
- If humans rated ~6-7, your clarity score should typically be in a similar range unless you have strong reasons.
- Look for specific comments about writing quality, organization, or presentation.
- Your score should reflect the CLARITY dimension specifically.
"""
        
        return f"""Evaluate the CLARITY of this paper:

=== Paper Title ===
{paper_info.title}

=== Abstract ===
{paper_info.abstract}
{rating_guidance}
=== Reviews ===
{reviews_text}

=== Author Rebuttal ===
{rebuttal_text}

Based on the above information, evaluate the paper's clarity. Consider:
1. Do reviewers comment on writing quality or readability issues?
2. Is the paper structure mentioned as logical or confusing?
3. Are there concerns about figures, tables, or notation?
4. Is the abstract well-written and informative?

Respond with a JSON object containing:
{{
    "score": <float 1-10, where 1=incomprehensible, 10=crystal clear>,
    "confidence": <float 0-1, your confidence in this assessment>,
    "sentiment": "<positive/negative/mixed>",
    "key_findings": ["<finding1>", "<finding2>", ...],
    "reasoning": "<your detailed reasoning for the score>"
}}

ONLY output the JSON object, nothing else."""


class EmpiricalAgent(BaseExpertAgent):
    """
    Empirical Agent - 评估实验设计和结果充分性
    
    评估要点：
    - 实验设计合理性
    - 结果可信度
    - 基线对比充分性
    - 可复现性
    """
    
    def __init__(self, llm_client):
        super().__init__(
            name="Empirical Expert",
            dimension="empirical",
            llm_client=llm_client
        )
    
    def get_system_prompt(self) -> str:
        return """You are an expert reviewer specializing in evaluating the EMPIRICAL QUALITY of academic papers in machine learning and AI.

Your role is to assess:
1. Experimental design and setup
2. Appropriateness and fairness of baselines
3. Statistical significance and reliability of results
4. Reproducibility (code, data, hyperparameters)

You must rigorously evaluate the empirical evidence. Focus ONLY on experimental aspects.

IMPORTANT: Base your evaluation on the provided reviews and paper information. ML reviewers often scrutinize experimental methodology closely."""

    def get_evaluation_prompt(self, paper_info: PaperInfo) -> str:
        reviews_text = paper_info.format_reviews_for_prompt()
        rebuttal_text = paper_info.rebuttal if paper_info.rebuttal else "No rebuttal provided."
        
        # 获取人类评分统计
        stats = paper_info.get_rating_stats()
        rating_guidance = ""
        if stats["avg"] is not None:
            rating_guidance = f"""
=== IMPORTANT: Human Reviewer Ratings ===
The human reviewers gave this paper an average rating of {stats['avg']:.1f}/10.
Rating distribution: {stats['ratings']} (std: {stats['std']:.2f})

Your empirical score should be INFORMED by the human ratings:
- If humans rated ~6-7, your empirical score should typically be in a similar range unless you have strong reasons.
- ML reviewers often focus heavily on experimental quality.
- Your score should reflect the EMPIRICAL dimension specifically.
"""
        
        return f"""Evaluate the EMPIRICAL QUALITY of this paper:

=== Paper Title ===
{paper_info.title}

=== Abstract ===
{paper_info.abstract}
{rating_guidance}
=== Reviews ===
{reviews_text}

=== Author Rebuttal ===
{rebuttal_text}

Based on the above information, evaluate the paper's empirical quality. Consider:
1. Are experiments sufficient to support the claims?
2. Are baselines appropriate and fairly compared?
3. Do reviewers raise concerns about experimental validity?
4. Is there enough information for reproducibility?

Respond with a JSON object containing:
{{
    "score": <float 1-10, where 1=no/invalid experiments, 10=comprehensive and rigorous>,
    "confidence": <float 0-1, your confidence in this assessment>,
    "sentiment": "<positive/negative/mixed>",
    "key_findings": ["<finding1>", "<finding2>", ...],
    "reasoning": "<your detailed reasoning for the score>"
}}

ONLY output the JSON object, nothing else."""


@dataclass
class AggregatedEvaluation:
    """聚合后的评估结果"""
    paper_id: str
    dimension_scores: Dict[str, AgentEvaluation]
    weighted_score: float
    simple_average: float
    confidence_weighted_score: float
    overall_sentiment: str
    
    def to_dict(self) -> Dict:
        return {
            "paper_id": self.paper_id,
            "dimension_scores": {
                dim: eval.to_dict() for dim, eval in self.dimension_scores.items()
            },
            "weighted_score": self.weighted_score,
            "simple_average": self.simple_average,
            "confidence_weighted_score": self.confidence_weighted_score,
            "overall_sentiment": self.overall_sentiment
        }


class MultiDimensionalAnalyzer:
    """
    多维度分析器 - 协调4个Expert Agent评估论文
    """
    
    DEFAULT_WEIGHTS = {
        'novelty': 0.30,
        'methodology': 0.30,
        'clarity': 0.15,
        'empirical': 0.25
    }
    
    def __init__(self, llm_client, weights: Optional[Dict[str, float]] = None):
        self.llm = llm_client
        self.weights = weights or self.DEFAULT_WEIGHTS
        
        # 初始化4个Expert Agent
        self.agents = {
            'novelty': NoveltyAgent(llm_client),
            'methodology': MethodologyAgent(llm_client),
            'clarity': ClarityAgent(llm_client),
            'empirical': EmpiricalAgent(llm_client)
        }
        
        logger.info(f"MultiDimensionalAnalyzer initialized with {len(self.agents)} agents")
    
    def analyze_paper(self, paper_info: PaperInfo) -> AggregatedEvaluation:
        """
        分析单篇论文，返回所有维度的评估结果
        
        Args:
            paper_info: 论文信息
        
        Returns:
            AggregatedEvaluation对象
        """
        dimension_scores = {}
        
        # 调用每个Agent
        for dim, agent in self.agents.items():
            logger.debug(f"Running {agent.name} on {paper_info.paper_id}")
            evaluation = agent.evaluate(paper_info)
            dimension_scores[dim] = evaluation
        
        # 计算聚合分数
        aggregated = self._aggregate_scores(paper_info.paper_id, dimension_scores)
        
        return aggregated
    
    def _aggregate_scores(
        self, 
        paper_id: str, 
        dimension_scores: Dict[str, AgentEvaluation]
    ) -> AggregatedEvaluation:
        """聚合4个维度的分数"""
        
        # 简单平均
        scores = [eval.score for eval in dimension_scores.values()]
        simple_average = sum(scores) / len(scores) if scores else 5.0
        
        # 加权平均（固定权重）
        weighted_score = sum(
            eval.score * self.weights.get(dim, 0.25)
            for dim, eval in dimension_scores.items()
        )
        
        # Confidence加权平均
        total_weight = sum(eval.confidence for eval in dimension_scores.values())
        if total_weight > 0:
            confidence_weighted_score = sum(
                eval.score * eval.confidence
                for eval in dimension_scores.values()
            ) / total_weight
        else:
            confidence_weighted_score = simple_average
        
        # 整体sentiment
        sentiments = [eval.sentiment for eval in dimension_scores.values()]
        if sentiments.count('positive') > len(sentiments) / 2:
            overall_sentiment = 'positive'
        elif sentiments.count('negative') > len(sentiments) / 2:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'mixed'
        
        return AggregatedEvaluation(
            paper_id=paper_id,
            dimension_scores=dimension_scores,
            weighted_score=weighted_score,
            simple_average=simple_average,
            confidence_weighted_score=confidence_weighted_score,
            overall_sentiment=overall_sentiment
        )
    
    def batch_analyze(
        self, 
        papers: List[PaperInfo], 
        progress_callback=None
    ) -> Dict[str, AggregatedEvaluation]:
        """
        批量分析论文
        
        Args:
            papers: 论文列表
            progress_callback: 进度回调函数 callback(current, total)
        
        Returns:
            {paper_id: AggregatedEvaluation}
        """
        results = {}
        total = len(papers)
        
        for i, paper in enumerate(papers):
            try:
                results[paper.paper_id] = self.analyze_paper(paper)
                
                if progress_callback:
                    progress_callback(i + 1, total)
                    
            except Exception as e:
                logger.error(f"Error analyzing paper {paper.paper_id}: {e}")
                # 创建错误评估
                error_evals = {
                    dim: AgentEvaluation.create_error_evaluation(dim, str(e))
                    for dim in self.agents.keys()
                }
                results[paper.paper_id] = AggregatedEvaluation(
                    paper_id=paper.paper_id,
                    dimension_scores=error_evals,
                    weighted_score=5.0,
                    simple_average=5.0,
                    confidence_weighted_score=5.0,
                    overall_sentiment='mixed'
                )
        
        return results


if __name__ == "__main__":
    # 测试代码
    from .llm_client import OllamaClient
    
    # 创建测试论文
    test_paper = PaperInfo(
        paper_id="test_001",
        title="A Novel Approach to Machine Learning",
        abstract="We propose a new method that significantly improves upon existing approaches...",
        reviews=[
            {
                "rating": "6: Marginally above acceptance threshold",
                "confidence": "3: The reviewer is fairly confident",
                "review_text": "The paper presents an interesting idea but lacks novelty..."
            }
        ],
        rebuttal="We thank the reviewer for their feedback..."
    )
    
    # 运行分析
    client = OllamaClient(model="llama3.1:8b")
    analyzer = MultiDimensionalAnalyzer(client)
    result = analyzer.analyze_paper(test_paper)
    
    print(f"Paper: {result.paper_id}")
    print(f"Simple Average: {result.simple_average:.2f}")
    print(f"Weighted Score: {result.weighted_score:.2f}")
    for dim, eval in result.dimension_scores.items():
        print(f"  {dim}: {eval.score:.1f} (confidence: {eval.confidence:.2f})")