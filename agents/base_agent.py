"""
Base Agent class for Expert Agents.

All expert agents inherit from this base class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentEvaluation:
    """Agent评估结果结构"""
    dimension: str
    score: float  # 1-10
    confidence: float  # 0-1
    sentiment: str  # positive, negative, mixed
    key_findings: List[str] = field(default_factory=list)
    reasoning: str = ""
    raw_response: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "dimension": self.dimension,
            "score": self.score,
            "confidence": self.confidence,
            "sentiment": self.sentiment,
            "key_findings": self.key_findings,
            "reasoning": self.reasoning
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentEvaluation':
        return cls(
            dimension=data.get("dimension", "unknown"),
            score=float(data.get("score", 5.0)),
            confidence=float(data.get("confidence", 0.5)),
            sentiment=data.get("sentiment", "mixed"),
            key_findings=data.get("key_findings", []),
            reasoning=data.get("reasoning", ""),
            raw_response=data.get("raw_response")
        )
    
    @classmethod
    def create_error_evaluation(cls, dimension: str, error_msg: str) -> 'AgentEvaluation':
        """创建错误情况下的默认评估"""
        return cls(
            dimension=dimension,
            score=5.0,
            confidence=0.2,
            sentiment="mixed",
            key_findings=[f"Evaluation error: {error_msg}"],
            reasoning=f"Could not complete evaluation due to: {error_msg}"
        )


@dataclass
class PaperInfo:
    """论文信息结构"""
    paper_id: str
    title: str
    abstract: str
    reviews: List[Dict]
    rebuttal: Optional[str] = None
    meta_review: Optional[str] = None
    decision: Optional[str] = None  # 真实决策
    year: int = 0
    venue: str = ""
    
    def format_reviews_for_prompt(self, max_length: int = 3000) -> str:
        """格式化reviews用于prompt，包含人类评分汇总"""
        if not self.reviews:
            return "No reviews available."
        
        # 提取数值化的评分和置信度
        ratings = []
        confidences = []
        for review in self.reviews:
            # 提取rating数值
            rating_str = str(review.get('rating', ''))
            try:
                r = float(rating_str.split(':')[0].strip())
                ratings.append(r)
            except:
                pass
            
            # 提取confidence数值
            conf_str = str(review.get('confidence', ''))
            try:
                c = float(conf_str.split(':')[0].strip())
                confidences.append(c)
            except:
                pass
        
        # 构建汇总信息
        summary_parts = ["=== Human Reviewer Summary ==="]
        summary_parts.append(f"Number of reviewers: {len(self.reviews)}")
        
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            summary_parts.append(f"Ratings: {ratings}")
            summary_parts.append(f"Average rating: {avg_rating:.2f}/10")
            summary_parts.append(f"Rating range: {min(ratings):.0f} - {max(ratings):.0f}")
            rating_std = (sum((r - avg_rating)**2 for r in ratings) / len(ratings)) ** 0.5
            summary_parts.append(f"Rating std dev: {rating_std:.2f}")
            
            # 判断reviewer共识程度
            if rating_std < 0.8:
                summary_parts.append("Reviewer consensus: HIGH (reviewers largely agree)")
            elif rating_std < 1.5:
                summary_parts.append("Reviewer consensus: MEDIUM (some disagreement)")
            else:
                summary_parts.append("Reviewer consensus: LOW (significant disagreement)")
        
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            summary_parts.append(f"Average confidence: {avg_conf:.2f}/5")
        
        summary = "\n".join(summary_parts)
        
        # 格式化每个review
        formatted = [summary, "\n=== Individual Reviews ==="]
        for i, review in enumerate(self.reviews, 1):
            rating = review.get('rating', 'N/A')
            confidence = review.get('confidence', 'N/A')
            text = review.get('review_text', '')
            
            # 截断过长的review
            if len(text) > max_length // len(self.reviews):
                text = text[:max_length // len(self.reviews)] + "..."
            
            formatted.append(f"""
--- Review {i} ---
Rating: {rating}
Confidence: {confidence}
Review:
{text}
""")
        
        return "\n".join(formatted)
    
    def get_rating_stats(self) -> Dict:
        """获取评分统计信息"""
        ratings = []
        for review in self.reviews:
            rating_str = str(review.get('rating', ''))
            try:
                r = float(rating_str.split(':')[0].strip())
                ratings.append(r)
            except:
                continue
        
        if not ratings:
            return {"avg": None, "std": None, "min": None, "max": None, "count": 0}
        
        avg = sum(ratings) / len(ratings)
        std = (sum((r - avg)**2 for r in ratings) / len(ratings)) ** 0.5
        
        return {
            "avg": avg,
            "std": std,
            "min": min(ratings),
            "max": max(ratings),
            "count": len(ratings),
            "ratings": ratings
        }
    
    def get_average_rating(self) -> Optional[float]:
        """计算平均评分"""
        ratings = []
        for review in self.reviews:
            rating_str = str(review.get('rating', ''))
            # 提取数字（如 "5: Marginally below..." -> 5）
            try:
                rating = float(rating_str.split(':')[0].strip())
                ratings.append(rating)
            except:
                continue
        
        return sum(ratings) / len(ratings) if ratings else None
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PaperInfo':
        return cls(
            paper_id=data.get('paper_id', ''),
            title=data.get('title', ''),
            abstract=data.get('abstract', ''),
            reviews=data.get('reviews', []),
            rebuttal=data.get('rebuttal'),
            meta_review=data.get('meta_review'),
            decision=data.get('decision'),
            year=data.get('year', 0),
            venue=data.get('venue', '')
        )


class BaseExpertAgent(ABC):
    """Expert Agent基类"""
    
    def __init__(self, name: str, dimension: str, llm_client):
        self.name = name
        self.dimension = dimension
        self.llm = llm_client
        
    @abstractmethod
    def get_system_prompt(self) -> str:
        """返回系统提示"""
        pass
    
    @abstractmethod
    def get_evaluation_prompt(self, paper_info: PaperInfo) -> str:
        """返回评估提示"""
        pass
    
    def evaluate(self, paper_info: PaperInfo) -> AgentEvaluation:
        """
        评估论文
        
        Args:
            paper_info: 论文信息
        
        Returns:
            AgentEvaluation对象
        """
        try:
            system_prompt = self.get_system_prompt()
            eval_prompt = self.get_evaluation_prompt(paper_info)
            
            # 调用LLM
            response = self.llm.generate_json(
                prompt=eval_prompt,
                system_prompt=system_prompt
            )
            
            # 解析响应
            evaluation = self._parse_response(response)
            evaluation.dimension = self.dimension
            evaluation.raw_response = response
            
            # 验证并修正分数范围
            evaluation.score = max(1.0, min(10.0, evaluation.score))
            evaluation.confidence = max(0.0, min(1.0, evaluation.confidence))
            
            logger.debug(f"{self.name} evaluated {paper_info.paper_id}: score={evaluation.score}")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"{self.name} error evaluating {paper_info.paper_id}: {e}")
            return AgentEvaluation.create_error_evaluation(self.dimension, str(e))
    
    def _parse_response(self, response: Dict) -> AgentEvaluation:
        """解析LLM响应为AgentEvaluation"""
        # 处理解析错误的情况
        if response.get("parse_error"):
            return AgentEvaluation(
                dimension=self.dimension,
                score=response.get("score", 5.0),
                confidence=response.get("confidence", 0.3),
                sentiment=response.get("sentiment", "mixed"),
                key_findings=response.get("key_findings", []),
                reasoning=response.get("reasoning", "")
            )
        
        # 正常解析
        score = response.get("score", 5.0)
        if isinstance(score, str):
            try:
                score = float(score.split(':')[0].strip())
            except:
                score = 5.0
        
        confidence = response.get("confidence", 0.5)
        if isinstance(confidence, str):
            try:
                confidence = float(confidence)
            except:
                confidence = 0.5
        
        sentiment = response.get("sentiment", "mixed")
        if sentiment not in ["positive", "negative", "mixed"]:
            sentiment = "mixed"
        
        key_findings = response.get("key_findings", [])
        if isinstance(key_findings, str):
            key_findings = [key_findings]
        
        reasoning = response.get("reasoning", "")
        
        return AgentEvaluation(
            dimension=self.dimension,
            score=float(score),
            confidence=float(confidence),
            sentiment=sentiment,
            key_findings=key_findings,
            reasoning=reasoning
        )
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, dimension={self.dimension})"