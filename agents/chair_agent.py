"""
Chair Agent for final decision making with iterative calibration.

The Chair Agent synthesizes expert evaluations and makes accept/reject decisions.
The iterative calibration mechanism ensures simulated acceptance count matches real count.

Usage:
    from agents.chair_agent import ChairAgent, iterative_calibration
    chair = ChairAgent(llm_client, venue="iclr", year=2018)
    decisions = iterative_calibration(papers, chair, analyzer, real_accept_count=337)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from .base_agent import PaperInfo, AgentEvaluation
from .expert_agents import AggregatedEvaluation

logger = logging.getLogger(__name__)


@dataclass
class ChairDecision:
    """Chair Agent决策结果"""
    paper_id: str
    decision: str  # Accept or Reject
    confidence: float  # 0-1
    reasoning: str
    key_factors: List[str]
    expert_summary: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            "paper_id": self.paper_id,
            "decision": self.decision,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "key_factors": self.key_factors,
            "expert_summary": self.expert_summary
        }


class ChairAgent:
    """
    Chair Agent - 综合Expert评估做出最终决策
    
    基于4个维度的Expert评估，Chair Agent做出Accept/Reject决策。
    """
    
    def __init__(self, llm_client, venue: str = "iclr", year: int = 2018):
        self.llm = llm_client
        self.venue = venue
        self.year = year
    
    def get_system_prompt(self) -> str:
        return f"""You are the Area Chair for {self.venue.upper()} {self.year}. Your role is to make the final accept/reject decision for papers based on expert evaluations.

You have received evaluations from four expert reviewers, each focusing on a different dimension:
- Novelty Expert: evaluates originality and innovation
- Methodology Expert: evaluates technical soundness
- Clarity Expert: evaluates presentation quality
- Empirical Expert: evaluates experimental rigor

Your job is to synthesize these evaluations and make a final decision. Consider:
1. The relative importance of each dimension
2. How critical weaknesses might be addressed in revisions
3. The overall contribution to the field
4. The consistency and confidence of expert opinions

Be fair but rigorous. The acceptance rate is typically around 25-35%."""

    def decide(
        self, 
        paper_info: PaperInfo, 
        expert_results: AggregatedEvaluation
    ) -> ChairDecision:
        """
        基于Expert评估做出决策
        
        Args:
            paper_info: 论文信息
            expert_results: 4个Expert的聚合评估结果
        
        Returns:
            ChairDecision对象
        """
        prompt = self._build_decision_prompt(paper_info, expert_results)
        
        try:
            response = self.llm.generate_json(
                prompt=prompt,
                system_prompt=self.get_system_prompt()
            )
            
            decision = self._parse_decision(paper_info.paper_id, expert_results, response)
            return decision
            
        except Exception as e:
            logger.error(f"Chair decision error for {paper_info.paper_id}: {e}")
            return self._fallback_decision(paper_info.paper_id, expert_results)
    
    def _build_decision_prompt(
        self, 
        paper_info: PaperInfo, 
        expert_results: AggregatedEvaluation
    ) -> str:
        # 格式化Expert评估
        expert_summary = []
        for dim, eval in expert_results.dimension_scores.items():
            expert_summary.append(f"""
{dim.upper()} Expert (Score: {eval.score:.1f}/10, Confidence: {eval.confidence:.2f}):
- Sentiment: {eval.sentiment}
- Key Findings: {', '.join(eval.key_findings[:3]) if eval.key_findings else 'N/A'}
- Reasoning: {eval.reasoning[:300]}...""")
        
        expert_text = "\n".join(expert_summary)
        
        return f"""Make an accept/reject decision for the following paper:

=== Paper ===
Title: {paper_info.title}
Abstract: {paper_info.abstract[:500]}...

=== Expert Evaluations ===
{expert_text}

=== Aggregate Scores ===
Simple Average: {expert_results.simple_average:.2f}/10
Weighted Score: {expert_results.weighted_score:.2f}/10
Overall Sentiment: {expert_results.overall_sentiment}

Based on the expert evaluations, make your decision.

Respond with a JSON object:
{{
    "decision": "<Accept or Reject>",
    "confidence": <float 0-1>,
    "reasoning": "<your detailed reasoning>",
    "key_factors": ["<factor1>", "<factor2>", ...]
}}

ONLY output the JSON object, nothing else."""

    def _parse_decision(
        self, 
        paper_id: str, 
        expert_results: AggregatedEvaluation,
        response: Dict
    ) -> ChairDecision:
        """解析LLM响应"""
        decision = response.get("decision", "Reject")
        if decision not in ["Accept", "Reject"]:
            # 尝试修复常见变体
            if "accept" in str(decision).lower():
                decision = "Accept"
            else:
                decision = "Reject"
        
        confidence = float(response.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        
        reasoning = response.get("reasoning", "")
        key_factors = response.get("key_factors", [])
        
        expert_summary = {
            dim: eval.score for dim, eval in expert_results.dimension_scores.items()
        }
        
        return ChairDecision(
            paper_id=paper_id,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=key_factors,
            expert_summary=expert_summary
        )
    
    def _fallback_decision(
        self, 
        paper_id: str, 
        expert_results: AggregatedEvaluation
    ) -> ChairDecision:
        """当LLM调用失败时的fallback决策"""
        # 基于分数阈值决策
        score = expert_results.weighted_score
        decision = "Accept" if score >= 5.5 else "Reject"
        
        return ChairDecision(
            paper_id=paper_id,
            decision=decision,
            confidence=0.5,
            reasoning="Fallback decision based on weighted score threshold",
            key_factors=["weighted_score_threshold"],
            expert_summary={
                dim: eval.score for dim, eval in expert_results.dimension_scores.items()
            }
        )
    
    def re_evaluate(
        self, 
        papers_with_evals: List[Tuple[PaperInfo, AggregatedEvaluation]],
        context: str
    ) -> Dict[str, str]:
        """
        重新评估一批论文（用于迭代校准）
        
        Args:
            papers_with_evals: [(PaperInfo, AggregatedEvaluation), ...]
            context: 当前校准状态的上下文说明
        
        Returns:
            {paper_id: 'Accept' or 'Reject'}
        """
        decisions = {}
        
        for paper_info, expert_results in papers_with_evals:
            prompt = self._build_reevaluation_prompt(paper_info, expert_results, context)
            
            try:
                response = self.llm.generate_json(
                    prompt=prompt,
                    system_prompt=self.get_system_prompt()
                )
                
                decision = response.get("decision", "Reject")
                if "accept" in str(decision).lower():
                    decisions[paper_info.paper_id] = "Accept"
                else:
                    decisions[paper_info.paper_id] = "Reject"
                    
            except Exception as e:
                logger.warning(f"Re-evaluation failed for {paper_info.paper_id}: {e}")
                # 保持原决策或基于分数
                score = expert_results.weighted_score
                decisions[paper_info.paper_id] = "Accept" if score >= 5.5 else "Reject"
        
        return decisions
    
    def _build_reevaluation_prompt(
        self, 
        paper_info: PaperInfo, 
        expert_results: AggregatedEvaluation,
        context: str
    ) -> str:
        expert_scores = {
            dim: f"{eval.score:.1f}" for dim, eval in expert_results.dimension_scores.items()
        }
        
        return f"""Re-evaluate this paper in the context of the calibration process:

=== Calibration Context ===
{context}

=== Paper ===
Title: {paper_info.title}
Abstract: {paper_info.abstract[:300]}...

=== Expert Scores ===
{expert_scores}
Weighted Average: {expert_results.weighted_score:.2f}/10

Should this paper be accepted or rejected given the current situation?

Respond with a JSON object:
{{
    "decision": "<Accept or Reject>",
    "reasoning": "<brief reasoning>"
}}

ONLY output the JSON object, nothing else."""


def get_top_by_aggregate_score(
    papers: List[Tuple[PaperInfo, AggregatedEvaluation]], 
    n: int
) -> List[Tuple[PaperInfo, AggregatedEvaluation]]:
    """获取Agent分数最高的n篇论文"""
    sorted_papers = sorted(
        papers, 
        key=lambda x: x[1].weighted_score, 
        reverse=True
    )
    return sorted_papers[:n]


def get_bottom_by_aggregate_score(
    papers: List[Tuple[PaperInfo, AggregatedEvaluation]], 
    n: int
) -> List[Tuple[PaperInfo, AggregatedEvaluation]]:
    """获取Agent分数最低的n篇论文"""
    sorted_papers = sorted(
        papers, 
        key=lambda x: x[1].weighted_score
    )
    return sorted_papers[:n]


def iterative_calibration(
    papers_with_evals: List[Tuple[PaperInfo, AggregatedEvaluation]],
    chair_agent: ChairAgent,
    real_accept_count: int,
    max_iterations: int = 10,
    batch_extra: int = 2
) -> Dict[str, ChairDecision]:
    """
    迭代校准机制
    
    确保模拟接受数与真实接受数一致，同时保持Agent的自主决策权。
    如果迭代无法收敛，使用人类评审分数强制对齐。
    
    Args:
        papers_with_evals: [(PaperInfo, AggregatedEvaluation), ...]
        chair_agent: Chair Agent实例
        real_accept_count: 真实接受论文数
        max_iterations: 最大迭代次数
        batch_extra: 每轮额外选取的论文数
    
    Returns:
        {paper_id: ChairDecision}
    """
    logger.info(f"Starting iterative calibration. Target accept count: {real_accept_count}")
    
    # 第一轮：Chair Agent对所有论文独立决策
    decisions = {}
    for paper_info, expert_results in papers_with_evals:
        decision = chair_agent.decide(paper_info, expert_results)
        decisions[paper_info.paper_id] = decision
    
    # 构建paper_id到(PaperInfo, AggregatedEvaluation)的映射
    paper_map = {p.paper_id: (p, e) for p, e in papers_with_evals}
    
    prev_accept_count = -1
    same_count_iterations = 0
    
    for iteration in range(max_iterations):
        # 统计当前接受/拒绝
        accepted = [(paper_map[pid][0], paper_map[pid][1]) 
                   for pid, d in decisions.items() if d.decision == "Accept"]
        rejected = [(paper_map[pid][0], paper_map[pid][1]) 
                   for pid, d in decisions.items() if d.decision == "Reject"]
        
        current_accept_count = len(accepted)
        
        logger.info(f"Iteration {iteration + 1}: Accepted={current_accept_count}, Target={real_accept_count}")
        
        # 检查是否达到目标
        if current_accept_count == real_accept_count:
            logger.info("Calibration converged!")
            return decisions
        
        # 检查是否陷入循环
        if current_accept_count == prev_accept_count:
            same_count_iterations += 1
            if same_count_iterations >= 2:
                logger.warning("Calibration stuck, using human score fallback")
                return fallback_by_human_scores(
                    papers_with_evals, real_accept_count, decisions
                )
        else:
            same_count_iterations = 0
        
        prev_accept_count = current_accept_count
        
        if current_accept_count < real_accept_count:
            # 接受数不足：从rejected中选取高分论文重评
            need_more = real_accept_count - current_accept_count
            candidates = get_top_by_aggregate_score(rejected, n=need_more + batch_extra)
            
            context = f"""
Current Calibration Status:
- Currently accepted papers: {current_accept_count}
- Target acceptance count: {real_accept_count}
- Papers being re-evaluated: {len(candidates)}
- Additional papers needed: {need_more}

These papers were rejected in the previous round but scored highly among rejected papers.
Please reconsider whether they should be accepted given the available acceptance slots.
            """
            
            new_decisions = chair_agent.re_evaluate(candidates, context)
            
            for paper_id, new_decision in new_decisions.items():
                if new_decision == "Accept":
                    old_decision = decisions[paper_id]
                    decisions[paper_id] = ChairDecision(
                        paper_id=paper_id,
                        decision="Accept",
                        confidence=old_decision.confidence,
                        reasoning=f"Re-evaluated in calibration round {iteration + 1}",
                        key_factors=["calibration_re_evaluation"],
                        expert_summary=old_decision.expert_summary
                    )
        
        else:
            # 接受数过多：从accepted中选取低分论文重评
            need_remove = current_accept_count - real_accept_count
            candidates = get_bottom_by_aggregate_score(accepted, n=need_remove + batch_extra)
            
            context = f"""
Current Calibration Status:
- Currently accepted papers: {current_accept_count}
- Target acceptance count: {real_accept_count}
- Papers being re-evaluated: {len(candidates)}
- Papers to potentially reject: {need_remove}

These papers were accepted in the previous round but scored lower among accepted papers.
Please reconsider whether they should remain accepted given the limited acceptance slots.
            """
            
            new_decisions = chair_agent.re_evaluate(candidates, context)
            
            for paper_id, new_decision in new_decisions.items():
                if new_decision == "Reject":
                    old_decision = decisions[paper_id]
                    decisions[paper_id] = ChairDecision(
                        paper_id=paper_id,
                        decision="Reject",
                        confidence=old_decision.confidence,
                        reasoning=f"Re-evaluated in calibration round {iteration + 1}",
                        key_factors=["calibration_re_evaluation"],
                        expert_summary=old_decision.expert_summary
                    )
    
    # 迭代结束仍未收敛，强制使用人类分数对齐
    logger.warning("Max iterations reached, using human score fallback")
    return fallback_by_human_scores(papers_with_evals, real_accept_count, decisions)


def fallback_by_human_scores(
    papers_with_evals: List[Tuple[PaperInfo, AggregatedEvaluation]],
    real_accept_count: int,
    current_decisions: Dict[str, ChairDecision]
) -> Dict[str, ChairDecision]:
    """
    兜底机制：使用人类评审分数强制对齐接受数
    
    当迭代校准无法收敛时，用人类评审的平均分数微调边界论文，
    确保最终模拟接受数严格等于真实接受数。
    
    Args:
        papers_with_evals: [(PaperInfo, AggregatedEvaluation), ...]
        real_accept_count: 真实接受数
        current_decisions: 当前的决策字典
    
    Returns:
        调整后的决策字典，保证Accept数量 == real_accept_count
    """
    logger.info("Using fallback mechanism based on HUMAN review scores")
    
    # 分离当前的Accept和Reject，并附带人类评审分数
    accepted_papers = []
    rejected_papers = []
    
    for paper_info, expert_result in papers_with_evals:
        paper_id = paper_info.paper_id
        decision = current_decisions.get(paper_id)
        
        # 获取人类评审平均分
        human_avg_score = paper_info.get_average_rating()
        if human_avg_score is None:
            human_avg_score = 5.0  # 默认值
        
        item = (paper_info, expert_result, human_avg_score, decision)
        
        if decision and decision.decision == "Accept":
            accepted_papers.append(item)
        else:
            rejected_papers.append(item)
    
    current_accept_count = len(accepted_papers)
    diff = current_accept_count - real_accept_count
    
    logger.info(f"Fallback: Current Accept={current_accept_count}, Target={real_accept_count}, Diff={diff}")
    
    # 复制当前决策
    new_decisions = dict(current_decisions)
    
    if diff > 0:
        # 接受太多，需要把一些改成Reject
        # 按人类评分从低到高排序，踢掉人类评分最低的diff篇
        accepted_papers.sort(key=lambda x: x[2])  # 按human_avg_score升序
        
        for i in range(diff):
            paper_info, expert_result, human_score, old_decision = accepted_papers[i]
            new_decisions[paper_info.paper_id] = ChairDecision(
                paper_id=paper_info.paper_id,
                decision="Reject",
                confidence=old_decision.confidence if old_decision else 0.5,
                reasoning=f"Fallback: changed to Reject (human_score={human_score:.2f}, lowest among accepted)",
                key_factors=["human_score_fallback", "forced_reject"],
                expert_summary=old_decision.expert_summary if old_decision else {}
            )
            logger.info(f"  Fallback REJECT: {paper_info.paper_id[:20]}... (human_score={human_score:.2f})")
    
    elif diff < 0:
        # 接受太少，需要把一些改成Accept
        # 按人类评分从高到低排序，拉回人类评分最高的|diff|篇
        rejected_papers.sort(key=lambda x: x[2], reverse=True)  # 按human_avg_score降序
        
        for i in range(abs(diff)):
            paper_info, expert_result, human_score, old_decision = rejected_papers[i]
            new_decisions[paper_info.paper_id] = ChairDecision(
                paper_id=paper_info.paper_id,
                decision="Accept",
                confidence=old_decision.confidence if old_decision else 0.5,
                reasoning=f"Fallback: changed to Accept (human_score={human_score:.2f}, highest among rejected)",
                key_factors=["human_score_fallback", "forced_accept"],
                expert_summary=old_decision.expert_summary if old_decision else {}
            )
            logger.info(f"  Fallback ACCEPT: {paper_info.paper_id[:20]}... (human_score={human_score:.2f})")
    
    # 验证最终数量
    final_accept_count = sum(1 for d in new_decisions.values() if d.decision == "Accept")
    logger.info(f"Fallback complete: Final Accept={final_accept_count}, Target={real_accept_count}")
    
    assert final_accept_count == real_accept_count, \
        f"Fallback failed: {final_accept_count} != {real_accept_count}"
    
    return new_decisions


def simple_threshold_decisions(
    papers_with_evals: List[Tuple[PaperInfo, AggregatedEvaluation]],
    threshold: float = 5.5
) -> Dict[str, str]:
    """
    简单阈值决策（M0机制）
    
    分数 >= threshold 则Accept，否则Reject
    """
    decisions = {}
    for paper_info, expert_results in papers_with_evals:
        if expert_results.weighted_score >= threshold:
            decisions[paper_info.paper_id] = "Accept"
        else:
            decisions[paper_info.paper_id] = "Reject"
    return decisions