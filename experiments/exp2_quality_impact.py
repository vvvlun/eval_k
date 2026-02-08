"""
Experiment 2: Quality-Impact Alignment Validation

验证桥梁假设：论文的审稿评分与长期影响力（引用）显著正相关。
如果这个假设成立，后续实验的"减少估计误差→更好识别影响力"逻辑才有意义。

Usage:
    # 基本用法
    python exp2_quality_impact.py \
        --citation-dir ./data/iclr_cite_analys \
        --review-dir ./data/iclr_review_analys \
        --agent-dir ./data/results \
        --output ./data/results/exp2_quality_impact.json
    
    # 指定年份范围
    python exp2_quality_impact.py \
        --citation-dir ./data/iclr_cite_analys \
        --review-dir ./data/iclr_review_analys \
        --agent-dir ./data/results \
        --years 2018 2019 2020 2021 2022 \
        --output ./data/results/exp2_quality_impact.json
    
    # 指定agent模型
    python exp2_quality_impact.py \
        --citation-dir ./data/iclr_cite_analys \
        --review-dir ./data/iclr_review_analys \
        --agent-dir ./data/results \
        --agent-model llama8b \
        --output ./data/results/exp2_quality_impact.json

Output:
    - JSON结果文件包含：
      - 整体相关性分析
      - 按年份分层分析
      - 按决策分层分析
      - LaTeX表格格式输出
"""

import os
import sys
import json
import argparse
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict

# 统计库
try:
    from scipy.stats import spearmanr, pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Install with: pip install scipy")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available. Install with: pip install numpy")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class PaperWithImpact:
    """论文及其影响力数据"""
    paper_id: str
    title: str
    venue: str
    year: int
    decision: str  # Accept/Reject
    
    # 审稿数据
    avg_rating: Optional[float] = None
    rating_std: Optional[float] = None
    num_reviewers: int = 0
    ratings: List[float] = field(default_factory=list)
    
    # 多智能体数据
    agent_weighted_score: Optional[float] = None
    agent_scores: Dict[str, float] = field(default_factory=dict)
    
    # 影响力数据
    citations: int = 0
    influential_citations: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CorrelationResult:
    """相关性分析结果"""
    metric_name: str
    description: str
    n: int
    spearman_rho: float
    spearman_p: float
    pearson_r: float
    pearson_p: float
    significant_001: bool  # p < 0.001
    significant_005: bool  # p < 0.05
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# 数据加载函数
# ============================================================================

def extract_rating_number(rating_str: Any) -> Optional[float]:
    """
    从rating字符串中提取数字
    
    Examples:
        "5: Marginally below acceptance threshold" -> 5.0
        "8: Strong Accept" -> 8.0
        8 -> 8.0
        None -> None
    """
    if rating_str is None:
        return None
    
    if isinstance(rating_str, (int, float)):
        return float(rating_str)
    
    rating_str = str(rating_str).strip()
    if not rating_str:
        return None
    
    # 尝试匹配开头的数字
    match = re.match(r'^(\d+(?:\.\d+)?)', rating_str)
    if match:
        return float(match.group(1))
    
    return None


def load_citation_data(citation_dir: str, years: List[int]) -> Dict[str, Dict]:
    """
    加载引用数据
    
    Args:
        citation_dir: 引用数据目录
        years: 年份列表
    
    Returns:
        {paper_id: citation_data}
    """
    citation_data = {}
    citation_dir = Path(citation_dir)
    
    for year in years:
        # 尝试多种文件名格式
        possible_files = [
            f"citation_iclr_{year}.json",
            f"citations_iclr_{year}.json",
            f"iclr_{year}_citations.json",
        ]
        
        file_found = False
        for filename in possible_files:
            filepath = citation_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for item in data:
                    paper_id = item.get('paper_id', '')
                    if paper_id:
                        citation_data[paper_id] = {
                            'citations': item.get('citation_count', 0) or 0,
                            'influential_citations': item.get('influential_citation_count', 0) or 0,
                            'year': item.get('year', year),
                            'venue': 'iclr'
                        }
                
                logger.info(f"Loaded {len(data)} citation records from {filepath}")
                file_found = True
                break
        
        if not file_found:
            logger.warning(f"Citation file not found for year {year} in {citation_dir}")
    
    return citation_data


def load_review_data(review_dir: str, years: List[int]) -> Dict[str, Dict]:
    """
    加载原始审稿数据
    
    Args:
        review_dir: 审稿数据目录
        years: 年份列表
    
    Returns:
        {paper_id: review_data}
    """
    review_data = {}
    review_dir = Path(review_dir)
    
    for year in years:
        # 尝试多种文件名格式
        possible_files = [
            f"iclr_{year}.json",
            f"iclr{year}.json",
        ]
        
        file_found = False
        for filename in possible_files:
            filepath = review_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for item in data:
                    paper_id = item.get('paper_id', '')
                    if not paper_id:
                        continue
                    
                    # 提取评分
                    reviews = item.get('reviews', [])
                    ratings = []
                    for review in reviews:
                        rating = extract_rating_number(review.get('rating'))
                        if rating is not None:
                            ratings.append(rating)
                    
                    # 计算统计量
                    avg_rating = sum(ratings) / len(ratings) if ratings else None
                    rating_std = None
                    if len(ratings) > 1:
                        mean = avg_rating
                        variance = sum((r - mean) ** 2 for r in ratings) / len(ratings)
                        rating_std = variance ** 0.5
                    
                    review_data[paper_id] = {
                        'title': item.get('title', ''),
                        'decision': item.get('decision', ''),
                        'year': item.get('year', year),
                        'venue': item.get('venue', 'iclr'),
                        'avg_rating': avg_rating,
                        'rating_std': rating_std,
                        'num_reviewers': len(ratings),
                        'ratings': ratings
                    }
                
                logger.info(f"Loaded {len(data)} review records from {filepath}")
                file_found = True
                break
        
        if not file_found:
            logger.warning(f"Review file not found for year {year} in {review_dir}")
    
    return review_data


def load_agent_data(agent_dir: str, years: List[int], model: str = "llama8b") -> Dict[str, Dict]:
    """
    加载多智能体评估数据
    
    Args:
        agent_dir: 智能体结果目录
        years: 年份列表
        model: 模型名称（如 llama8b, qwen7b）
    
    Returns:
        {paper_id: agent_data}
    """
    agent_data = {}
    agent_dir = Path(agent_dir)
    
    for year in years:
        # 尝试多种文件名格式
        possible_files = [
            f"iclr{year}_con_{model}.json",
            f"iclr_{year}_con_{model}.json",
            f"iclr{year}_{model}.json",
        ]
        
        file_found = False
        for filename in possible_files:
            filepath = agent_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                paper_details = data.get('paper_details', [])
                for item in paper_details:
                    paper_id = item.get('paper_id', '')
                    if not paper_id:
                        continue
                    
                    agent_data[paper_id] = {
                        'weighted_score': item.get('weighted_score'),
                        'expert_scores': item.get('expert_scores', {}),
                        'simulated_decision': item.get('simulated_decision', ''),
                        'overall_sentiment': item.get('overall_sentiment', '')
                    }
                
                logger.info(f"Loaded {len(paper_details)} agent records from {filepath}")
                file_found = True
                break
        
        if not file_found:
            logger.warning(f"Agent file not found for year {year}, model {model} in {agent_dir}")
    
    return agent_data


def merge_data(
    citation_data: Dict[str, Dict],
    review_data: Dict[str, Dict],
    agent_data: Dict[str, Dict]
) -> List[PaperWithImpact]:
    """
    合并所有数据源
    
    Returns:
        PaperWithImpact列表
    """
    papers = []
    
    # 以review_data为主键（因为它包含决策信息）
    for paper_id, review in review_data.items():
        citation = citation_data.get(paper_id, {})
        agent = agent_data.get(paper_id, {})
        
        paper = PaperWithImpact(
            paper_id=paper_id,
            title=review.get('title', ''),
            venue=review.get('venue', 'iclr'),
            year=review.get('year', 0),
            decision=review.get('decision', ''),
            avg_rating=review.get('avg_rating'),
            rating_std=review.get('rating_std'),
            num_reviewers=review.get('num_reviewers', 0),
            ratings=review.get('ratings', []),
            agent_weighted_score=agent.get('weighted_score'),
            agent_scores=agent.get('expert_scores', {}),
            citations=citation.get('citations', 0),
            influential_citations=citation.get('influential_citations', 0)
        )
        
        papers.append(paper)
    
    logger.info(f"Merged {len(papers)} papers with all data sources")
    return papers


# ============================================================================
# 相关性计算
# ============================================================================

def compute_correlation(
    scores: List[float],
    citations: List[int],
    metric_name: str,
    description: str
) -> Optional[CorrelationResult]:
    """
    计算单个指标与引用的相关性
    
    Args:
        scores: 评分列表
        citations: 引用数列表
        metric_name: 指标名称
        description: 指标描述
    
    Returns:
        CorrelationResult 或 None（如果数据不足）
    """
    if not SCIPY_AVAILABLE:
        logger.error("scipy is required for correlation computation")
        return None
    
    # 过滤无效数据
    valid_pairs = [(s, c) for s, c in zip(scores, citations) if s is not None and c is not None]
    
    if len(valid_pairs) < 10:
        logger.warning(f"Not enough valid data for {metric_name}: {len(valid_pairs)} pairs")
        return None
    
    scores_valid = [p[0] for p in valid_pairs]
    citations_valid = [p[1] for p in valid_pairs]
    
    # 检查是否为常数数组（会导致相关性未定义）
    if len(set(scores_valid)) == 1 or len(set(citations_valid)) == 1:
        logger.warning(f"Constant input for {metric_name}, correlation undefined")
        return CorrelationResult(
            metric_name=metric_name,
            description=description,
            n=len(valid_pairs),
            spearman_rho=float('nan'),
            spearman_p=float('nan'),
            pearson_r=float('nan'),
            pearson_p=float('nan'),
            significant_001=False,
            significant_005=False
        )
    
    # Spearman相关（适合非正态分布）
    spearman_rho, spearman_p = spearmanr(scores_valid, citations_valid)
    
    # Pearson相关（作为参考）
    pearson_r, pearson_p = pearsonr(scores_valid, citations_valid)
    
    # 处理可能的nan值
    spearman_rho = float(spearman_rho) if not (spearman_rho != spearman_rho) else 0.0
    spearman_p = float(spearman_p) if not (spearman_p != spearman_p) else 1.0
    pearson_r = float(pearson_r) if not (pearson_r != pearson_r) else 0.0
    pearson_p = float(pearson_p) if not (pearson_p != pearson_p) else 1.0
    
    return CorrelationResult(
        metric_name=metric_name,
        description=description,
        n=len(valid_pairs),
        spearman_rho=spearman_rho,
        spearman_p=spearman_p,
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        significant_001=bool(spearman_p < 0.001),
        significant_005=bool(spearman_p < 0.05)
    )


def compute_all_correlations(papers: List[PaperWithImpact]) -> Dict[str, CorrelationResult]:
    """
    计算所有指标与引用的相关性
    
    Args:
        papers: 论文列表
    
    Returns:
        {metric_name: CorrelationResult}
    """
    results = {}
    
    citations = [p.citations for p in papers]
    
    # 1. 原始审稿人平均分 vs 引用
    avg_ratings = [p.avg_rating for p in papers]
    result = compute_correlation(
        avg_ratings, citations,
        'original_reviewer_avg',
        '原始审稿人平均分 vs 引用'
    )
    if result:
        results['original_reviewer_avg'] = result
    
    # 2. 多智能体加权分 vs 引用
    agent_scores = [p.agent_weighted_score for p in papers]
    result = compute_correlation(
        agent_scores, citations,
        'multi_agent_score',
        '多智能体加权分 vs 引用'
    )
    if result:
        results['multi_agent_score'] = result
    
    # 3. 决策二元变量 vs 引用
    decision_binary = []
    for p in papers:
        if p.decision:
            if 'accept' in p.decision.lower():
                decision_binary.append(1.0)
            elif 'reject' in p.decision.lower():
                decision_binary.append(0.0)
            else:
                decision_binary.append(None)
        else:
            decision_binary.append(None)
    
    result = compute_correlation(
        decision_binary, citations,
        'decision_binary',
        'Accept=1/Reject=0 vs 引用'
    )
    if result:
        results['decision_binary'] = result
    
    # 4. 评分标准差 vs 引用（探索性）
    rating_stds = [p.rating_std for p in papers]
    result = compute_correlation(
        rating_stds, citations,
        'rating_std',
        '审稿人评分标准差 vs 引用（探索性）'
    )
    if result:
        results['rating_std'] = result
    
    return results


def compute_stratified_analysis(
    papers: List[PaperWithImpact]
) -> Dict[str, Any]:
    """
    分层分析
    
    Returns:
        包含按年份、按venue、按决策的分层分析结果
    """
    results = {
        'by_year': {},
        'by_venue': {},
        'by_decision': {}
    }
    
    # 按年份分层
    papers_by_year = defaultdict(list)
    for p in papers:
        if p.year:
            papers_by_year[p.year].append(p)
    
    for year, year_papers in sorted(papers_by_year.items()):
        if len(year_papers) >= 20:
            correlations = compute_all_correlations(year_papers)
            results['by_year'][year] = {
                'n': len(year_papers),
                'correlations': {k: v.to_dict() for k, v in correlations.items()}
            }
    
    # 按venue分层
    papers_by_venue = defaultdict(list)
    for p in papers:
        if p.venue:
            papers_by_venue[p.venue.lower()].append(p)
    
    for venue, venue_papers in papers_by_venue.items():
        if len(venue_papers) >= 20:
            correlations = compute_all_correlations(venue_papers)
            results['by_venue'][venue] = {
                'n': len(venue_papers),
                'correlations': {k: v.to_dict() for k, v in correlations.items()}
            }
    
    # 按决策分层（Accept vs Reject组内相关性）
    accept_papers = [p for p in papers if p.decision and 'accept' in p.decision.lower()]
    reject_papers = [p for p in papers if p.decision and 'reject' in p.decision.lower()]
    
    if len(accept_papers) >= 20:
        correlations = compute_all_correlations(accept_papers)
        results['by_decision']['Accept'] = {
            'n': len(accept_papers),
            'correlations': {k: v.to_dict() for k, v in correlations.items()}
        }
    
    if len(reject_papers) >= 20:
        correlations = compute_all_correlations(reject_papers)
        results['by_decision']['Reject'] = {
            'n': len(reject_papers),
            'correlations': {k: v.to_dict() for k, v in correlations.items()}
        }
    
    return results


def compute_descriptive_statistics(papers: List[PaperWithImpact]) -> Dict[str, Any]:
    """
    计算描述性统计
    """
    stats = {}
    
    # 引用分布
    citations = [p.citations for p in papers if p.citations is not None]
    if citations and NUMPY_AVAILABLE:
        stats['citations'] = {
            'n': int(len(citations)),
            'mean': float(np.mean(citations)),
            'median': float(np.median(citations)),
            'std': float(np.std(citations)),
            'min': int(min(citations)),
            'max': int(max(citations)),
            'q25': float(np.percentile(citations, 25)),
            'q75': float(np.percentile(citations, 75))
        }
    
    # 评分分布
    ratings = [p.avg_rating for p in papers if p.avg_rating is not None]
    if ratings and NUMPY_AVAILABLE:
        stats['avg_rating'] = {
            'n': int(len(ratings)),
            'mean': float(np.mean(ratings)),
            'median': float(np.median(ratings)),
            'std': float(np.std(ratings)),
            'min': float(min(ratings)),
            'max': float(max(ratings))
        }
    
    # 智能体评分分布
    agent_scores = [p.agent_weighted_score for p in papers if p.agent_weighted_score is not None]
    if agent_scores and NUMPY_AVAILABLE:
        stats['agent_score'] = {
            'n': int(len(agent_scores)),
            'mean': float(np.mean(agent_scores)),
            'median': float(np.median(agent_scores)),
            'std': float(np.std(agent_scores)),
            'min': float(min(agent_scores)),
            'max': float(max(agent_scores))
        }
    
    # 决策分布
    accept_count = sum(1 for p in papers if p.decision and 'accept' in p.decision.lower())
    reject_count = sum(1 for p in papers if p.decision and 'reject' in p.decision.lower())
    stats['decisions'] = {
        'accept': int(accept_count),
        'reject': int(reject_count),
        'accept_rate': float(accept_count / (accept_count + reject_count)) if (accept_count + reject_count) > 0 else 0.0
    }
    
    # 年份分布
    years = [p.year for p in papers if p.year]
    stats['years'] = {
        'years': sorted(set(years)),
        'counts': {int(y): int(years.count(y)) for y in sorted(set(years))}
    }
    
    return stats


# ============================================================================
# 输出格式化
# ============================================================================

def format_latex_table(results: Dict[str, CorrelationResult]) -> str:
    """
    生成LaTeX表格格式
    """
    lines = []
    lines.append("% Quality-Impact Alignment Results")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Correlation between review scores and long-term citations}")
    lines.append("\\label{tab:alignment}")
    lines.append("\\begin{tabular}{lrr}")
    lines.append("\\toprule")
    lines.append("Score Type & Spearman $\\rho$ & $p$-value \\\\")
    lines.append("\\midrule")
    
    for metric_name, result in results.items():
        # 格式化名称
        display_name = {
            'original_reviewer_avg': 'Original reviewer average',
            'multi_agent_score': 'Multi-agent score',
            'decision_binary': 'Decision (Accept=1)',
            'rating_std': 'Rating std. dev.'
        }.get(metric_name, metric_name)
        
        # 格式化p值
        if result.spearman_p < 0.001:
            p_str = "$<$0.001"
        else:
            p_str = f"{result.spearman_p:.3f}"
        
        lines.append(f"{display_name} & {result.spearman_rho:.3f} & {p_str} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def format_summary(
    overall_results: Dict[str, CorrelationResult],
    stratified_results: Dict[str, Any],
    stats: Dict[str, Any]
) -> str:
    """
    生成可读的摘要报告
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Experiment 2: Quality-Impact Alignment Validation")
    lines.append("=" * 60)
    
    # 描述性统计
    lines.append("\n--- Descriptive Statistics ---")
    if 'citations' in stats:
        c = stats['citations']
        lines.append(f"Citations: n={c['n']}, mean={c['mean']:.1f}, median={c['median']:.0f}, std={c['std']:.1f}")
    if 'avg_rating' in stats:
        r = stats['avg_rating']
        lines.append(f"Avg Rating: n={r['n']}, mean={r['mean']:.2f}, std={r['std']:.2f}")
    if 'decisions' in stats:
        d = stats['decisions']
        lines.append(f"Decisions: Accept={d['accept']}, Reject={d['reject']}, Rate={d['accept_rate']:.1%}")
    
    # 整体结果
    lines.append("\n--- Overall Correlations ---")
    lines.append(f"{'Metric':<25} {'n':>8} {'Spearman ρ':>12} {'p-value':>12} {'Significant':>12}")
    lines.append("-" * 70)
    
    for metric_name, result in overall_results.items():
        sig = "***" if result.significant_001 else ("*" if result.significant_005 else "")
        p_str = "<0.001" if result.spearman_p < 0.001 else f"{result.spearman_p:.4f}"
        lines.append(f"{metric_name:<25} {result.n:>8} {result.spearman_rho:>12.4f} {p_str:>12} {sig:>12}")
    
    # 按年份分析
    if stratified_results.get('by_year'):
        lines.append("\n--- By Year Analysis (original_reviewer_avg) ---")
        for year, data in sorted(stratified_results['by_year'].items()):
            if 'original_reviewer_avg' in data['correlations']:
                corr = data['correlations']['original_reviewer_avg']
                lines.append(f"  {year}: n={data['n']}, ρ={corr['spearman_rho']:.4f}, p={corr['spearman_p']:.4f}")
    
    # 按决策分析
    if stratified_results.get('by_decision'):
        lines.append("\n--- By Decision Analysis (original_reviewer_avg) ---")
        for decision, data in stratified_results['by_decision'].items():
            if 'original_reviewer_avg' in data['correlations']:
                corr = data['correlations']['original_reviewer_avg']
                lines.append(f"  {decision}: n={data['n']}, ρ={corr['spearman_rho']:.4f}, p={corr['spearman_p']:.4f}")
    
    # 结论
    lines.append("\n--- Interpretation ---")
    if 'original_reviewer_avg' in overall_results:
        result = overall_results['original_reviewer_avg']
        if result.significant_001:
            lines.append("✓ Quality-Impact Alignment CONFIRMED: Review scores significantly correlate with citations")
            lines.append(f"  Spearman ρ = {result.spearman_rho:.3f} (p < 0.001)")
        elif result.significant_005:
            lines.append("~ Quality-Impact Alignment WEAKLY CONFIRMED: Correlation significant at p < 0.05")
        else:
            lines.append("✗ Quality-Impact Alignment NOT CONFIRMED: No significant correlation found")
    
    if 'multi_agent_score' in overall_results:
        agent_result = overall_results['multi_agent_score']
        human_result = overall_results.get('original_reviewer_avg')
        if human_result and agent_result.spearman_rho > human_result.spearman_rho:
            lines.append(f"✓ Multi-agent score shows STRONGER correlation (ρ={agent_result.spearman_rho:.3f}) than human reviewers (ρ={human_result.spearman_rho:.3f})")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Experiment 2: Quality-Impact Alignment Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基本用法
  python exp2_quality_impact.py \\
      --citation-dir ./data/iclr_cite_analys \\
      --review-dir ./data/iclr_review_analys \\
      --agent-dir ./data/results \\
      --output ./data/results/exp2_quality_impact.json
  
  # 指定年份
  python exp2_quality_impact.py \\
      --citation-dir ./data/iclr_cite_analys \\
      --review-dir ./data/iclr_review_analys \\
      --agent-dir ./data/results \\
      --years 2018 2019 2020 \\
      --output ./data/results/exp2_quality_impact.json
        """
    )
    
    # 数据路径参数
    parser.add_argument('--citation-dir', type=str, required=True,
                        help='引用数据目录 (包含 citation_iclr_YYYY.json)')
    parser.add_argument('--review-dir', type=str, required=True,
                        help='审稿数据目录 (包含 iclr_YYYY.json)')
    parser.add_argument('--agent-dir', type=str, default=None,
                        help='智能体结果目录 (包含 iclrYYYY_con_MODEL.json)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='输出JSON文件路径')
    
    # 筛选参数
    parser.add_argument('--years', type=int, nargs='+', default=[2018, 2019, 2020, 2021, 2022],
                        help='年份列表 (默认: 2018-2022)')
    parser.add_argument('--agent-model', type=str, default='llama8b',
                        help='智能体模型名称 (默认: llama8b)')
    
    # 输出参数
    parser.add_argument('--latex', action='store_true',
                        help='同时输出LaTeX表格')
    
    args = parser.parse_args()
    
    # 检查依赖
    if not SCIPY_AVAILABLE:
        print("Error: scipy is required. Install with: pip install scipy")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("Experiment 2: Quality-Impact Alignment Validation")
    print(f"{'='*60}\n")
    
    print(f"Parameters:")
    print(f"  Citation dir: {args.citation_dir}")
    print(f"  Review dir: {args.review_dir}")
    print(f"  Agent dir: {args.agent_dir}")
    print(f"  Years: {args.years}")
    print(f"  Agent model: {args.agent_model}")
    print(f"  Output: {args.output}")
    print()
    
    # 加载数据
    print("Loading data...")
    citation_data = load_citation_data(args.citation_dir, args.years)
    print(f"  Citation records: {len(citation_data)}")
    
    review_data = load_review_data(args.review_dir, args.years)
    print(f"  Review records: {len(review_data)}")
    
    agent_data = {}
    if args.agent_dir:
        agent_data = load_agent_data(args.agent_dir, args.years, args.agent_model)
        print(f"  Agent records: {len(agent_data)}")
    
    # 合并数据
    print("\nMerging data...")
    papers = merge_data(citation_data, review_data, agent_data)
    print(f"  Total papers: {len(papers)}")
    
    # 过滤有效数据
    valid_papers = [p for p in papers if p.avg_rating is not None and p.citations is not None]
    print(f"  Valid papers (with rating and citations): {len(valid_papers)}")
    
    if len(valid_papers) < 50:
        print("Error: Not enough valid papers for analysis")
        sys.exit(1)
    
    # 计算描述性统计
    print("\nComputing descriptive statistics...")
    stats = compute_descriptive_statistics(valid_papers)
    
    # 计算整体相关性
    print("Computing overall correlations...")
    overall_results = compute_all_correlations(valid_papers)
    
    # 计算分层分析
    print("Computing stratified analysis...")
    stratified_results = compute_stratified_analysis(valid_papers)
    
    # 生成摘要
    summary = format_summary(overall_results, stratified_results, stats)
    print("\n" + summary)
    
    # 生成LaTeX表格
    latex_table = format_latex_table(overall_results)
    if args.latex:
        print("\n--- LaTeX Table ---")
        print(latex_table)
    
    # 保存结果
    output_data = {
        'experiment': 'exp2_quality_impact_alignment',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'citation_dir': args.citation_dir,
            'review_dir': args.review_dir,
            'agent_dir': args.agent_dir,
            'years': args.years,
            'agent_model': args.agent_model
        },
        'statistics': stats,
        'overall_correlations': {k: v.to_dict() for k, v in overall_results.items()},
        'stratified_analysis': stratified_results,
        'latex_table': latex_table,
        'summary': {
            'total_papers': len(papers),
            'valid_papers': len(valid_papers),
            'quality_impact_confirmed': bool(overall_results.get('original_reviewer_avg', 
                                                             CorrelationResult('', '', 0, 0, 1, 0, 1, False, False)).significant_001)
        }
    }
    
    # 创建输出目录
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {args.output}")
    
    # 打印关键结论
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    
    if 'original_reviewer_avg' in overall_results:
        result = overall_results['original_reviewer_avg']
        print(f"\n1. Original Reviewer Average vs Citations:")
        print(f"   Spearman ρ = {result.spearman_rho:.4f}")
        print(f"   p-value = {'<0.001' if result.spearman_p < 0.001 else f'{result.spearman_p:.4f}'}")
        print(f"   Significant: {'YES' if result.significant_001 else 'NO'}")
    
    if 'multi_agent_score' in overall_results:
        result = overall_results['multi_agent_score']
        print(f"\n2. Multi-Agent Score vs Citations:")
        print(f"   Spearman ρ = {result.spearman_rho:.4f}")
        print(f"   p-value = {'<0.001' if result.spearman_p < 0.001 else f'{result.spearman_p:.4f}'}")
        print(f"   Significant: {'YES' if result.significant_001 else 'NO'}")
    
    if 'decision_binary' in overall_results:
        result = overall_results['decision_binary']
        print(f"\n3. Decision (Accept=1) vs Citations:")
        print(f"   Spearman ρ = {result.spearman_rho:.4f}")
        print(f"   p-value = {'<0.001' if result.spearman_p < 0.001 else f'{result.spearman_p:.4f}'}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()