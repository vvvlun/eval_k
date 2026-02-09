"""
Inter-Reviewer Agreement Calculator
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import itertools

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ReviewerDecision:
    """单个审稿人的决策"""
    reviewer_id: str
    rating: Optional[float]
    decision: str  # 'Accept' or 'Reject'


@dataclass
class PaperReviewerAgreement:
    """单篇论文的审稿人一致性"""
    paper_id: str
    title: str
    num_reviewers: int
    decisions: List[str]  # ['Accept', 'Reject', 'Accept', ...]
    ratings: List[float]
    all_agree: bool
    majority_decision: str
    actual_decision: str
    agreement_rate: float  # 成对一致率


@dataclass
class VenueYearStats:
    """某个 venue-year 的统计"""
    venue: str
    year: int
    threshold: float  # 使用的接受阈值
    total_papers: int
    papers_with_reviews: int
    avg_reviewers_per_paper: float
    
    # 一致性指标
    full_agreement_rate: float  # 所有审稿人都同意的比例
    pairwise_agreement_rate: float  # 成对一致的平均比例
    fleiss_kappa: float  # Fleiss' Kappa
    
    # 按决策类型分
    accept_unanimous_rate: float  # 全员接受的比例
    reject_unanimous_rate: float  # 全员拒稿的比例
    
    # 额外信息
    avg_rating: float
    rating_std: float


# ============================================================================
# 评分阈值配置（按 venue-year）
# ============================================================================

# NeurIPS 评分标准变化历史：
# - 2021: 1-10分, 5="Marginally below acceptance threshold" (拒稿)
# - 2022+: 评分标准调整, 5="Borderline Accept" (接受)
# 
# ICLR 评分标准：
# - 通常 1-10分, 6="Marginally above acceptance threshold"

THRESHOLD_CONFIG = {
    # ICLR: 统一用 6 作为阈值
    'iclr': {
        'default': 6.0,
        # 如果某年有特殊情况，可以单独配置
        # 2018: 6.0,
        # 2019: 6.0,
    },
    # NeurIPS: 2021 和 2022+ 阈值不同
    'neurips': {
        'default': 5.0,  # 2022+ 的默认值
        2021: 6.0,       # 2021年 5分是拒稿，所以阈值是6
        # 2022+: 5分是接受，所以阈值是5
    },
}


def get_threshold(venue: str, year: int, default: float = 6.0) -> float:
    """
    获取指定 venue-year 的接受阈值
    
    Args:
        venue: 会议名称 (iclr, neurips)
        year: 年份
        default: 默认阈值
    
    Returns:
        接受阈值（rating >= threshold 为接受）
    """
    venue_lower = venue.lower()
    
    if venue_lower not in THRESHOLD_CONFIG:
        return default
    
    venue_config = THRESHOLD_CONFIG[venue_lower]
    
    # 先查找特定年份的配置
    if year in venue_config:
        return venue_config[year]
    
    # 使用默认值
    return venue_config.get('default', default)


# ============================================================================
# Rating 解析
# ============================================================================

def extract_rating(rating_str: Any) -> Optional[float]:
    """
    从 rating 字符串中提取数字
    
    Examples:
        "4: Ok but not good enough - rejection" -> 4.0
        "8: Strong Accept" -> 8.0
        "5: Marginally below the acceptance threshold" -> 5.0
        8 -> 8.0
        None -> None
    """
    if rating_str is None:
        return None
    
    # 如果已经是数字
    if isinstance(rating_str, (int, float)):
        return float(rating_str)
    
    # 转为字符串处理
    rating_str = str(rating_str).strip()
    
    if not rating_str:
        return None
    
    # 尝试匹配开头的数字（可能带小数点）
    # 匹配模式: "4:", "4.5:", "4 :", "4" 等
    match = re.match(r'^(\d+(?:\.\d+)?)', rating_str)
    if match:
        return float(match.group(1))
    
    # 尝试在字符串中找数字
    numbers = re.findall(r'\d+(?:\.\d+)?', rating_str)
    if numbers:
        return float(numbers[0])
    
    return None


def rating_to_decision(rating: Optional[float], threshold: float = 6.0) -> Optional[str]:
    """
    将 rating 转换为二元决策
    
    Args:
        rating: 评分数字
        threshold: 接受阈值，>= threshold 为接受
    
    Returns:
        'Accept', 'Reject', 或 None
    """
    if rating is None:
        return None
    return 'Accept' if rating >= threshold else 'Reject'


# ============================================================================
# 一致性计算
# ============================================================================

def compute_pairwise_agreement(decisions: List[str]) -> float:
    """
    计算成对一致率
    
    对于 n 个审稿人，计算所有 C(n,2) 对中一致的比例
    """
    if len(decisions) < 2:
        return 1.0
    
    pairs = list(itertools.combinations(decisions, 2))
    if not pairs:
        return 1.0
    
    agreements = sum(1 for d1, d2 in pairs if d1 == d2)
    return agreements / len(pairs)


def compute_fleiss_kappa(papers_decisions: List[List[str]]) -> float:
    """
    计算 Fleiss' Kappa（多评分者一致性）
    
    Args:
        papers_decisions: 每篇论文的决策列表 [['Accept', 'Reject', 'Accept'], ...]
    
    Returns:
        Fleiss' Kappa 值
    
    Note:
        Fleiss' Kappa 适用于多个评分者对多个对象进行分类的情况
        范围: [-1, 1]，1 表示完全一致，0 表示随机一致，负值表示低于随机
    """
    if not papers_decisions:
        return 0.0
    
    # 类别
    categories = ['Accept', 'Reject']
    n_categories = len(categories)
    
    # 过滤掉评分者数量不一致或太少的论文
    # Fleiss' Kappa 需要每个对象有相同数量的评分者
    reviewer_counts = [len(d) for d in papers_decisions]
    if not reviewer_counts:
        return 0.0
    
    # 使用最常见的评分者数量
    from collections import Counter
    count_freq = Counter(reviewer_counts)
    most_common_count = count_freq.most_common(1)[0][0]
    
    # 只保留有足够评分者的论文
    filtered = [d for d in papers_decisions if len(d) == most_common_count and most_common_count >= 2]
    
    if len(filtered) < 2:
        return 0.0
    
    n = len(filtered)  # 论文数
    k = most_common_count  # 每篇论文的评分者数
    
    # 构建计数矩阵：每篇论文每个类别的投票数
    # matrix[i][j] = 论文 i 中选择类别 j 的评分者数
    matrix = []
    for decisions in filtered:
        counts = [decisions.count(cat) for cat in categories]
        matrix.append(counts)
    
    # 计算每个类别的总体比例 p_j
    total_votes = n * k
    p = [sum(row[j] for row in matrix) / total_votes for j in range(n_categories)]
    
    # 计算 P_e（期望的随机一致性）
    P_e = sum(p_j ** 2 for p_j in p)
    
    # 计算每篇论文的一致性 P_i
    P_i_list = []
    for row in matrix:
        if k <= 1:
            P_i_list.append(1.0)
        else:
            P_i = (sum(n_ij ** 2 for n_ij in row) - k) / (k * (k - 1))
            P_i_list.append(P_i)
    
    # 计算 P（观察到的一致性）
    P = sum(P_i_list) / n
    
    # 计算 Fleiss' Kappa
    if P_e == 1:
        return 1.0 if P == 1 else 0.0
    
    kappa = (P - P_e) / (1 - P_e)
    return kappa


def analyze_paper(paper: Dict, threshold: float = 6.0) -> Optional[PaperReviewerAgreement]:
    """
    分析单篇论文的审稿人一致性
    """
    paper_id = paper.get('paper_id', 'unknown')
    title = paper.get('title', 'Unknown')
    reviews = paper.get('reviews', [])
    actual_decision = paper.get('decision', 'Unknown')
    
    if not reviews or len(reviews) < 2:
        return None
    
    # 提取每个审稿人的决策
    ratings = []
    decisions = []
    
    for review in reviews:
        rating = extract_rating(review.get('rating'))
        if rating is not None:
            decision = rating_to_decision(rating, threshold)
            ratings.append(rating)
            decisions.append(decision)
    
    if len(decisions) < 2:
        return None
    
    # 计算一致性
    all_agree = len(set(decisions)) == 1
    accept_count = decisions.count('Accept')
    majority_decision = 'Accept' if accept_count > len(decisions) / 2 else 'Reject'
    agreement_rate = compute_pairwise_agreement(decisions)
    
    return PaperReviewerAgreement(
        paper_id=paper_id,
        title=title[:100],
        num_reviewers=len(decisions),
        decisions=decisions,
        ratings=ratings,
        all_agree=all_agree,
        majority_decision=majority_decision,
        actual_decision=actual_decision,
        agreement_rate=agreement_rate
    )


def compute_venue_year_stats(
    papers: List[Dict], 
    venue: str, 
    year: int,
    threshold: float = None  # None 表示使用自动阈值
) -> VenueYearStats:
    """
    计算某个 venue-year 的一致性统计
    
    Args:
        papers: 论文列表
        venue: 会议名称
        year: 年份
        threshold: 手动指定阈值，None 则使用自动配置
    """
    # 获取阈值
    if threshold is None:
        threshold = get_threshold(venue, year)
    
    # 分析每篇论文
    paper_results = []
    all_ratings = []
    all_decisions_lists = []
    
    for paper in papers:
        result = analyze_paper(paper, threshold)
        if result:
            paper_results.append(result)
            all_ratings.extend(result.ratings)
            all_decisions_lists.append(result.decisions)
    
    if not paper_results:
        return VenueYearStats(
            venue=venue, year=year, threshold=threshold,
            total_papers=len(papers), papers_with_reviews=0,
            avg_reviewers_per_paper=0, full_agreement_rate=0,
            pairwise_agreement_rate=0, fleiss_kappa=0,
            accept_unanimous_rate=0, reject_unanimous_rate=0,
            avg_rating=0, rating_std=0
        )
    
    # 统计
    total_papers = len(papers)
    papers_with_reviews = len(paper_results)
    avg_reviewers = sum(r.num_reviewers for r in paper_results) / papers_with_reviews
    
    # 一致性指标
    full_agreement_count = sum(1 for r in paper_results if r.all_agree)
    full_agreement_rate = full_agreement_count / papers_with_reviews
    
    pairwise_rates = [r.agreement_rate for r in paper_results]
    pairwise_agreement_rate = sum(pairwise_rates) / len(pairwise_rates)
    
    fleiss_kappa = compute_fleiss_kappa(all_decisions_lists)
    
    # 按决策类型
    accept_unanimous = sum(1 for r in paper_results if r.all_agree and r.decisions[0] == 'Accept')
    reject_unanimous = sum(1 for r in paper_results if r.all_agree and r.decisions[0] == 'Reject')
    
    # Rating 统计
    avg_rating = sum(all_ratings) / len(all_ratings) if all_ratings else 0
    if len(all_ratings) > 1:
        variance = sum((r - avg_rating) ** 2 for r in all_ratings) / len(all_ratings)
        rating_std = variance ** 0.5
    else:
        rating_std = 0
    
    return VenueYearStats(
        venue=venue,
        year=year,
        threshold=threshold,
        total_papers=total_papers,
        papers_with_reviews=papers_with_reviews,
        avg_reviewers_per_paper=round(avg_reviewers, 2),
        full_agreement_rate=round(full_agreement_rate, 4),
        pairwise_agreement_rate=round(pairwise_agreement_rate, 4),
        fleiss_kappa=round(fleiss_kappa, 4),
        accept_unanimous_rate=round(accept_unanimous / papers_with_reviews, 4) if papers_with_reviews else 0,
        reject_unanimous_rate=round(reject_unanimous / papers_with_reviews, 4) if papers_with_reviews else 0,
        avg_rating=round(avg_rating, 2),
        rating_std=round(rating_std, 2)
    )


# ============================================================================
# 数据加载
# ============================================================================

def load_papers(filepath: str) -> List[Dict]:
    """加载论文 JSON 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    return papers


def find_data_files(data_dir: str, iclr_years: List[int] = None, neurips_years: List[int] = None) -> Dict[str, List[Tuple[int, str]]]:
    """
    在目录中查找数据文件
    
    Returns:
        {'iclr': [(2018, 'path/to/iclr_2018.json'), ...], 'neurips': [...]}
    """
    data_dir = Path(data_dir)
    
    if iclr_years is None:
        iclr_years = list(range(2018, 2026))  # 2018-2025
    if neurips_years is None:
        neurips_years = list(range(2021, 2026))  # 2021-2025
    
    result = {'iclr': [], 'neurips': []}
    
    for year in iclr_years:
        path = data_dir / f"iclr_{year}.json"
        if path.exists():
            result['iclr'].append((year, str(path)))
    
    for year in neurips_years:
        path = data_dir / f"neurips_{year}.json"
        if path.exists():
            result['neurips'].append((year, str(path)))
    
    return result


# ============================================================================
# 主函数
# ============================================================================

def print_stats(stats: VenueYearStats):
    """打印单个 venue-year 的统计"""
    print(f"\n  {stats.venue.upper()} {stats.year} (threshold >= {stats.threshold}):")
    print(f"    Papers: {stats.papers_with_reviews} (with reviews) / {stats.total_papers} (total)")
    print(f"    Avg reviewers/paper: {stats.avg_reviewers_per_paper}")
    print(f"    Avg rating: {stats.avg_rating} ± {stats.rating_std}")
    print(f"    Full Agreement Rate: {stats.full_agreement_rate:.2%}")
    print(f"    Pairwise Agreement:  {stats.pairwise_agreement_rate:.2%}")
    print(f"    Fleiss' Kappa:       {stats.fleiss_kappa:.4f}")
    print(f"    Unanimous Accept:    {stats.accept_unanimous_rate:.2%}")
    print(f"    Unanimous Reject:    {stats.reject_unanimous_rate:.2%}")


def compute_aggregate_stats(all_stats: List[VenueYearStats]) -> Dict:
    """计算聚合统计"""
    if not all_stats:
        return {}
    
    total_papers = sum(s.papers_with_reviews for s in all_stats)
    
    # 加权平均（按论文数加权）
    weighted_full_agreement = sum(s.full_agreement_rate * s.papers_with_reviews for s in all_stats) / total_papers
    weighted_pairwise = sum(s.pairwise_agreement_rate * s.papers_with_reviews for s in all_stats) / total_papers
    weighted_kappa = sum(s.fleiss_kappa * s.papers_with_reviews for s in all_stats) / total_papers
    
    return {
        'total_papers': total_papers,
        'num_venue_years': len(all_stats),
        'weighted_full_agreement_rate': round(weighted_full_agreement, 4),
        'weighted_pairwise_agreement': round(weighted_pairwise, 4),
        'weighted_fleiss_kappa': round(weighted_kappa, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description='计算审稿人之间的一致性（Inter-Reviewer Agreement）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 计算单个文件（使用自动阈值）
  python inter_reviewer_agreement.py --data ./data/raw/openreview/iclr_2018.json
  
  # 计算整个目录（使用自动阈值，推荐）
  python inter_reviewer_agreement.py --data-dir ./data/raw/openreview
  
  # 手动指定统一阈值（覆盖自动配置）
  python inter_reviewer_agreement.py --data-dir ./data/raw/openreview --threshold 5.5
  
  # 指定年份
  python inter_reviewer_agreement.py --data-dir ./data/raw/openreview --iclr-years 2018 2019 2020

Threshold Configuration:
  默认使用自动阈值配置（按 venue-year）：
  - ICLR: 统一 >= 6 为接受
  - NeurIPS 2021: >= 6 为接受 (5分是 "marginally below")
  - NeurIPS 2022+: >= 5 为接受 (5分是 "borderline accept")
  
  使用 --threshold 可覆盖自动配置

Output:
  控制台输出 + JSON 结果文件
        """
    )
    
    parser.add_argument('--data', type=str, help='单个数据文件路径')
    parser.add_argument('--data-dir', type=str, help='数据目录（自动查找 iclr_*.json 和 neurips_*.json）')
    parser.add_argument('--output', '-o', type=str, default='./data/results',
                       help='输出目录')
    parser.add_argument('--threshold', type=float, default=None,
                       help='手动指定接受阈值（覆盖自动配置）。不指定则使用自动阈值。')
    parser.add_argument('--iclr-years', type=int, nargs='+', default=None,
                       help='ICLR 年份列表（默认 2018-2025）')
    parser.add_argument('--neurips-years', type=int, nargs='+', default=None,
                       help='NeurIPS 年份列表（默认 2021-2025）')
    
    args = parser.parse_args()
    
    if not args.data and not args.data_dir:
        parser.error("必须指定 --data 或 --data-dir")
    
    print(f"\n{'='*60}")
    print("Inter-Reviewer Agreement Calculator")
    print(f"{'='*60}")
    
    if args.threshold is not None:
        print(f"\nThreshold mode: MANUAL (rating >= {args.threshold} for all)")
    else:
        print(f"\nThreshold mode: AUTO (venue-year specific)")
        print("  - ICLR: >= 6.0")
        print("  - NeurIPS 2021: >= 6.0")
        print("  - NeurIPS 2022+: >= 5.0")
    
    all_stats = []
    venue_stats = {'iclr': [], 'neurips': []}
    
    if args.data:
        # 单个文件
        filepath = Path(args.data)
        if not filepath.exists():
            print(f"Error: File not found: {args.data}")
            return
        
        # 从文件名解析 venue 和 year
        filename = filepath.stem
        parts = filename.split('_')
        venue = parts[0] if parts else 'unknown'
        year = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        
        papers = load_papers(str(filepath))
        stats = compute_venue_year_stats(papers, venue, year, args.threshold)
        all_stats.append(stats)
        if venue in venue_stats:
            venue_stats[venue].append(stats)
        
        print_stats(stats)
    
    else:
        # 目录模式
        data_files = find_data_files(args.data_dir, args.iclr_years, args.neurips_years)
        
        if not data_files['iclr'] and not data_files['neurips']:
            print(f"Error: No data files found in {args.data_dir}")
            return
        
        # 处理 ICLR
        if data_files['iclr']:
            print(f"\n{'='*40}")
            print("ICLR")
            print(f"{'='*40}")
            
            for year, filepath in sorted(data_files['iclr']):
                papers = load_papers(filepath)
                stats = compute_venue_year_stats(papers, 'iclr', year, args.threshold)
                all_stats.append(stats)
                venue_stats['iclr'].append(stats)
                print_stats(stats)
        
        # 处理 NeurIPS
        if data_files['neurips']:
            print(f"\n{'='*40}")
            print("NeurIPS")
            print(f"{'='*40}")
            
            for year, filepath in sorted(data_files['neurips']):
                papers = load_papers(filepath)
                stats = compute_venue_year_stats(papers, 'neurips', year, args.threshold)
                all_stats.append(stats)
                venue_stats['neurips'].append(stats)
                print_stats(stats)
    
    # 聚合统计
    print(f"\n{'='*60}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*60}")
    
    if venue_stats['iclr']:
        iclr_agg = compute_aggregate_stats(venue_stats['iclr'])
        print(f"\n  ICLR (all years):")
        print(f"    Total papers: {iclr_agg['total_papers']}")
        print(f"    Full Agreement Rate: {iclr_agg['weighted_full_agreement_rate']:.2%}")
        print(f"    Pairwise Agreement:  {iclr_agg['weighted_pairwise_agreement']:.2%}")
        print(f"    Fleiss' Kappa:       {iclr_agg['weighted_fleiss_kappa']:.4f}")
    
    if venue_stats['neurips']:
        neurips_agg = compute_aggregate_stats(venue_stats['neurips'])
        print(f"\n  NeurIPS (all years):")
        print(f"    Total papers: {neurips_agg['total_papers']}")
        print(f"    Full Agreement Rate: {neurips_agg['weighted_full_agreement_rate']:.2%}")
        print(f"    Pairwise Agreement:  {neurips_agg['weighted_pairwise_agreement']:.2%}")
        print(f"    Fleiss' Kappa:       {neurips_agg['weighted_fleiss_kappa']:.4f}")
    
    overall_agg = compute_aggregate_stats(all_stats)
    print(f"\n  OVERALL (all venues, all years):")
    print(f"    Total papers: {overall_agg['total_papers']}")
    print(f"    Full Agreement Rate: {overall_agg['weighted_full_agreement_rate']:.2%}")
    print(f"    Pairwise Agreement:  {overall_agg['weighted_pairwise_agreement']:.2%}")
    print(f"    Fleiss' Kappa:       {overall_agg['weighted_fleiss_kappa']:.4f}")
    
    # 保存结果
    Path(args.output).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) / f"inter_reviewer_agreement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'threshold': args.threshold,
        'by_venue_year': [asdict(s) for s in all_stats],
        'aggregate': {
            'iclr': compute_aggregate_stats(venue_stats['iclr']) if venue_stats['iclr'] else {},
            'neurips': compute_aggregate_stats(venue_stats['neurips']) if venue_stats['neurips'] else {},
            'overall': overall_agg
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")
    
    # 解释
    print("INTERPRETATION:")
    print("-" * 40)
    print("• Full Agreement Rate: 所有审稿人都同意的论文比例")
    print("• Pairwise Agreement: 任意两个审稿人一致的平均比例")
    print("• Fleiss' Kappa: 多评分者一致性指标")
    print("  - κ > 0.75: 优秀一致性")
    print("  - κ = 0.40-0.75: 中等一致性")
    print("  - κ < 0.40: 较差一致性")
    print()
    print("NOTE: NeurIPS 2014/2021 实验显示人类审稿人之间")
    print("      的 Kappa 约为 0.25，即约 50% 的决策是随机的。")
    print()


if __name__ == '__main__':
    main()
