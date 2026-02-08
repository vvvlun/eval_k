"""
Evaluation Metrics for Peer Review Analysis.

Computes Accuracy, Cohen's Kappa, Confusion Matrix, and other metrics
for comparing simulated decisions with real decisions.

Usage:
    from evaluation.metrics import compute_all_metrics
    metrics = compute_all_metrics(simulated_decisions, real_decisions)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import json


@dataclass
class ConfusionMatrix:
    """混淆矩阵"""
    true_positive: int  # 真实Accept且模拟Accept
    true_negative: int  # 真实Reject且模拟Reject
    false_positive: int  # 真实Reject但模拟Accept
    false_negative: int  # 真实Accept但模拟Reject
    
    @property
    def total(self) -> int:
        return self.true_positive + self.true_negative + self.false_positive + self.false_negative
    
    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.true_positive + self.true_negative) / self.total
    
    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        if denominator == 0:
            return 0.0
        return self.true_positive / denominator
    
    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        if denominator == 0:
            return 0.0
        return self.true_positive / denominator
    
    @property
    def f1_score(self) -> float:
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
    
    @property
    def false_positive_rate(self) -> float:
        denominator = self.false_positive + self.true_negative
        if denominator == 0:
            return 0.0
        return self.false_positive / denominator
    
    @property
    def false_negative_rate(self) -> float:
        denominator = self.true_positive + self.false_negative
        if denominator == 0:
            return 0.0
        return self.false_negative / denominator
    
    def to_dict(self) -> Dict:
        return {
            "true_positive": self.true_positive,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "fpr": round(self.false_positive_rate, 4),
            "fnr": round(self.false_negative_rate, 4)
        }
    
    def __str__(self) -> str:
        return f"""
Confusion Matrix:
                 Predicted
              Accept    Reject
Actual Accept   {self.true_positive:5d}     {self.false_negative:5d}
       Reject   {self.false_positive:5d}     {self.true_negative:5d}

Accuracy:  {self.accuracy:.4f}
Precision: {self.precision:.4f}
Recall:    {self.recall:.4f}
F1 Score:  {self.f1_score:.4f}
"""


@dataclass
class EvaluationMetrics:
    """完整评估指标"""
    accuracy: float
    cohens_kappa: float
    confusion_matrix: ConfusionMatrix
    agreement_rate: float
    
    # 详细统计
    total_papers: int
    simulated_accept: int
    simulated_reject: int
    real_accept: int
    real_reject: int
    
    def to_dict(self) -> Dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "cohens_kappa": round(self.cohens_kappa, 4),
            "agreement_rate": round(self.agreement_rate, 4),
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "total_papers": self.total_papers,
            "simulated_accept": self.simulated_accept,
            "simulated_reject": self.simulated_reject,
            "real_accept": self.real_accept,
            "real_reject": self.real_reject
        }
    
    def summary(self) -> str:
        return f"""
=== Evaluation Metrics ===
Total Papers: {self.total_papers}

Distribution:
  Real:      Accept={self.real_accept} ({100*self.real_accept/self.total_papers:.1f}%), Reject={self.real_reject} ({100*self.real_reject/self.total_papers:.1f}%)
  Simulated: Accept={self.simulated_accept} ({100*self.simulated_accept/self.total_papers:.1f}%), Reject={self.simulated_reject} ({100*self.simulated_reject/self.total_papers:.1f}%)

Agreement:
  Accuracy:     {self.accuracy:.4f} ({100*self.accuracy:.2f}%)
  Cohen's Kappa: {self.cohens_kappa:.4f}
  
{self.confusion_matrix}
"""


def compute_confusion_matrix(
    simulated: Dict[str, str], 
    real: Dict[str, str]
) -> ConfusionMatrix:
    """
    计算混淆矩阵
    
    Args:
        simulated: {paper_id: 'Accept'/'Reject'}
        real: {paper_id: 'Accept'/'Reject'}
    
    Returns:
        ConfusionMatrix对象
    """
    tp = tn = fp = fn = 0
    
    for paper_id, sim_decision in simulated.items():
        real_decision = real.get(paper_id)
        if real_decision is None:
            continue
        
        # 统一格式
        sim_accept = sim_decision.lower().startswith('accept')
        real_accept = real_decision.lower().startswith('accept')
        
        if real_accept and sim_accept:
            tp += 1
        elif not real_accept and not sim_accept:
            tn += 1
        elif not real_accept and sim_accept:
            fp += 1
        else:  # real_accept and not sim_accept
            fn += 1
    
    return ConfusionMatrix(
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn
    )


def compute_cohens_kappa(
    simulated: Dict[str, str], 
    real: Dict[str, str]
) -> float:
    """
    计算Cohen's Kappa
    
    衡量排除随机一致性后的一致性程度。
    
    Kappa解释：
    - < 0: 低于随机水平
    - 0.01-0.20: 轻微一致
    - 0.21-0.40: 尚可一致
    - 0.41-0.60: 中等一致
    - 0.61-0.80: 高度一致
    - 0.81-1.00: 几乎完全一致
    """
    # 获取共同的paper_ids
    common_ids = set(simulated.keys()) & set(real.keys())
    n = len(common_ids)
    
    if n == 0:
        return 0.0
    
    # 计算观察一致率
    agreements = sum(
        1 for pid in common_ids
        if simulated[pid].lower().startswith('accept') == real[pid].lower().startswith('accept')
    )
    p_o = agreements / n
    
    # 计算期望一致率
    sim_accept = sum(1 for pid in common_ids if simulated[pid].lower().startswith('accept'))
    real_accept = sum(1 for pid in common_ids if real[pid].lower().startswith('accept'))
    
    p_sim_accept = sim_accept / n
    p_real_accept = real_accept / n
    
    p_e = (p_sim_accept * p_real_accept) + ((1 - p_sim_accept) * (1 - p_real_accept))
    
    # Kappa
    if p_e == 1:
        return 1.0 if p_o == 1 else 0.0
    
    kappa = (p_o - p_e) / (1 - p_e)
    
    return kappa


def compute_all_metrics(
    simulated: Dict[str, str], 
    real: Dict[str, str]
) -> EvaluationMetrics:
    """
    计算所有评估指标
    
    Args:
        simulated: {paper_id: 'Accept'/'Reject'}
        real: {paper_id: 'Accept'/'Reject'}
    
    Returns:
        EvaluationMetrics对象
    """
    # 获取共同的paper_ids
    common_ids = set(simulated.keys()) & set(real.keys())
    n = len(common_ids)
    
    if n == 0:
        raise ValueError("No common papers between simulated and real decisions")
    
    # 混淆矩阵
    cm = compute_confusion_matrix(simulated, real)
    
    # Cohen's Kappa
    kappa = compute_cohens_kappa(simulated, real)
    
    # 统计
    sim_accept = sum(1 for pid in common_ids if simulated[pid].lower().startswith('accept'))
    sim_reject = n - sim_accept
    real_accept = sum(1 for pid in common_ids if real[pid].lower().startswith('accept'))
    real_reject = n - real_accept
    
    return EvaluationMetrics(
        accuracy=cm.accuracy,
        cohens_kappa=kappa,
        confusion_matrix=cm,
        agreement_rate=cm.accuracy,
        total_papers=n,
        simulated_accept=sim_accept,
        simulated_reject=sim_reject,
        real_accept=real_accept,
        real_reject=real_reject
    )


def compute_spearman_correlation(
    scores: Dict[str, float],
    ground_truth: Dict[str, float]
) -> Tuple[float, float]:
    """
    计算Spearman相关系数
    
    Args:
        scores: {paper_id: score}
        ground_truth: {paper_id: impact_score}
    
    Returns:
        (correlation, p_value)
    """
    try:
        from scipy import stats
    except ImportError:
        # Fallback实现
        return _spearman_fallback(scores, ground_truth)
    
    common_ids = list(set(scores.keys()) & set(ground_truth.keys()))
    if len(common_ids) < 3:
        return 0.0, 1.0
    
    x = [scores[pid] for pid in common_ids]
    y = [ground_truth[pid] for pid in common_ids]
    
    rho, p_value = stats.spearmanr(x, y)
    
    return float(rho), float(p_value)


def _spearman_fallback(
    scores: Dict[str, float],
    ground_truth: Dict[str, float]
) -> Tuple[float, float]:
    """Spearman相关系数的简单实现"""
    common_ids = list(set(scores.keys()) & set(ground_truth.keys()))
    n = len(common_ids)
    
    if n < 3:
        return 0.0, 1.0
    
    # 获取排名
    x = [scores[pid] for pid in common_ids]
    y = [ground_truth[pid] for pid in common_ids]
    
    x_ranks = _get_ranks(x)
    y_ranks = _get_ranks(y)
    
    # 计算差的平方和
    d_squared_sum = sum((x_ranks[i] - y_ranks[i]) ** 2 for i in range(n))
    
    # Spearman公式
    rho = 1 - (6 * d_squared_sum) / (n * (n ** 2 - 1))
    
    # 简化的p值估计
    t_stat = rho * np.sqrt((n - 2) / (1 - rho ** 2)) if abs(rho) < 1 else 0
    p_value = 2 * (1 - _t_cdf(abs(t_stat), n - 2))
    
    return float(rho), float(p_value)


def _get_ranks(values: List[float]) -> List[float]:
    """计算排名"""
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for rank, idx in enumerate(sorted_indices, 1):
        ranks[idx] = float(rank)
    return ranks


def _t_cdf(t: float, df: int) -> float:
    """T分布CDF的简单近似"""
    # 简化实现，仅用于估计
    import math
    x = df / (df + t ** 2)
    return 0.5 + 0.5 * math.copysign(1 - x ** (df / 2), t)


def mcnemar_test(
    simulated: Dict[str, str],
    real: Dict[str, str]
) -> Dict[str, float]:
    """
    McNemar检验 - 用于比较两种决策方法
    
    检验两种方法的一致性是否存在显著差异
    """
    common_ids = list(set(simulated.keys()) & set(real.keys()))
    
    # 计算不一致的类型
    b = 0  # simulated=Accept, real=Reject
    c = 0  # simulated=Reject, real=Accept
    
    for pid in common_ids:
        sim_accept = simulated[pid].lower().startswith('accept')
        real_accept = real[pid].lower().startswith('accept')
        
        if sim_accept and not real_accept:
            b += 1
        elif not sim_accept and real_accept:
            c += 1
    
    # McNemar statistic
    if b + c == 0:
        return {"chi_squared": 0.0, "p_value": 1.0, "significant": False}
    
    chi_squared = (abs(b - c) - 1) ** 2 / (b + c)
    
    # 简化的p值计算（卡方分布，df=1）
    import math
    p_value = math.exp(-chi_squared / 2)
    
    return {
        "chi_squared": round(chi_squared, 4),
        "p_value": round(p_value, 4),
        "b": b,  # sim Accept, real Reject
        "c": c,  # sim Reject, real Accept
        "significant": p_value < 0.05
    }


if __name__ == "__main__":
    # 测试代码
    sim = {
        "p1": "Accept", "p2": "Accept", "p3": "Reject", "p4": "Reject", "p5": "Accept"
    }
    real = {
        "p1": "Accept", "p2": "Reject", "p3": "Reject", "p4": "Accept", "p5": "Accept"
    }
    
    metrics = compute_all_metrics(sim, real)
    print(metrics.summary())
    print(f"\nJSON output:\n{json.dumps(metrics.to_dict(), indent=2)}")
