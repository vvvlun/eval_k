"""
Experiment 1: Simulation Validity Test

验证多智能体系统能否有效模拟人类审稿决策。

Usage:
    # 测试10篇论文
    python exp1_simulation_validity.py --data ./data/raw/openreview/iclr_2018.json --limit 10 --model llama3.1:8b
    
    # 完整实验
    python exp1_simulation_validity.py --data ./data/raw/openreview/iclr_2018.json --model llama3.1:70b
    
    # 使用qwen
    python exp1_simulation_validity.py --data ./data/raw/openreview/iclr_2018.json --model qwen2.5:72b

Output:
    保存至: {output}/exp1_results_{timestamp}.json
    包含: 评估指标、每篇论文的决策详情
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import asdict

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Config, LLMConfig
from agents.llm_client import OllamaClient, test_ollama_connection
from agents.base_agent import PaperInfo
from agents.expert_agents import MultiDimensionalAnalyzer, AggregatedEvaluation
from agents.chair_agent import ChairAgent, ChairDecision, iterative_calibration
from evaluation.metrics import compute_all_metrics, EvaluationMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_papers(data_path: str, limit: Optional[int] = None) -> List[Dict]:
    """加载论文数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # 过滤有reviews和decision的论文
    valid_papers = [
        p for p in papers 
        if p.get('reviews') and p.get('decision') and len(p['reviews']) > 0
    ]
    
    if limit:
        valid_papers = valid_papers[:limit]
    
    logger.info(f"Loaded {len(valid_papers)} valid papers from {data_path}")
    return valid_papers


def papers_to_paperinfo(papers: List[Dict]) -> List[PaperInfo]:
    """转换为PaperInfo对象"""
    return [PaperInfo.from_dict(p) for p in papers]


def get_real_decisions(papers: List[PaperInfo]) -> Dict[str, str]:
    """获取真实决策"""
    return {p.paper_id: p.decision for p in papers}


def run_expert_evaluation(
    papers: List[PaperInfo],
    analyzer: MultiDimensionalAnalyzer,
    cache_dir: Optional[str] = None
) -> Dict[str, AggregatedEvaluation]:
    """
    运行Expert Agent评估
    
    Args:
        papers: 论文列表
        analyzer: 多维度分析器
        cache_dir: 缓存目录（可选）
    
    Returns:
        {paper_id: AggregatedEvaluation}
    """
    results = {}
    total = len(papers)
    start_time = time.time()
    
    # 尝试加载缓存
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / "expert_evaluations.json"
        if cache_path.exists():
            logger.info(f"Loading cached expert evaluations from {cache_path}")
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            # TODO: 实现缓存加载
    
    for i, paper in enumerate(papers):
        try:
            logger.info(f"[{i+1}/{total}] Evaluating: {paper.title[:50]}...")
            
            result = analyzer.analyze_paper(paper)
            results[paper.paper_id] = result
            
            # 进度日志
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (total - i - 1)
            
            logger.info(
                f"  Score: {result.weighted_score:.2f}, "
                f"Sentiment: {result.overall_sentiment}, "
                f"ETA: {remaining/60:.1f} min"
            )
            
        except Exception as e:
            logger.error(f"Error evaluating {paper.paper_id}: {e}")
            continue
    
    # 保存缓存
    if cache_dir and results:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_path = Path(cache_dir) / "expert_evaluations.json"
        cache_data = {
            pid: result.to_dict() for pid, result in results.items()
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        logger.info(f"Saved expert evaluations to {cache_path}")
    
    return results


def run_chair_decisions(
    papers: List[PaperInfo],
    expert_results: Dict[str, AggregatedEvaluation],
    chair_agent: ChairAgent,
    real_accept_count: int,
    use_calibration: bool = True
) -> Dict[str, ChairDecision]:
    """
    运行Chair Agent决策
    
    Args:
        papers: 论文列表
        expert_results: Expert评估结果
        chair_agent: Chair Agent
        real_accept_count: 真实接受数（用于校准）
        use_calibration: 是否使用迭代校准
    
    Returns:
        {paper_id: ChairDecision}
    """
    # 组装输入
    papers_with_evals = [
        (p, expert_results[p.paper_id])
        for p in papers
        if p.paper_id in expert_results
    ]
    
    if use_calibration:
        logger.info(f"Running Chair Agent with iterative calibration (target: {real_accept_count})")
        decisions = iterative_calibration(
            papers_with_evals=papers_with_evals,
            chair_agent=chair_agent,
            real_accept_count=real_accept_count
        )
    else:
        logger.info("Running Chair Agent without calibration")
        decisions = {}
        for paper_info, expert_result in papers_with_evals:
            decision = chair_agent.decide(paper_info, expert_result)
            decisions[paper_info.paper_id] = decision
    
    return decisions


def parse_input_filename(data_path: str) -> Tuple[str, int]:
    """
    从输入文件路径解析venue和year
    
    例如: /home/yuwl/ai_review/data/raw/iclr_2018.json -> ('iclr', 2018)
    """
    filename = Path(data_path).stem  # 获取不带扩展名的文件名，如 "iclr_2018"
    parts = filename.split('_')
    
    venue = parts[0] if parts else 'unknown'
    year = 2018  # 默认值
    
    # 尝试从文件名提取年份
    for part in parts:
        if part.isdigit() and len(part) == 4:
            year = int(part)
            break
    
    return venue, year


def get_model_short_name(model: str) -> str:
    """
    获取模型的简短名称
    
    例如: 
        llama3.1:latest -> llama8b
        llama3.1:8b -> llama8b
        llama3.1:70b -> llama70b
        qwen2.5:72b -> qwen72b
        qwen2.5:latest -> qwen7b
    """
    model_lower = model.lower()
    
    # 提取基础模型名
    base_name = model.split(':')[0].replace('.', '').replace('-', '')
    # 简化名称
    if 'llama' in base_name:
        base_name = 'llama'
    elif 'qwen' in base_name:
        base_name = 'qwen'
    elif 'deepseek' in base_name:
        base_name = 'deepseek'
    
    # 提取模型大小
    size = '8b'  # 默认
    if ':' in model:
        tag = model.split(':')[1].lower()
        if '70b' in tag:
            size = '70b'
        elif '72b' in tag:
            size = '72b'
        elif '30b' in tag:
            size = '30b'
        elif '14b' in tag:
            size = '14b'
        elif '8b' in tag:
            size = '8b'
        elif '7b' in tag:
            size = '7b'
        elif 'latest' in tag:
            # latest通常是较小的模型
            if 'llama' in model_lower:
                size = '8b'
            elif 'qwen' in model_lower:
                size = '7b'
            else:
                size = '7b'
    
    return f"{base_name}{size}"


def save_results(
    output_dir: str,
    metrics: EvaluationMetrics,
    decisions: Dict[str, ChairDecision],
    expert_results: Dict[str, AggregatedEvaluation],
    real_decisions: Dict[str, str],
    config: Dict,
    venue: str,
    year: int,
    model_short: str
) -> str:
    """保存实验结果"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 新命名格式: iclr2018_con_llama8b.json
    output_filename = f"{venue}{year}_con_{model_short}.json"
    output_path = Path(output_dir) / output_filename
    
    # 构建详细结果
    paper_details = []
    for paper_id, decision in decisions.items():
        expert_result = expert_results.get(paper_id)
        detail = {
            "paper_id": paper_id,
            "real_decision": real_decisions.get(paper_id, "Unknown"),
            "simulated_decision": decision.decision,
            "agreement": decision.decision == real_decisions.get(paper_id),
            "chair_confidence": decision.confidence,
            "expert_scores": decision.expert_summary,
            "key_factors": decision.key_factors
        }
        if expert_result:
            detail["weighted_score"] = expert_result.weighted_score
            detail["overall_sentiment"] = expert_result.overall_sentiment
        paper_details.append(detail)
    
    results = {
        "experiment": "exp1_simulation_validity",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": config,
        "metrics": metrics.to_dict(),
        "summary": {
            "accuracy": metrics.accuracy,
            "cohens_kappa": metrics.cohens_kappa,
            "total_papers": metrics.total_papers,
            "agreement_rate": metrics.agreement_rate
        },
        "paper_details": paper_details
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 1: Simulation Validity Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 10 papers
  python exp1_simulation_validity.py --data ./data/raw/iclr_2018.json --limit 10

  # Full experiment with larger model
  python exp1_simulation_validity.py --data ./data/raw/iclr_2018.json --model llama3.1:70b

  # Use qwen2.5
  python exp1_simulation_validity.py --data ./data/raw/iclr_2018.json --model qwen2.5:72b

Output filename format: {venue}{year}_con_{model}.json
  e.g., iclr2018_con_llama8b.json
        """
    )
    
    # 数据参数
    parser.add_argument('--data', '-d', type=str, required=True,
                       help='Path to paper data JSON file (e.g., iclr_2018.json)')
    parser.add_argument('--output', '-o', type=str, default='./data/results',
                       help='Output directory for results')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of papers to process (for testing)')
    
    # 模型参数
    parser.add_argument('--model', '-m', type=str, default='llama3.1:8b',
                       help='LLM model name (e.g., llama3.1:8b, llama3.1:70b, qwen2.5:7b, qwen2.5:72b)')
    parser.add_argument('--base-url', type=str, default='http://localhost:11434',
                       help='Ollama API base URL')
    parser.add_argument('--temperature', type=float, default=0.3,
                       help='LLM temperature')
    parser.add_argument('--timeout', type=int, default=120,
                       help='Request timeout in seconds')
    
    # 实验参数
    parser.add_argument('--no-calibration', action='store_true',
                       help='Disable iterative calibration')
    parser.add_argument('--cache-dir', type=str, default='./data/cache',
                       help='Cache directory for intermediate results')
    
    args = parser.parse_args()
    
    # 从文件名自动解析venue和year
    venue, year = parse_input_filename(args.data)
    model_short = get_model_short_name(args.model)
    
    # 检查数据文件
    if not Path(args.data).exists():
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)
    
    # 检查Ollama连接
    print(f"\n{'='*60}")
    print("Experiment 1: Simulation Validity Test")
    print(f"{'='*60}\n")
    
    print(f"Detected from filename:")
    print(f"  Venue: {venue}")
    print(f"  Year: {year}")
    print(f"  Model: {args.model} -> {model_short}")
    print(f"  Output will be: {venue}{year}_con_{model_short}.json\n")
    
    print(f"Checking Ollama connection at {args.base_url}...")
    if not test_ollama_connection(args.base_url):
        print("Error: Cannot connect to Ollama. Please ensure Ollama is running.")
        print(f"  Try: ollama serve")
        sys.exit(1)
    print("✓ Ollama is running\n")
    
    # 初始化LLM客户端
    print(f"Initializing LLM client with model: {args.model}")
    llm_client = OllamaClient(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        timeout=args.timeout
    )
    
    # 加载数据
    print(f"\nLoading data from: {args.data}")
    papers_data = load_papers(args.data, limit=args.limit)
    papers = papers_to_paperinfo(papers_data)
    
    # 统计
    real_decisions = get_real_decisions(papers)
    real_accept_count = sum(1 for d in real_decisions.values() if d == 'Accept')
    real_reject_count = len(real_decisions) - real_accept_count
    
    print(f"\nDataset Statistics:")
    print(f"  Total papers: {len(papers)}")
    print(f"  Real Accept: {real_accept_count} ({100*real_accept_count/len(papers):.1f}%)")
    print(f"  Real Reject: {real_reject_count} ({100*real_reject_count/len(papers):.1f}%)")
    
    # 初始化Agent
    print(f"\nInitializing Multi-Agent System...")
    analyzer = MultiDimensionalAnalyzer(llm_client)
    chair_agent = ChairAgent(llm_client, venue=venue, year=year)
    
    # 运行Expert评估
    print(f"\n{'='*60}")
    print("Phase 1: Expert Agent Evaluation")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    expert_results = run_expert_evaluation(
        papers=papers,
        analyzer=analyzer,
        cache_dir=args.cache_dir
    )
    expert_time = time.time() - start_time
    print(f"\nExpert evaluation completed in {expert_time/60:.1f} minutes")
    
    # 运行Chair决策
    print(f"\n{'='*60}")
    print("Phase 2: Chair Agent Decision")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    chair_decisions = run_chair_decisions(
        papers=papers,
        expert_results=expert_results,
        chair_agent=chair_agent,
        real_accept_count=real_accept_count,
        use_calibration=not args.no_calibration
    )
    chair_time = time.time() - start_time
    print(f"\nChair decisions completed in {chair_time/60:.1f} minutes")
    
    # 提取决策字符串
    simulated_decisions = {
        pid: decision.decision for pid, decision in chair_decisions.items()
    }
    
    # 计算评估指标
    print(f"\n{'='*60}")
    print("Phase 3: Evaluation")
    print(f"{'='*60}\n")
    
    metrics = compute_all_metrics(simulated_decisions, real_decisions)
    print(metrics.summary())
    
    # 保存结果
    config_dict = {
        "model": args.model,
        "model_short": model_short,
        "temperature": args.temperature,
        "venue": venue,
        "year": year,
        "limit": args.limit,
        "calibration": not args.no_calibration,
        "data_file": args.data
    }
    
    output_path = save_results(
        output_dir=args.output,
        metrics=metrics,
        decisions=chair_decisions,
        expert_results=expert_results,
        real_decisions=real_decisions,
        config=config_dict,
        venue=venue,
        year=year,
        model_short=model_short
    )
    
    # 总结
    print(f"\n{'='*60}")
    print("Experiment Complete")
    print(f"{'='*60}\n")
    
    print(f"Key Results:")
    print(f"  Accuracy:      {metrics.accuracy:.4f} ({100*metrics.accuracy:.2f}%)")
    print(f"  Cohen's Kappa: {metrics.cohens_kappa:.4f}")
    print(f"  F1 Score:      {metrics.confusion_matrix.f1_score:.4f}")
    
    print(f"\nInterpretation:")
    if metrics.cohens_kappa >= 0.6:
        print("  ✓ Substantial agreement - system effectively simulates human review")
    elif metrics.cohens_kappa >= 0.4:
        print("  ~ Moderate agreement - system captures key patterns")
    else:
        print("  ✗ Fair agreement - consider model improvements")
    
    print(f"\nResults saved to: {output_path}")
    print(f"Total time: {(expert_time + chair_time)/60:.1f} minutes")


if __name__ == '__main__':
    main()