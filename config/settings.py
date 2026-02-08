"""
Global configuration settings for Multi-Agent Peer Review Analysis.

Usage:
    from config.settings import Config
    config = Config()
    config.llm_model = "llama3.1:70b"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: str = "ollama"  # ollama, openai, etc.
    model: str = "llama3.1:8b"  # llama3.1:8b, llama3.1:70b, qwen2.5:7b, qwen2.5:72b
    base_url: str = "http://localhost:11434"
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout: int = 120
    retry_attempts: int = 3
    retry_delay: float = 2.0


@dataclass
class ExperimentConfig:
    """实验配置"""
    # 数据路径
    data_dir: str = "./data/raw/openreview"
    output_dir: str = "./data/results"
    cache_dir: str = "./data/cache"
    
    # 实验参数
    venue: str = "iclr"
    year: int = 2018
    limit: Optional[int] = None  # None表示处理全部
    
    # 评估阈值
    accept_threshold: float = 5.5  # 简单平均的接受阈值
    high_impact_percentile: float = 30  # Top 30% 为高影响力
    
    # 随机种子
    random_seed: int = 42


@dataclass
class AgentConfig:
    """Agent配置"""
    # Expert Agent维度权重（用于M2 Confidence Weighted）
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        'novelty': 0.30,
        'methodology': 0.30,
        'clarity': 0.15,
        'empirical': 0.25
    })
    
    # 迭代校准参数
    max_calibration_iterations: int = 10
    calibration_batch_extra: int = 2  # 每轮多选几篇重评
    
    # 讨论轮数（M4 Iterative Consensus）
    max_discussion_rounds: int = 3
    convergence_threshold: float = 0.5


@dataclass
class Config:
    """主配置类"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    
    def __post_init__(self):
        # 确保目录存在
        Path(self.experiment.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.experiment.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.experiment.cache_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_args(cls, args) -> 'Config':
        """从命令行参数创建配置"""
        config = cls()
        
        if hasattr(args, 'model') and args.model:
            config.llm.model = args.model
        if hasattr(args, 'base_url') and args.base_url:
            config.llm.base_url = args.base_url
        if hasattr(args, 'temperature') and args.temperature:
            config.llm.temperature = args.temperature
        if hasattr(args, 'data_dir') and args.data_dir:
            config.experiment.data_dir = args.data_dir
        if hasattr(args, 'output_dir') and args.output_dir:
            config.experiment.output_dir = args.output_dir
        if hasattr(args, 'year') and args.year:
            config.experiment.year = args.year
        if hasattr(args, 'limit') and args.limit:
            config.experiment.limit = args.limit
            
        return config


# 预设配置
PRESETS = {
    'llama3.1-8b': LLMConfig(model="llama3.1:8b"),
    'llama3.1-70b': LLMConfig(model="llama3.1:70b", timeout=300),
    'qwen2.5-7b': LLMConfig(model="qwen2.5:7b"),
    'qwen2.5-72b': LLMConfig(model="qwen2.5:72b", timeout=300),
}


def get_preset_config(preset_name: str) -> Config:
    """获取预设配置"""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    
    config = Config()
    config.llm = PRESETS[preset_name]
    return config
