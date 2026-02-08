# Agents module
from .llm_client import OllamaClient, LLMResponse
from .base_agent import BaseExpertAgent, AgentEvaluation, PaperInfo
from .expert_agents import (
    NoveltyAgent, 
    MethodologyAgent, 
    ClarityAgent, 
    EmpiricalAgent,
    MultiDimensionalAnalyzer,
    AggregatedEvaluation
)
from .chair_agent import (
    ChairAgent, 
    ChairDecision,
    iterative_calibration,
    simple_threshold_decisions
)
