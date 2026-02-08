"""
LLM Client for Ollama API.

Supports:
- llama3.1 (8b, 70b)
- qwen2.5 (7b, 72b)

Usage:
    from agents.llm_client import OllamaClient
    client = OllamaClient(model="llama3.1:8b")
    response = client.generate("Hello, world!")
"""

import json
import time
import re
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    import requests
except ImportError:
    raise ImportError("requests is required. Install with: pip install requests")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM响应结构"""
    content: str
    model: str
    tokens_used: int = 0
    latency: float = 0.0
    raw_response: Dict = None


class OllamaClient:
    """Ollama API Client with retry logic and structured output support."""
    
    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        timeout: int = 120,
        retry_attempts: int = 3,
        retry_delay: float = 2.0
    ):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # 验证连接
        self._verify_connection()
    
    def _verify_connection(self):
        """验证Ollama服务是否可用"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                logger.info(f"Connected to Ollama. Available models: {model_names[:5]}...")
                
                # 检查模型是否存在
                if not any(self.model in name for name in model_names):
                    logger.warning(f"Model {self.model} not found locally. It will be pulled on first use.")
            else:
                logger.warning(f"Ollama API returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not verify Ollama connection: {e}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> LLMResponse:
        """
        生成文本响应
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示（可选）
            temperature: 温度参数
            max_tokens: 最大token数
            json_mode: 是否期望JSON输出
        
        Returns:
            LLMResponse对象
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        # JSON模式提示
        if json_mode:
            if system_prompt:
                payload["system"] += "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation, just JSON."
            else:
                payload["system"] = "IMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation, just JSON."
        
        # 重试逻辑
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                start_time = time.time()
                response = requests.post(url, json=payload, timeout=self.timeout)
                latency = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    return LLMResponse(
                        content=result.get('response', ''),
                        model=self.model,
                        tokens_used=result.get('eval_count', 0),
                        latency=latency,
                        raw_response=result
                    )
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    logger.warning(f"Attempt {attempt + 1} failed: {last_error}")
                    
            except requests.Timeout:
                last_error = "Request timeout"
                logger.warning(f"Attempt {attempt + 1} timed out after {self.timeout}s")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt < self.retry_attempts - 1:
                time.sleep(self.retry_delay * (attempt + 1))
        
        raise RuntimeError(f"Failed after {self.retry_attempts} attempts. Last error: {last_error}")
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        生成JSON响应并解析
        
        Returns:
            解析后的JSON字典
        """
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            json_mode=True
        )
        
        return self._parse_json_response(response.content)
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """
        解析JSON响应，处理常见的格式问题
        
        增强功能：
        1. 尝试直接解析完整JSON
        2. 移除markdown代码块后解析
        3. 尝试修复被截断的JSON（补全括号）
        4. 使用正则表达式提取已生成的字段
        """
        # 清理内容
        content = content.strip()
        
        # 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # 移除markdown代码块
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'`(.*?)`'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue
        
        # 尝试找到JSON对象边界
        brace_match = re.search(r'\{.*\}', content, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass
        
        # 尝试修复被截断的JSON
        repaired = self._try_repair_truncated_json(content)
        if repaired:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        
        # 最后尝试：用正则提取已有字段
        extracted = self._extract_fields_from_partial_json(content)
        if extracted.get("score") != 5.0 or extracted.get("confidence") != 0.3:
            # 成功提取到了一些有效值
            logger.warning(f"Extracted partial JSON: score={extracted.get('score')}, confidence={extracted.get('confidence')}")
            return extracted
        
        # 如果所有尝试都失败，返回带有原始内容的错误字典
        logger.warning(f"Failed to parse JSON from response: {content[:200]}...")
        return {
            "parse_error": True,
            "raw_content": content,
            "score": 5.0,  # 默认中等分数
            "confidence": 0.3,
            "sentiment": "mixed",
            "key_findings": ["Unable to parse response"],
            "reasoning": content[:500]
        }
    
    def _try_repair_truncated_json(self, content: str) -> Optional[str]:
        """
        尝试修复被截断的JSON
        
        常见情况：
        - 缺少结尾的 ]} 或 }
        - 字符串中间被截断
        """
        # 找到JSON开始位置
        start_idx = content.find('{')
        if start_idx == -1:
            return None
        
        json_part = content[start_idx:]
        
        # 计算括号平衡
        brace_count = 0
        bracket_count = 0
        in_string = False
        escape_next = False
        last_valid_pos = 0
        
        for i, char in enumerate(json_part):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count >= 0:
                        last_valid_pos = i
                elif char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count >= 0:
                        last_valid_pos = i
        
        # 如果不平衡，尝试修复
        if brace_count > 0 or bracket_count > 0:
            # 截断到最后一个完整的键值对位置
            # 找到最后一个逗号或冒号后的值结束位置
            repaired = json_part
            
            # 如果在字符串中间被截断，先闭合字符串
            if in_string:
                repaired += '"'
            
            # 补全缺失的括号
            repaired += ']' * bracket_count
            repaired += '}' * brace_count
            
            return repaired
        
        return None
    
    def _extract_fields_from_partial_json(self, content: str) -> Dict[str, Any]:
        """
        使用正则表达式从不完整的JSON中提取字段
        
        即使JSON格式损坏，也尝试提取已生成的有效字段
        """
        result = {
            "parse_error": True,
            "partial_extraction": True,
            "score": 5.0,
            "confidence": 0.3,
            "sentiment": "mixed",
            "key_findings": [],
            "reasoning": ""
        }
        
        # 提取 score（支持整数和小数）
        score_match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', content)
        if score_match:
            try:
                score = float(score_match.group(1))
                result["score"] = max(1.0, min(10.0, score))
            except ValueError:
                pass
        
        # 提取 confidence
        conf_match = re.search(r'"confidence"\s*:\s*(\d+(?:\.\d+)?)', content)
        if conf_match:
            try:
                conf = float(conf_match.group(1))
                result["confidence"] = max(0.0, min(1.0, conf))
            except ValueError:
                pass
        
        # 提取 sentiment
        sent_match = re.search(r'"sentiment"\s*:\s*"(positive|negative|mixed)"', content, re.IGNORECASE)
        if sent_match:
            result["sentiment"] = sent_match.group(1).lower()
        
        # 提取 reasoning（可能被截断）
        reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*(?:\\.[^"]*)*)', content)
        if reason_match:
            reasoning = reason_match.group(1)
            # 处理转义字符
            reasoning = reasoning.replace('\\"', '"').replace('\\n', '\n')
            result["reasoning"] = reasoning[:500]  # 截断过长的内容
        
        # 提取 key_findings 数组中的元素
        findings_match = re.search(r'"key_findings"\s*:\s*\[(.*?)(?:\]|$)', content, re.DOTALL)
        if findings_match:
            findings_content = findings_match.group(1)
            # 提取数组中的字符串元素
            items = re.findall(r'"([^"]*(?:\\.[^"]*)*)"', findings_content)
            if items:
                result["key_findings"] = [item.replace('\\"', '"')[:200] for item in items[:5]]
        
        # 提取 decision（用于Chair Agent）
        decision_match = re.search(r'"decision"\s*:\s*"(Accept|Reject)"', content, re.IGNORECASE)
        if decision_match:
            result["decision"] = decision_match.group(1).capitalize()
        
        return result
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        多轮对话接口
        
        Args:
            messages: 消息列表，格式: [{"role": "user/assistant/system", "content": "..."}]
        """
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            }
        }
        
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                start_time = time.time()
                response = requests.post(url, json=payload, timeout=self.timeout)
                latency = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    message = result.get('message', {})
                    return LLMResponse(
                        content=message.get('content', ''),
                        model=self.model,
                        tokens_used=result.get('eval_count', 0),
                        latency=latency,
                        raw_response=result
                    )
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    
            except requests.Timeout:
                last_error = "Request timeout"
            except Exception as e:
                last_error = str(e)
            
            if attempt < self.retry_attempts - 1:
                time.sleep(self.retry_delay * (attempt + 1))
        
        raise RuntimeError(f"Chat failed after {self.retry_attempts} attempts. Last error: {last_error}")


def test_ollama_connection(base_url: str = "http://localhost:11434") -> bool:
    """测试Ollama连接"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


if __name__ == "__main__":
    # 测试代码
    print("Testing Ollama connection...")
    if test_ollama_connection():
        print("✓ Ollama is running")
        
        client = OllamaClient(model="llama3.1:8b")
        
        # 测试简单生成
        response = client.generate("Say 'Hello' in JSON format: {\"greeting\": \"...\"}")
        print(f"Response: {response.content}")
        
        # 测试JSON生成
        json_response = client.generate_json("What is 2+2? Respond as: {\"answer\": <number>}")
        print(f"JSON Response: {json_response}")
    else:
        print("✗ Ollama is not running. Please start Ollama first.")