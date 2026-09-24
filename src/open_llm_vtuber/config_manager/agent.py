"""
This module contains the pydantic model for the configurations of
different types of agents.
"""

from pydantic import BaseModel, Field
from typing import Dict, ClassVar, Optional, Literal, List
from .i18n import I18nMixin, Description
from .stateless_llm import StatelessLLMConfigs

# ======== Configurations for different Agents ========


class BasicMemoryAgentConfig(I18nMixin, BaseModel):
    """Configuration for the basic memory agent."""

    llm_provider: Literal[
        "stateless_llm_with_template",
        "openai_compatible_llm",
        "claude_llm",
        "llama_cpp_llm",
        "ollama_llm",
        "lmstudio_llm",
        "openai_llm",
        "gemini_llm",
        "zhipu_llm",
        "deepseek_llm",
        "groq_llm",
        "mistral_llm",
    ] = Field(..., alias="llm_provider")

    faster_first_response: Optional[bool] = Field(True, alias="faster_first_response")
    segment_method: Literal["regex", "pysbd"] = Field("pysbd", alias="segment_method")
    use_mcpp: Optional[bool] = Field(False, alias="use_mcpp")
    mcp_enabled_servers: Optional[List[str]] = Field([], alias="mcp_enabled_servers")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "llm_provider": Description(
            en="LLM provider to use for this agent",
            zh="Basic Memory Agent 智能体使用的大语言模型选项",
        ),
        "faster_first_response": Description(
            en="Whether to respond as soon as encountering a comma in the first sentence to reduce latency (default: True)",
            zh="是否在第一句回应时遇上逗号就直接生成音频以减少首句延迟（默认：True）",
        ),
        "segment_method": Description(
            en="Method for segmenting sentences: 'regex' or 'pysbd' (default: 'pysbd')",
            zh="分割句子的方法：'regex' 或 'pysbd'（默认：'pysbd'）",
        ),
        "use_mcpp": Description(
            en="Whether to use MCP (Model Context Protocol) for the agent (default: True)",
            zh="是否使用为智能体启用 MCP (Model Context Protocol) Plus（默认：False）",
        ),
        "mcp_enabled_servers": Description(
            en="List of MCP servers to enable for the agent",
            zh="为智能体启用 MCP 服务器列表",
        ),
    }


class Mem0VectorStoreConfig(I18nMixin, BaseModel):
    """Configuration for Mem0 vector store."""

    provider: str = Field(..., alias="provider")
    config: Dict = Field(..., alias="config")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "provider": Description(
            en="Vector store provider (e.g., qdrant)", zh="向量存储提供者（如 qdrant）"
        ),
        "config": Description(
            en="Provider-specific configuration", zh="提供者特定配置"
        ),
    }


class Mem0LLMConfig(I18nMixin, BaseModel):
    """Configuration for Mem0 LLM."""

    provider: str = Field(..., alias="provider")
    config: Dict = Field(..., alias="config")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "provider": Description(en="LLM provider name", zh="语言模型提供者名称"),
        "config": Description(
            en="Provider-specific configuration", zh="提供者特定配置"
        ),
    }


class Mem0EmbedderConfig(I18nMixin, BaseModel):
    """Configuration for Mem0 embedder."""

    provider: str = Field(..., alias="provider")
    config: Dict = Field(..., alias="config")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "provider": Description(en="Embedder provider name", zh="嵌入模型提供者名称"),
        "config": Description(
            en="Provider-specific configuration", zh="提供者特定配置"
        ),
    }


class Mem0Config(I18nMixin, BaseModel):
    """Configuration for Mem0."""

    vector_store: Mem0VectorStoreConfig = Field(..., alias="vector_store")
    llm: Mem0LLMConfig = Field(..., alias="llm")
    embedder: Mem0EmbedderConfig = Field(..., alias="embedder")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "vector_store": Description(en="Vector store configuration", zh="向量存储配置"),
        "llm": Description(en="LLM configuration", zh="语言模型配置"),
        "embedder": Description(en="Embedder configuration", zh="嵌入模型配置"),
    }


# =================================


class HumeAIConfig(I18nMixin, BaseModel):
    """Configuration for the Hume AI agent."""

    api_key: str = Field(..., alias="api_key")
    host: str = Field("api.hume.ai", alias="host")
    config_id: Optional[str] = Field(None, alias="config_id")
    idle_timeout: int = Field(15, alias="idle_timeout")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "api_key": Description(
            en="API key for Hume AI service", zh="Hume AI 服务的 API 密钥"
        ),
        "host": Description(
            en="Host URL for Hume AI service (default: api.hume.ai)",
            zh="Hume AI 服务的主机地址（默认：api.hume.ai）",
        ),
        "config_id": Description(
            en="Configuration ID for EVI settings", zh="EVI 配置 ID"
        ),
        "idle_timeout": Description(
            en="Idle timeout in seconds before disconnecting (default: 15)",
            zh="空闲超时断开连接的秒数（默认：15）",
        ),
    }


# =================================


class LettaConfig(I18nMixin, BaseModel):
    """Configuration for the Letta agent."""

    host: str = Field("localhost", alias="host")
    port: int = Field(8283, alias="port")
    id: str = Field(..., alias="id")
    faster_first_response: Optional[bool] = Field(True, alias="faster_first_response")
    segment_method: Literal["regex", "pysbd"] = Field("pysbd", alias="segment_method")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "host": Description(
            en="Host address for the Letta server", zh="Letta服务器的主机地址"
        ),
        "port": Description(
            en="Port number for the Letta server (default: 8283)",
            zh="Letta服务器的端口号（默认：8283）",
        ),
        "id": Description(
            en="Agent instance ID running on the Letta server",
            zh="指定Letta服务器上运行的Agent实例id",
        ),
    }


# =================================


class RealtimeAgentConfig(I18nMixin, BaseModel):
    """Configuration for the real-time agent (fast talker + Jev routing + Hermes tasks)."""

    openrouter_api_key: str = Field("", alias="openrouter_api_key")
    openrouter_base_url: str = Field(
        "https://openrouter.ai/api/v1", alias="openrouter_base_url"
    )
    talker_model: str = Field("deepseek/deepseek-v4.1-flash", alias="talker_model")
    talker_provider: str = Field("", alias="talker_provider")
    talker_temperature: float = Field(0.8, alias="talker_temperature")
    talker_max_tokens: int = Field(200, alias="talker_max_tokens")
    hedge_after_s: float = Field(1.5, alias="hedge_after_s")
    jev_model: str = Field("typesafe/jev-1.13", alias="jev_model")
    jev_url: str = Field("https://openrouter.ai/api/alpha/decisions", alias="jev_url")
    jev_timeout_s: float = Field(1.5, alias="jev_timeout_s")
    jev_fallback_after_s: float = Field(0.6, alias="jev_fallback_after_s")
    unsure_low: float = Field(0.35, alias="unsure_low")
    unsure_high: float = Field(0.65, alias="unsure_high")
    fallback_router_model: str = Field(
        "meta-llama/llama-3.3-70b-instruct", alias="fallback_router_model"
    )
    fallback_router_provider: str = Field("groq", alias="fallback_router_provider")
    hermes_base_url: str = Field("http://localhost:8642", alias="hermes_base_url")
    hermes_api_key: str = Field("", alias="hermes_api_key")
    task_timeout_s: float = Field(600, alias="task_timeout_s")
    max_active_tasks: int = Field(3, alias="max_active_tasks")
    soul_path: str = Field("", alias="soul_path")
    user_profile_path: str = Field("", alias="user_profile_path")
    max_turns: int = Field(8, alias="max_turns")
    quiet_gap_s: float = Field(1.0, alias="quiet_gap_s")
    prerender_acks: bool = Field(True, alias="prerender_acks")
    faster_first_response: bool = Field(True, alias="faster_first_response")
    segment_method: Literal["regex", "pysbd"] = Field("pysbd", alias="segment_method")
    openers: List[str] = Field(default_factory=list, alias="openers")
    opener_carrier: str = Field("I suppose that makes sense.", alias="opener_carrier")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "openers": Description(
            en="Interjections the talker starts replies with; each plays from a pre-rendered clip ([] disables)",
            zh="对话模型用于开头的语气词；每个都会预先生成音频并立即播放（[] 表示关闭）",
        ),
        "opener_carrier": Description(
            en="Sentence each opener is rendered in before being cut out, for natural pitch",
            zh="渲染开头语气词时使用的载体句子，截取后音调更自然",
        ),
        "openrouter_api_key": Description(
            en="OpenRouter API key for the talker, Jev and the fallback router (falls back to OPENROUTER_API_KEY)",
            zh="用于对话模型、Jev 和备用路由的 OpenRouter API 密钥（为空时读取 OPENROUTER_API_KEY）",
        ),
        "talker_model": Description(
            en="Fast model that always answers", zh="始终负责回答的快速模型"
        ),
        "talker_provider": Description(
            en="Pin one OpenRouter provider for the talker ('' = fastest)",
            zh="为对话模型固定一个 OpenRouter 提供商（留空 = 最快）",
        ),
        "hedge_after_s": Description(
            en="Send a backup request if no token arrives by then (0 disables)",
            zh="超过该秒数仍无输出时发送备用请求（0 表示关闭）",
        ),
        "jev_model": Description(
            en="Decision model used to route each turn", zh="用于每轮路由判断的决策模型"
        ),
        "jev_fallback_after_s": Description(
            en="Also ask the fallback router if Jev hasn't answered by then",
            zh="超过该秒数 Jev 仍未回答时，同时询问备用路由",
        ),
        "unsure_low": Description(
            en="Lower task probability where Yuna asks before starting a task",
            zh="任务概率下限，处于区间内时先询问用户",
        ),
        "unsure_high": Description(
            en="Upper task probability where Yuna asks before starting a task",
            zh="任务概率上限，处于区间内时先询问用户",
        ),
        "fallback_router_model": Description(
            en="Classifier used when Jev fails ('' disables)",
            zh="Jev 失败时使用的分类模型（留空表示关闭）",
        ),
        "hermes_base_url": Description(
            en="hermes-agent API server address", zh="hermes-agent API 服务器地址"
        ),
        "hermes_api_key": Description(
            en="hermes-agent API_SERVER_KEY (falls back to HERMES_API_KEY)",
            zh="hermes-agent 的 API_SERVER_KEY（为空时读取 HERMES_API_KEY）",
        ),
        "soul_path": Description(
            en="Persona file prepended to the system prompt (e.g. Hermes SOUL.md)",
            zh="添加到系统提示词开头的人设文件（例如 Hermes 的 SOUL.md）",
        ),
        "user_profile_path": Description(
            en="File of facts about the user (e.g. Hermes memories/USER.md)",
            zh="关于用户的事实文件（例如 Hermes 的 memories/USER.md）",
        ),
        "max_turns": Description(
            en="Conversation exchanges sent to the talker",
            zh="发送给对话模型的对话轮数",
        ),
        "quiet_gap_s": Description(
            en="Silence before announcing finished tasks",
            zh="播报已完成任务前需要的安静时间",
        ),
        "prerender_acks": Description(
            en="Prepare 'on it' acknowledgements as audio ahead of time",
            zh="提前生成“收到”确认语音",
        ),
    }


class AgentSettings(I18nMixin, BaseModel):
    """Settings for different types of agents."""

    basic_memory_agent: Optional[BasicMemoryAgentConfig] = Field(
        None, alias="basic_memory_agent"
    )
    mem0_agent: Optional[Mem0Config] = Field(None, alias="mem0_agent")
    hume_ai_agent: Optional[HumeAIConfig] = Field(None, alias="hume_ai_agent")
    letta_agent: Optional[LettaConfig] = Field(None, alias="letta_agent")
    realtime_agent: Optional[RealtimeAgentConfig] = Field(None, alias="realtime_agent")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "basic_memory_agent": Description(
            en="Configuration for basic memory agent", zh="基础记忆代理配置"
        ),
        "mem0_agent": Description(en="Configuration for Mem0 agent", zh="Mem0代理配置"),
        "hume_ai_agent": Description(
            en="Configuration for Hume AI agent", zh="Hume AI 代理配置"
        ),
        "letta_agent": Description(
            en="Configuration for Letta agent", zh="Letta 代理配置"
        ),
        "realtime_agent": Description(
            en="Configuration for the real-time agent", zh="实时代理配置"
        ),
    }


class AgentConfig(I18nMixin, BaseModel):
    """This class contains all of the configurations related to agent."""

    conversation_agent_choice: Literal[
        "basic_memory_agent",
        "mem0_agent",
        "hume_ai_agent",
        "letta_agent",
        "realtime_agent",
    ] = Field(..., alias="conversation_agent_choice")
    agent_settings: AgentSettings = Field(..., alias="agent_settings")
    llm_configs: StatelessLLMConfigs = Field(..., alias="llm_configs")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "conversation_agent_choice": Description(
            en="Type of conversation agent to use", zh="要使用的对话代理类型"
        ),
        "agent_settings": Description(
            en="Settings for different agent types", zh="不同代理类型的设置"
        ),
        "llm_configs": Description(
            en="Pool of LLM provider configurations", zh="语言模型提供者配置池"
        ),
        "faster_first_response": Description(
            en="Whether to respond as soon as encountering a comma in the first sentence to reduce latency (default: True)",
            zh="是否在第一句回应时遇上逗号就直接生成音频以减少首句延迟（默认：True）",
        ),
        "segment_method": Description(
            en="Method for segmenting sentences: 'regex' or 'pysbd' (default: 'pysbd')",
            zh="分割句子的方法：'regex' 或 'pysbd'（默认：'pysbd'）",
        ),
    }
