"""Groq OpenAI-compatible chat model integration built on `langchain-core`.

Uses `GROQ_API_KEY`, `GROQ_ENDPOINT` (`https://api.groq.com/openai/v1`), and
`GROQ_MODEL` (`qwen/qwen3.8-27b`) from `app.config.Settings`.
Supports `bind_tools()` with either `StructuredTool` instances, `ToolRegistry`,
or OpenAI function-calling schema dicts.
"""

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field, SecretStr

from app.agent.tools_adapter import (
    LLM_WIRE_DESCRIPTIONS,
    build_openai_tool_schemas,
    compact_input_schema,
)
from app.config import Settings, get_settings
from app.tools.errors import redact
from app.tools.registry import ToolRegistry

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class AgentModelError(Exception):
    """Sanitized exception raised when the LLM endpoint fails or is misconfigured."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(redact(message))
        self.code = code
        self.message = redact(message)
        self.retryable = retryable
        self.status_code = status_code


def strip_reasoning_tags(text: Optional[str]) -> str:
    """Ensure private `<think>...</think>` blocks are never persisted or exposed."""
    if not text:
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.strip()


def langchain_messages_to_openai(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    """Serialize `langchain-core` messages into OpenAI/Groq chat-completion message dicts."""
    payload_messages: List[Dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            payload_messages.append(
                {
                    "role": "system",
                    "content": str(msg.content or ""),
                }
            )
        elif isinstance(msg, HumanMessage):
            payload_messages.append(
                {
                    "role": "user",
                    "content": str(msg.content or ""),
                }
            )
        elif isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                oa_calls = []
                for idx, tc in enumerate(tool_calls):
                    call_id = str(tc.get("id") or f"call_{idx}")
                    name = str(tc.get("name") or "")
                    args = tc.get("args")
                    if isinstance(args, dict) and "_raw_args" in args and len(args) == 1:
                        args_str = str(args["_raw_args"])
                    elif isinstance(args, dict):
                        args_str = json.dumps(args, default=str)
                    elif isinstance(args, str):
                        args_str = args
                    else:
                        args_str = "{}"
                    oa_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": args_str,
                            },
                        }
                    )
                entry: Dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": oa_calls,
                }
                content_str = str(msg.content or "").strip()
                entry["content"] = content_str if content_str else None
                payload_messages.append(entry)
            else:
                payload_messages.append(
                    {
                        "role": "assistant",
                        "content": str(msg.content or ""),
                    }
                )
        elif isinstance(msg, ToolMessage):
            payload_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(msg.tool_call_id),
                    "name": str(getattr(msg, "name", None) or "tool"),
                    "content": str(msg.content or ""),
                }
            )
        else:
            payload_messages.append(
                {
                    "role": "user",
                    "content": str(msg.content or ""),
                }
            )
    return payload_messages


def parse_openai_chat_response(data: Dict[str, Any]) -> AIMessage:
    """Reconstruct a `langchain_core.messages.AIMessage` from an OpenAI/Groq response body."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AgentModelError(
            "invalid_model_response",
            "Model response did not contain any completion choices",
            retryable=True,
        )

    first = choices[0] if isinstance(choices[0], dict) else {}
    msg = first.get("message") if isinstance(first.get("message"), dict) else {}
    raw_content = msg.get("content")
    clean_content = strip_reasoning_tags(raw_content if isinstance(raw_content, str) else "")

    raw_tool_calls = msg.get("tool_calls") or []
    lc_tool_calls: List[Dict[str, Any]] = []
    if isinstance(raw_tool_calls, list):
        for idx, tc in enumerate(raw_tool_calls):
            if not isinstance(tc, dict):
                continue
            call_id = str(tc.get("id") or f"call_{idx}")
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            fn_name = str(fn.get("name") or "")
            raw_args = fn.get("arguments")
            parsed_args: Dict[str, Any]
            if isinstance(raw_args, dict):
                parsed_args = raw_args
            elif isinstance(raw_args, str):
                stripped = raw_args.strip()
                if not stripped:
                    parsed_args = {}
                else:
                    try:
                        loaded = json.loads(stripped)
                        if isinstance(loaded, dict):
                            parsed_args = loaded
                        else:
                            parsed_args = {"_raw_args": raw_args}
                    except Exception:
                        parsed_args = {"_raw_args": raw_args}
            else:
                parsed_args = {}

            lc_tool_calls.append(
                {
                    "id": call_id,
                    "name": fn_name,
                    "args": parsed_args,
                    "type": "tool_call",
                }
            )

    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    response_meta = {
        "model_name": data.get("model"),
        "finish_reason": first.get("finish_reason"),
        "token_usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
    }
    return AIMessage(
        content=clean_content,
        tool_calls=lc_tool_calls,
        response_metadata=response_meta,
    )


def _convert_tool_to_openai_dict(tool: Any) -> Dict[str, Any]:
    """Convert a `StructuredTool`, `BaseTool`, or dict into an OpenAI function tool schema."""
    if isinstance(tool, dict):
        if "type" in tool and "function" in tool:
            return tool
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": compact_input_schema(tool.get("parameters") or tool.get("input_schema") or {}),
            },
        }
    if isinstance(tool, BaseTool):
        raw_schema: Dict[str, Any] = {}
        if tool.args_schema is not None and hasattr(tool.args_schema, "model_json_schema"):
            raw_schema = tool.args_schema.model_json_schema()  # type: ignore[union-attr]
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": LLM_WIRE_DESCRIPTIONS.get(tool.name, tool.description),
                "parameters": compact_input_schema(raw_schema),
            },
        }
    raise TypeError(f"Unsupported tool type for bind_tools: {type(tool).__name__}")


class GroqChatModel(BaseChatModel):
    """OpenAI-compatible `langchain-core` chat model targeting Groq's `/chat/completions` endpoint."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: Optional[SecretStr] = Field(default=None, exclude=True)
    base_url: str = Field(default="https://api.groq.com/openai/v1")
    model_name: str = Field(default="qwen/qwen3.8-27b")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=2048)
    timeout_seconds: float = Field(default=45.0)
    max_http_retries: int = Field(default=2)
    bound_tools: List[Dict[str, Any]] = Field(default_factory=list)
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default="auto")
    http_client: Optional[httpx.Client] = Field(default=None, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "groq-openai-chat"

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        """Return a new `GroqChatModel` instance with the given tools bound for native tool calling."""
        converted = [_convert_tool_to_openai_dict(t) for t in tools]
        return self.model_copy(
            update={
                "bound_tools": converted,
                "tool_choice": tool_choice,
            }
        )

    def _chat_completions_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raw_key = self.api_key.get_secret_value() if self.api_key else ""
        if not raw_key:
            raise AgentModelError(
                "missing_api_key",
                "GROQ_API_KEY is not configured on the backend.",
                retryable=False,
            )

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": langchain_messages_to_openai(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if stop:
            payload["stop"] = stop

        tools = kwargs.get("tools") or self.bound_tools
        if tools:
            payload["tools"] = tools
            choice = kwargs.get("tool_choice", self.tool_choice)
            if choice is not None:
                payload["tool_choice"] = choice

        headers = {
            "Authorization": f"Bearer {raw_key}",
            "Content-Type": "application/json",
        }
        url = self._chat_completions_url()

        attempts = max(1, int(self.max_http_retries) + 1)
        last_exc: Optional[AgentModelError] = None

        for attempt in range(1, attempts + 1):
            try:
                if self.http_client is not None:
                    resp = self.http_client.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout_seconds,
                    )
                else:
                    with httpx.Client(timeout=self.timeout_seconds) as client:
                        resp = client.post(url, headers=headers, json=payload)
            except httpx.TimeoutException:
                last_exc = AgentModelError(
                    "model_timeout",
                    "The language model request timed out.",
                    retryable=True,
                )
                if attempt < attempts:
                    time.sleep(min(2.0 * attempt, 5.0))
                    continue
                raise last_exc
            except httpx.HTTPError as exc:
                last_exc = AgentModelError(
                    "model_network_error",
                    f"Failed to reach language model endpoint: {type(exc).__name__}",
                    retryable=True,
                )
                if attempt < attempts:
                    time.sleep(min(1.5 * attempt, 4.0))
                    continue
                raise last_exc

            if resp.status_code == 200:
                try:
                    body = resp.json()
                except Exception:
                    raise AgentModelError(
                        "invalid_model_response",
                        "Language model returned invalid JSON.",
                        retryable=True,
                        status_code=resp.status_code,
                    )
                ai_msg = parse_openai_chat_response(body)
                return ChatResult(generations=[ChatGeneration(message=ai_msg)])

            # Parse error details safely without leaking secrets
            err_msg = f"Language model request failed with HTTP {resp.status_code}"
            err_code = "model_api_error"
            try:
                err_json = resp.json()
                if isinstance(err_json, dict) and isinstance(err_json.get("error"), dict):
                    err_obj = err_json["error"]
                    if err_obj.get("message"):
                        err_msg = redact(str(err_obj["message"]))[:300]
                    if err_obj.get("code"):
                        err_code = str(err_obj["code"])[:64]
            except Exception:
                pass

            if resp.status_code in (401, 403):
                raise AgentModelError(
                    "model_auth_error",
                    "Language model authentication failed; verify GROQ_API_KEY configuration.",
                    retryable=False,
                    status_code=resp.status_code,
                )

            if resp.status_code == 429 or (resp.status_code == 413 and "rate_limit" in err_code):
                retry_after_hdr = resp.headers.get("retry-after")
                wait_s = 2.0 * attempt
                if retry_after_hdr:
                    try:
                        wait_s = min(float(retry_after_hdr), 12.0)
                    except ValueError:
                        pass
                last_exc = AgentModelError(
                    "model_rate_limited",
                    f"Language model rate limit reached: {err_msg}",
                    retryable=True,
                    status_code=resp.status_code,
                )
                if attempt < attempts and wait_s <= 12.0:
                    time.sleep(wait_s)
                    continue
                raise last_exc

            if resp.status_code >= 500:
                last_exc = AgentModelError(
                    "model_upstream_error",
                    err_msg,
                    retryable=True,
                    status_code=resp.status_code,
                )
                if attempt < attempts:
                    time.sleep(min(2.0 * attempt, 5.0))
                    continue
                raise last_exc

            if resp.status_code == 400 and err_code == "tool_use_failed":
                last_exc = AgentModelError(
                    "tool_use_failed",
                    err_msg,
                    retryable=True,
                    status_code=resp.status_code,
                )
                if attempt < attempts:
                    # Add a transient system reminder so the next attempt uses only registered tools.
                    payload["messages"] = [
                        *payload["messages"],
                        {
                            "role": "system",
                            "content": (
                                f"Your previous tool call failed validation ({err_msg}). "
                                "You must ONLY call tools from the provided tools list using their exact names and parameter types."
                            ),
                        },
                    ]
                    continue
                raise last_exc

            # 4xx non-retryable client error
            raise AgentModelError(
                err_code,
                err_msg,
                retryable=False,
                status_code=resp.status_code,
            )

        raise last_exc or AgentModelError("model_api_error", "Language model call failed", retryable=True)


def create_groq_chat_model(
    settings: Optional[Settings] = None,
    *,
    registry: Optional[ToolRegistry] = None,
    bind_default_tools: bool = True,
    http_client: Optional[httpx.Client] = None,
) -> Runnable[Any, AIMessage]:
    """Create a fresh per-run `GroqChatModel` configured from environment `Settings`."""
    cfg = settings or get_settings()
    model = GroqChatModel(
        api_key=SecretStr(cfg.groq_api_key) if cfg.groq_api_key else None,
        base_url=cfg.groq_endpoint,
        model_name=cfg.groq_model,
        timeout_seconds=float(cfg.agent_model_timeout_seconds),
        max_http_retries=cfg.agent_max_retries,
        http_client=http_client,
    )
    if bind_default_tools:
        tool_schemas = build_openai_tool_schemas(registry)
        return model.bind_tools(tool_schemas)
    return model
