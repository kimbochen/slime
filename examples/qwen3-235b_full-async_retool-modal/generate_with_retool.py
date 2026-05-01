# Adapted from https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py
#
# Qwen3-235B-A22B-Thinking-2507 native chat-template + tool-call grammar.
# Qwen3-Thinking emits:
#   <think>...reasoning...</think>\n\n<tool_call>\n{"name":...,"arguments":{...}}\n</tool_call><|im_end|>
# OR ...</think>\n\nAnswer: \\boxed{...}<|im_end|>
#
# Tool responses (per the model's chat_template.jinja) feed back as:
#   <|im_start|>user\n<tool_response>\n{result}\n</tool_response><|im_end|>\n
#   <|im_start|>assistant\n<think>\n
# The trailing `<think>\n` re-opens the assistant turn in the format the model
# was trained on (the chat template's `add_generation_prompt` does the same).
import json
import re
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Reward model: math_dapo (same as GLM example).
try:
    from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
except ImportError as e:
    raise ImportError("MathDapo is not installed") from e

# Modal-backed sandbox pool (replaces examples/retool/tool_sandbox.py).
from modal_tool_sandbox import SEMAPHORE, TOOL_CONFIGS, tool_registry


def format_conversation_with_tools(
    tokenizer,
    prompt: str,
    tools: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
) -> list[int]:
    """Build the initial prompt token ids using the model's NATIVE chat template.

    For Qwen3-235B-A22B-Thinking-2507 the trained format is:
      <|im_start|>system
      ... + tools spec ...
      <|im_end|>
      <|im_start|>user
      {prompt}
      <|im_end|>
      <|im_start|>assistant
      <think>
      ← model continues from here

    HF's `tokenizer.apply_chat_template(..., tools=..., add_generation_prompt=True)`
    handles the tools-spec injection per the model's own chat_template.jinja.
    """
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if prompt:
        messages.append({"role": "user", "content": prompt})

    return tokenizer.apply_chat_template(
        messages,
        tools=tools if tools else None,
        add_generation_prompt=True,
        tokenize=True,
    )


# --- Parsing -----------------------------------------------------------------
# Qwen3 native tool-call grammar (JSON):
#   <tool_call>
#   {"name": "code_interpreter", "arguments": {"code": "..."}}
#   </tool_call>
#
# The JSON value for `arguments.code` may contain raw newlines (the chat
# template does NOT escape them on emission), which is technically invalid
# JSON. We try a strict parse first, then fall back to a best-effort newline-
# escape inside string values.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_ANSWER_RE = re.compile(r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_IM_END_RE = re.compile(r"<\|im_end\|>")


def _parse_tool_call_json(body: str) -> dict | None:
    """Best-effort parse of the JSON body inside <tool_call>...</tool_call>.

    Qwen3 emits multi-line code as raw newlines inside a JSON string, which
    json.loads rejects. Fall back to escaping newlines/tabs/CRs inside string
    values only.
    """
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass

    # Naive but effective: walk the string, only escape control chars when
    # we're inside a string literal (i.e. between quotes that aren't escaped).
    out, in_str, escape = [], False, False
    for ch in body:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\":
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            out.append(ch)
            in_str = not in_str
            continue
        if in_str and ch == "\n":
            out.append("\\n")
        elif in_str and ch == "\r":
            out.append("\\r")
        elif in_str and ch == "\t":
            out.append("\\t")
        else:
            out.append(ch)
    try:
        return json.loads("".join(out))
    except json.JSONDecodeError:
        return None


def postprocess_predictions(prediction: str):
    """Extract (action, content) from a model output chunk for Qwen3-Thinking-2507.

    Scans the text outside of `<think>...</think>` blocks for either:
      * a final answer  `Answer: \\boxed{…}`  → ("answer", "<answer>")
      * a tool call     `<tool_call>{…json…}</tool_call>` → ("code", "<py code>")

    Returns (None, "") if neither matched.
    """
    # Strip <think>...</think> blocks for parsing — final answer / tool_call
    # only ever land in the post-thinking text. Unclosed <think> means the
    # model is still in the reasoning phase.
    visible = _THINK_RE.sub("", prediction)
    if "<think>" in visible and "</think>" not in visible:
        # Still inside an unclosed think block — no decision yet.
        return None, ""

    # Strip the trailing <|im_end|> if SGLang included it.
    visible = _IM_END_RE.sub("", visible)

    m = _ANSWER_RE.search(visible)
    if m:
        return "answer", m.group(1).strip()

    m = _TOOL_CALL_RE.search(visible)
    if m:
        body = m.group(1).strip()
        data = _parse_tool_call_json(body)
        if isinstance(data, dict) and data.get("name") == "code_interpreter":
            args = data.get("arguments", {}) or {}
            if isinstance(args, str):
                # Some models return arguments as a JSON string.
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            code = args.get("code", "") if isinstance(args, dict) else ""
            if isinstance(code, str) and code.strip():
                return "code", code

    return None, ""


def postprocess_responses(resp: str) -> str:
    """Truncate the model output at the latest complete tool-call or final answer.

    SGLang sometimes returns more tokens than form a clean tool call (e.g., the
    model started a second tool call and got cut off mid-block). Snipping at
    the last complete `</tool_call>` or final answer keeps the response well-
    formed for re-feeding.
    """
    if "</tool_call>" in resp:
        idx = resp.rfind("</tool_call>")
        return resp[: idx + len("</tool_call>")]

    if "Answer:" in resp and "\\boxed{" in resp:
        matches = list(_ANSWER_RE.finditer(resp))
        if matches:
            return resp[: matches[-1].end()]

    return resp


# --- Tool-call execution + observation re-opener -----------------------------


def _qwen_observation(content: str) -> str:
    """Wrap a tool result in Qwen3-Thinking-2507's native observation block +
    re-open the assistant turn so the model continues from `<think>`.

    Mirrors what tokenizer.apply_chat_template produces for a `tool` message
    followed by `add_generation_prompt=True`:
      <|im_start|>user
      <tool_response>
      {content}
      </tool_response><|im_end|>
      <|im_start|>assistant
      <think>

    The leading `<|im_end|>\\n` closes the previous assistant turn (in case
    SGLang stopped before emitting one). Doubled `<|im_end|>` is harmless if
    SGLang did emit one — we strip it in `postprocess_responses` is not done
    here, but the re-opener is robust either way at the token level.
    """
    return (
        "<|im_end|>\n"
        f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )


async def execute_predictions(prediction: str) -> tuple[str, bool]:
    """Execute predictions and return (next_obs, done)."""
    action, content = postprocess_predictions(prediction)

    if action == "code":
        code = content.strip()
        if code:
            async with SEMAPHORE:
                result = await tool_registry.execute_tool(
                    "code_interpreter", {"code": code}
                )
        else:
            result = "Error: No Python code found"
        return _qwen_observation(result), False

    if action == "answer":
        return "", True

    # Couldn't parse a tool call or final answer. Hint in Qwen's own format
    # so the model continues coherently rather than rambling further.
    hint = (
        "Error: I could not parse a tool call or final answer.\n"
        "Use the native tool-call format (JSON inside <tool_call> tags):\n"
        "<tool_call>\n"
        '{"name": "code_interpreter", "arguments": {"code": "your python code here"}}\n'
        "</tool_call>\n"
        "Or give the final answer as: Answer: \\boxed{answer}"
    )
    return _qwen_observation(hint), False


# --- Main rollout loop -------------------------------------------------------


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Custom generation function with multi-turn tool-call support (Qwen3)."""
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    tool_specs = tool_registry.get_tool_specs()
    prompt_tokens_ids = format_conversation_with_tools(
        state.tokenizer, sample.prompt, tools=tool_specs
    )
    prompt = state.tokenizer.decode(prompt_tokens_ids)
    response = ""
    response_token_ids: list[int] = []
    loss_masks: list[int] = []
    tool_call_count = 0

    for turn in range(TOOL_CONFIGS["max_turns"]):
        total_length = len(prompt_tokens_ids) + len(response_token_ids)
        if args.rollout_max_context_len is not None:
            max_context_length = args.rollout_max_context_len
        else:
            max_context_length = args.context_parallel_size * args.max_tokens_per_gpu
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break

        current_token_ids = prompt_tokens_ids + response_token_ids
        payload = {
            "input_ids": current_token_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
        }

        # Optional wandb debug — same as GLM example, key markers swapped to
        # Qwen3 (`<|im_start|>system` instead of `<|system|>`, `<tool_response>`
        # is identical).
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "debug/payload_length": len(prompt + response),
                    "debug/available_tools": len(tool_specs),
                    "debug/tools_used": response.count("<tool_response>"),
                    "debug/turn": turn,
                })
        except ImportError:
            pass

        output = await post(url, payload)

        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        if "output_token_logprobs" in output["meta_info"]:
            cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response = state.tokenizer.decode(cur_response_token_ids)
            cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            if sample.rollout_log_probs is None:
                sample.rollout_log_probs = []
            sample.rollout_log_probs += cur_log_probs
        else:
            cur_response = output["text"]
            cur_response = postprocess_responses(cur_response)
            cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)

        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        next_obs, done = await execute_predictions(cur_response)
        if done:
            break

        if "<tool_response>" in next_obs:
            tool_call_count += 1

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)

        if sample.rollout_log_probs is not None:
            sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)
            assert len(response_token_ids) == len(sample.rollout_log_probs), (
                f"Token/logp length mismatch at turn {turn}: "
                f"{len(response_token_ids)} tokens vs {len(sample.rollout_log_probs)} logps"
            )

        if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
            break

    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks

    # Wandb logging hooks (key names match the GLM example so metrics_report.py
    # works unchanged).
    sample.payload_text = prompt + response
    sample.payload_has_system = "<|im_start|>system" in (prompt + response)
    sample.payload_has_tools = "<tools>" in (prompt + response)
    sample.tool_call_count = tool_call_count

    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample, **kwargs):
    """Tool-call reward — math_dapo as primary, tool-use bonus for failures."""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    solution_str = sample.prompt + sample.response
    ground_truth = sample.label if sample.label is not None else ""
    num_turns = getattr(sample, "tool_call_count", 0)

    result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)

    if result["score"] < 0:
        # Encourage the model to call tools when it can't solve unaided.
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    return result
