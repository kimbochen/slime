# Adapted from https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py
import re
from typing import Any

# jinja2 was used by the legacy hand-rolled template (see commented-out
# block below); no longer needed — apply_chat_template handles formatting.
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Import reward models
try:
    from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
except ImportError as e:
    raise ImportError("MathDapo is not installed") from e

# Import tool sandbox functionality (Modal-backed variant)
from modal_tool_sandbox import SEMAPHORE, TOOL_CONFIGS, tool_registry

# ---------------------------------------------------------------------------
# LEGACY: Qwen-style hand-rolled Jinja template + formatter.
# Kept for reference; replaced for GLM-4.5-Air by `format_conversation_with_tools`
# below, which delegates to `tokenizer.apply_chat_template` so we get the
# model's native trained format ([gMASK]<sop><|system|>…<|user|>…<|assistant|>
# \n<think></think>\n) and native tool-call grammar (<tool_call>name\n<arg_key>…
# </arg_key>\n<arg_value>…</arg_value>\n</tool_call>) instead of Qwen JSON.
# ---------------------------------------------------------------------------
# TOOL_TEMPLATE = """<|im_start|>system
# {%- if messages[0]['role'] == 'system' %}
# {{- messages[0]['content'] }}
# {%- else %}
# You are a helpful assistant.
# {%- endif %}
# {%- if tools %}
# # Tools
#
# You may call one or more functions to assist with the user query.
#
# You are provided with function signatures within <tools></tools> XML tags:
# <tools>
# {%- for tool in tools %}
# {{- tool | tojson }}
# {%- endfor %}
# </tools>
#
# For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
# <tool_call>
# {"name": <function-name>, "arguments": <args-json-object>}
# </tool_call>
# {%- endif %}
# <|im_end|>
# {%- for message in messages %}
# {%- if message['role'] == 'user' %}
# <|im_start|>user
# {{- message['content'] }}<|im_end|>
# {%- elif message['role'] == 'assistant' %}
# <|im_start|>assistant
# {{- message['content'] }}<|im_end|>
# {%- endif %}
# {%- endfor %}
# <|im_start|>assistant
# """
#
#
# def format_conversation_with_tools_legacy(
#     prompt: str, tools: list[dict[str, Any]] = None, system_prompt: str = None, messages: list[dict[str, Any]] = None
# ) -> str:
#     """Format conversation using Jinja2 template with tool support"""
#     template = Template(TOOL_TEMPLATE)
#
#     # Prepare messages
#     messages_to_render = []
#
#     # Always add system message - use provided one or default
#     if system_prompt:
#         system_content = system_prompt
#     else:
#         system_content = (
#             "You are a helpful assistant that can use Python "
#             "tools to solve mathematical problems. When you need "
#             "to perform calculations, use the code_interpreter "
#             "tool to execute code and get results."
#         )
#
#     messages_to_render.append({"role": "system", "content": system_content})
#
#     # Add user message if provided
#     if prompt:
#         messages_to_render.append({"role": "user", "content": prompt})
#
#     # Add assistant responses from previous turns if provided
#     if messages:
#         messages_to_render.extend(messages)
#
#     # Render template
#     formatted_text = template.render(messages=messages_to_render, tools=tools or [])
#
#     return formatted_text


def format_conversation_with_tools(
    tokenizer,
    prompt: str,
    tools: list[dict[str, Any]] = None,
    system_prompt: str | None = None,
) -> list[int]:
    """Build the initial prompt token ids using the model's NATIVE chat template.

    For GLM-4.5-Air the trained format is:
      [gMASK]<sop><|system|>...<|user|>{prompt}<|assistant|>\\n<think></think>\\n

    HF's `tokenizer.apply_chat_template(..., tools=..., add_generation_prompt=True)`
    handles all of this — including injecting the tools spec into <|system|> per
    the model's own chat_template.jinja — so we don't hand-roll any of it.
    Returns token ids directly (the caller previously had to tokenize the string).
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


# ---------------------------------------------------------------------------
# LEGACY postprocess_predictions / postprocess_responses (Qwen-JSON / <code>).
# Replaced below for GLM-4.5-Air's native <tool_call>name\n<arg_key>...
# </arg_key>\n<arg_value>...</arg_value>\n</tool_call> grammar.
# ---------------------------------------------------------------------------
# def postprocess_predictions_legacy(prediction: str):
#     """Extract action and content from prediction string"""
#     # Check for Answer: \boxed{...} format (only format we need for math_dapo)
#     answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
#     answer_match = re.search(answer_pattern, prediction, re.DOTALL)
#     if answer_match:
#         content = answer_match.group(1).strip()
#         return "answer", content
#
#     # Then check for <tool_call> tags (Qwen JSON format from old Jinja2 template)
#     tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
#     tool_call_match = re.search(tool_call_pattern, prediction, re.DOTALL)
#     if tool_call_match:
#         try:
#             import json
#             json_str = tool_call_match.group(1).replace("\n", "\\n")
#             tool_call_data = json.loads(json_str)
#             tool_name = tool_call_data.get("name")
#             arguments = tool_call_data.get("arguments", {})
#             if tool_name == "code_interpreter":
#                 code = arguments.get("code", "")
#                 if code.strip():
#                     return "code", code
#         except (json.JSONDecodeError, KeyError, AttributeError):
#             pass
#
#     code_pattern = r"<code>(.*?)</code>"
#     code_match = re.search(code_pattern, prediction, re.DOTALL)
#     if code_match:
#         return "code", code_match.group(1).strip()
#
#     python_code_pattern = r"```python\s*(.*?)\s*```"
#     python_code_match = re.search(python_code_pattern, prediction, re.DOTALL)
#     if python_code_match:
#         return "code", python_code_match.group(1).strip()
#     return None, ""
#
#
# def postprocess_responses_legacy(resp: str) -> str:
#     """Post-process response to ensure tag completeness (Qwen JSON variant)."""
#     if "<tool_call>" in resp:
#         tool_call_pattern = r"<tool_call>\s*\{.*?\}\s*</tool_call>"
#         matches = list(re.finditer(tool_call_pattern, resp, re.DOTALL))
#         if matches:
#             return resp[: matches[-1].end()]
#     if "</code>" in resp:
#         return resp.split("</code>")[0] + "</code>"
#     if "```python" in resp:
#         python_pattern = r"```python\s*.*?```"
#         matches = list(re.finditer(python_pattern, resp, re.DOTALL))
#         if matches:
#             return resp[: matches[-1].end()]
#     if "Answer:" in resp and "\\boxed{" in resp:
#         answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
#         matches = list(re.finditer(answer_pattern, resp, re.DOTALL))
#         if matches:
#             return resp[: matches[-1].end()]
#     return resp


# Native GLM-4.5-Air tool-call grammar:
#   <tool_call>{function-name}\n
#   <arg_key>{key1}</arg_key>\n
#   <arg_value>{value1}</arg_value>\n
#   ...
#   </tool_call>
#
# NB: the function name sits on the line immediately after `<tool_call>` (no
# JSON braces). Args come as alternating <arg_key>/<arg_value> pairs.
_GLM_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*([^\n<]+?)\s*\n(.*?)</tool_call>", re.DOTALL
)
_GLM_ARG_PAIR_RE = re.compile(
    r"<arg_key>\s*([^<]+?)\s*</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
)
_ANSWER_RE = re.compile(
    r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL
)


def postprocess_predictions(prediction: str):
    """Extract (action, content) from a model output chunk for GLM-4.5-Air.

    Returns:
      ("answer", "<final answer>") when the model produced `Answer: \\boxed{…}`.
      ("code",   "<python code>")  when the model produced a native GLM tool call
                                   for the code_interpreter.
      (None, "")                   otherwise (model rambling / partial output).
    """
    m = _ANSWER_RE.search(prediction)
    if m:
        return "answer", m.group(1).strip()

    m = _GLM_TOOL_CALL_RE.search(prediction)
    if m:
        name = m.group(1).strip()
        body = m.group(2)
        args = {k.strip(): v for k, v in _GLM_ARG_PAIR_RE.findall(body)}
        if name == "code_interpreter" and args.get("code", "").strip():
            return "code", args["code"]

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


# ---------------------------------------------------------------------------
# LEGACY execute_predictions — emitted observations as
# `\n\n<interpreter>{result}</interpreter>\n\n`, which is the Qwen retool style.
# Replaced for GLM-4.5-Air below to use the model's NATIVE observation block:
#   \n<|observation|>\n<tool_response>\n{result}\n</tool_response>\n
#   <|assistant|>\n<think></think>\n
# The trailing `<|assistant|>\n<think></think>\n` re-opens the assistant turn
# so the model resumes generation in the format it was trained on.
# ---------------------------------------------------------------------------
# async def execute_predictions_legacy(prediction: str) -> str:
#     """Execute predictions and return results"""
#     action, content = postprocess_predictions(prediction)
#     if action == "code":
#         code = content.strip()
#         if code:
#             async with SEMAPHORE:
#                 result = await tool_registry.execute_tool("code_interpreter", {"code": code})
#             next_obs = f"\n\n<interpreter>\n{result}\n</interpreter>\n\n"
#             done = False
#         else:
#             next_obs = "\n\n<interpreter>\nError: No Python code found\n</interpreter>\n\n"
#             done = False
#     elif action == "answer":
#         next_obs = ""
#         done = True
#     else:
#         next_obs = (
#             "\nMy previous action is invalid. "
#             "If I want to execute code, I should put the code between "
#             "<code> and </code>. "
#             "If I want to give the final answer, I should use the format "
#             "'Answer: \\boxed{answer}'. Let me try again.\n"
#         )
#         done = False
#     return next_obs, done


def _glm_observation(content: str) -> str:
    """Wrap a tool result in GLM-4.5-Air's native observation block + reopen
    the assistant turn so the model continues from `<think></think>`."""
    return (
        f"\n<|observation|>\n<tool_response>\n{content}\n</tool_response>\n"
        f"<|assistant|>\n<think></think>\n"
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
        return _glm_observation(result), False

    if action == "answer":
        return "", True

    # Couldn't parse a tool call or final answer. Hint in GLM's own format
    # so the model continues coherently rather than rambling.
    hint = (
        "Error: I could not parse a tool call or final answer.\n"
        "Use the native tool-call format:\n"
        "<tool_call>code_interpreter\n"
        "<arg_key>code</arg_key>\n"
        "<arg_value>your python code here</arg_value>\n"
        "</tool_call>\n"
        "Or give the final answer as: Answer: \\boxed{answer}"
    )
    return _glm_observation(hint), False


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Custom generation function supporting tool calls"""
    assert not args.partial_rollout, "Partial rollout is not supported for " "this function at the moment."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Build initial prompt token ids via the model's NATIVE chat template
    # (apply_chat_template handles GLM's <|system|>/<|user|>/<|assistant|>
    # framing + tools spec injection).
    tool_specs = tool_registry.get_tool_specs()
    prompt_tokens_ids = format_conversation_with_tools(
        state.tokenizer, sample.prompt, tools=tool_specs
    )
    # Keep `prompt` as a string for wandb logging only; not fed back to SGLang.
    prompt = state.tokenizer.decode(prompt_tokens_ids)
    response = ""
    response_token_ids = []
    loss_masks = []
    tool_call_count = 0  # Track actual tool call rounds

    for turn in range(TOOL_CONFIGS["max_turns"]):
        # Check if total length exceeds max context length
        total_length = len(prompt_tokens_ids) + len(response_token_ids)
        if args.rollout_max_context_len is not None:
            max_context_length = args.rollout_max_context_len
        else:
            max_context_length = args.context_parallel_size * args.max_tokens_per_gpu
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break

        # Use token IDs instead of text
        current_token_ids = prompt_tokens_ids + response_token_ids
        payload = {
            "input_ids": current_token_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,  # Request log probabilities for training
        }

        # Log payload to wandb for debugging
        try:
            import wandb

            if wandb.run is not None:
                # Count available tools (from tool_specs)
                available_tools = len(tool_specs)
                # Count tools used in the current response (GLM observation block)
                tools_used = response.count("<tool_response>")

                wandb.log(
                    {
                        "debug/payload_length": len(prompt + response),
                        "debug/available_tools": available_tools,
                        "debug/tools_used": tools_used,
                        "debug/turn": turn,
                    }
                )
        except ImportError:
            pass  # wandb not available

        output = await post(url, payload)

        # Handle abort
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

        # Check length limit
        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        next_obs, done = await execute_predictions(cur_response)
        if done:
            break

        # Count tool calls (next_obs wraps tool output in <tool_response>)
        if "<tool_response>" in next_obs:
            tool_call_count += 1

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)

        # Add dummy log probs for observation tokens (they won't be used due to loss_mask=0)
        # Check if maximum tool call count reached
        if sample.rollout_log_probs is not None:
            sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)

            assert len(response_token_ids) == len(
                sample.rollout_log_probs
            ), f"Token/logp length mismatch at turn {turn}: {len(response_token_ids)} tokens vs {len(sample.rollout_log_probs)} logps"

        if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
            break

    # Set sample attributes
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks

    # Store payload information for wandb logging (GLM-native markers)
    sample.payload_text = prompt + response
    sample.payload_has_system = "<|system|>" in prompt + response
    sample.payload_has_tools = "<tools>" in prompt + response

    # Store tool call count for reward calculation
    sample.tool_call_count = tool_call_count

    # Set status
    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample, **kwargs):
    """Tool call reward function using math_dapo as primary reward model"""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # Build complete solution string
    solution_str = sample.prompt + sample.response

    # Get ground truth answer - label is a string, not a dict
    ground_truth = sample.label if sample.label is not None else ""

    # Get tool call count as num_turns
    num_turns = getattr(sample, "tool_call_count", 0)

    # use \\boxed{...} answer
    result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)

    # encourage model to call tools
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    return result
