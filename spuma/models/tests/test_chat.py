from langchain_core.messages import AIMessage, ToolMessage

from spuma.models.chat import RAW_TEXT_KEY, ChatFormatter, parse_qwen_output, to_chat_dicts


def test_parse_tool_calls_and_reasoning():
    text = '<think>hmm</think>Let me solve.\n<tool_call>\n{"name": "sympy_solve", "arguments": {"equations": ["x-1"]}}\n</tool_call>'
    p = parse_qwen_output(text)
    assert p.reasoning == "hmm" and p.content == "Let me solve."
    assert p.tool_calls == [{"name": "sympy_solve", "args": {"equations": ["x-1"]}, "id": "call_0"}]
    bad = parse_qwen_output("<tool_call>{not json}</tool_call>")
    assert not bad.tool_calls and "malformed" in bad.invalid_tool_calls[0]["error"]


def test_to_chat_dicts_replays_raw_text():
    raw = 'x <tool_call>{"name": "a", "arguments": {}}</tool_call>'
    msgs = [AIMessage("x", additional_kwargs={RAW_TEXT_KEY: raw}), ToolMessage("4", tool_call_id="call_0")]
    assert to_chat_dicts(msgs) == [{"role": "assistant", "content": raw}, {"role": "tool", "content": "4"}]


def test_formatter_roundtrip(tokenizer):
    chat = ChatFormatter(tokenizer, enable_thinking=False)
    ids = chat.encode("Solve for x")
    assert chat.decode(ids) == "Solve for x" and chat.count_tokens("Solve for x") == len(ids)
    assert len(chat.template_hash) == 12 and chat.vocab_size == len(tokenizer)
