import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from spuma.models.chat import ChatFormatter, parse_qwen_output, to_chat_dicts


def test_parse_reasoning_and_content():
    p = parse_qwen_output("<think>hmm</think>Let me solve.\n\\boxed{1}\n")
    assert p.reasoning == "hmm" and p.content == "Let me solve.\n\\boxed{1}"
    assert parse_qwen_output("no thinking").reasoning == ""


def test_to_chat_dicts_system_and_user_only():
    assert to_chat_dicts([SystemMessage("s"), HumanMessage("u")]) == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    with pytest.raises(TypeError):
        to_chat_dicts([AIMessage("a")])


def test_formatter_roundtrip(tokenizer):
    chat = ChatFormatter(tokenizer, enable_thinking=False)
    ids = chat.encode("Solve for x")
    assert chat.decode(ids) == "Solve for x" and chat.count_tokens("Solve for x") == len(ids)
    assert len(chat.template_hash) == 12 and chat.vocab_size == len(tokenizer)
