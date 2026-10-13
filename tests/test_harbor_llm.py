import pytest

from slime_plugins.harbor.llm import SlimeSGLangLLM, _normalize_token_ids


NUM_GPUS = 0


class _TokenOutput:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _TensorLike:
    def __init__(self, value):
        self._value = value

    def tolist(self):
        return self._value


def test_normalize_token_ids_from_mapping_batch():
    assert _normalize_token_ids({"input_ids": [[1, 2, 3]]}) == [1, 2, 3]


def test_normalize_token_ids_from_object_input_ids():
    assert _normalize_token_ids(_TokenOutput([4, 5, 6])) == [4, 5, 6]


def test_normalize_token_ids_from_tensor_like_batch():
    assert _normalize_token_ids(_TensorLike([[7, 8, 9]])) == [7, 8, 9]


def test_normalize_token_ids_requires_single_sequence():
    with pytest.raises(ValueError, match="batch size 2"):
        _normalize_token_ids([[1], [2]])


class _ChatTemplateTokenizer:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return [10, 11, 12]


def test_encode_messages_forwards_chat_template_kwargs():
    llm = object.__new__(SlimeSGLangLLM)
    tokenizer = _ChatTemplateTokenizer()
    llm._tokenizer = tokenizer
    llm._apply_chat_template_kwargs = {"enable_thinking": False}
    llm._add_generation_prompt = True

    assert llm._encode_messages([{"role": "user", "content": "hi"}]) == [10, 11, 12]
    assert tokenizer.kwargs["tokenize"] is True
    assert tokenizer.kwargs["add_generation_prompt"] is True
    assert tokenizer.kwargs["enable_thinking"] is False


class _SpanTokenizer:
    role_ids = {"user": 10, "assistant": 20}

    def apply_chat_template(self, messages, **kwargs):
        ids = []
        for message in messages:
            ids.append(self.role_ids[message["role"]])
            ids.extend(ord(ch) for ch in message.get("content", ""))
            ids.append(99)
        if kwargs.get("add_generation_prompt"):
            ids.append(self.role_ids["assistant"])
        return ids

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(token_id) for token_id in token_ids if token_id not in {10, 20, 99})


class _QwenPrefillSpanTokenizer:
    prefill = "<think>\n\n</think>\n\n"

    def apply_chat_template(self, messages, **kwargs):
        text = self._render(messages, add_generation_prompt=kwargs.get("add_generation_prompt", False))
        return self._encode_text(text)

    def __call__(self, text, **kwargs):
        input_ids = self._encode_text(text)
        payload = {"input_ids": input_ids}
        if kwargs.get("return_offsets_mapping"):
            payload["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return payload

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(token_id) for token_id in token_ids)

    def _encode_text(self, text):
        return [ord(ch) for ch in text]

    def _render(self, messages, *, add_generation_prompt):
        chunks = []
        for index, message in enumerate(messages):
            role = message["role"]
            content = message.get("content", "")
            chunks.append(f"<|im_start|>{role}\n")
            if role == "assistant" and index == len(messages) - 1 and not add_generation_prompt:
                chunks.append(self.prefill)
            chunks.append(content)
            chunks.append("<|im_end|>\n")
        if add_generation_prompt:
            chunks.append(f"<|im_start|>assistant\n{self.prefill}")
        return "".join(chunks)


def test_last_assistant_history_projection_records_content_and_span():
    llm = object.__new__(SlimeSGLangLLM)
    tokenizer = _SpanTokenizer()
    llm._tokenizer = tokenizer
    llm._apply_chat_template_kwargs = {}
    llm._add_generation_prompt = True

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "next"},
    ]
    prompt_ids = llm._encode_messages(messages)
    projection = llm._last_assistant_history_projection(messages, prompt_ids)

    before = tokenizer.apply_chat_template(messages[:1], tokenize=True, add_generation_prompt=False)
    through = tokenizer.apply_chat_template(messages[:2], tokenize=True, add_generation_prompt=False)
    assert projection == {
        "last_assistant_message_content": "done",
        "last_assistant_message_span": [len(before), len(through)],
    }


def test_last_assistant_history_projection_uses_prompt_text_when_template_prefix_drifts():
    llm = object.__new__(SlimeSGLangLLM)
    tokenizer = _QwenPrefillSpanTokenizer()
    llm._tokenizer = tokenizer
    llm._apply_chat_template_kwargs = {"enable_thinking": False}
    llm._add_generation_prompt = True

    messages = [
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "u1"},
    ]
    prompt_ids = llm._encode_messages(messages)
    prompt_text = tokenizer.decode(prompt_ids)

    through = tokenizer.apply_chat_template(messages[:2], tokenize=True, add_generation_prompt=False)
    assert prompt_ids[: len(through)] != through

    projection = llm._last_assistant_history_projection(messages, prompt_ids, prompt_text=prompt_text)
    start, end = projection["last_assistant_message_span"]
    assert projection["last_assistant_message_content"] == "a0"
    assert tokenizer.decode(prompt_ids[start:end]) == "a0"


def test_context_limit_uses_explicit_override():
    llm = object.__new__(SlimeSGLangLLM)
    llm._tokenizer = object()
    llm._max_context_tokens = 131072

    assert llm.get_model_context_limit() == 131072


class _NoUserQueryTokenizer:
    """Mimics Qwen's chat template: raises unless a user turn is present."""

    def apply_chat_template(self, messages, **_kwargs):
        if not any(message.get("role") == "user" for message in messages):
            raise ValueError("No user query found in messages.")
        return [1] * sum(len(message.get("content", "")) for message in messages)

    def encode(self, text, **_kwargs):
        return [ord(ch) for ch in text]


def test_count_message_tokens_uses_real_chat_template():
    llm = object.__new__(SlimeSGLangLLM)
    llm._tokenizer = _NoUserQueryTokenizer()
    llm._apply_chat_template_kwargs = {}
    llm._add_generation_prompt = True

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    # apply_chat_template returns one token per content char here: 3 + 5 = 8.
    assert llm.count_message_tokens(messages) == 8


def test_count_message_tokens_empty_is_zero():
    llm = object.__new__(SlimeSGLangLLM)
    llm._tokenizer = _NoUserQueryTokenizer()
    llm._apply_chat_template_kwargs = {}
    llm._add_generation_prompt = True

    assert llm.count_message_tokens([]) == 0


def test_count_message_tokens_falls_back_when_template_rejects_shape():
    # System-only / no-user-turn shapes can appear transiently while Terminus2
    # unwinds message pairs. The chat template raises on them, but the counter
    # must degrade to a non-zero per-message estimate instead of propagating.
    llm = object.__new__(SlimeSGLangLLM)
    llm._tokenizer = _NoUserQueryTokenizer()
    llm._apply_chat_template_kwargs = {}
    llm._add_generation_prompt = True

    messages = [{"role": "system", "content": "abcd"}]  # 4 content tokens
    # Falls back to _estimate_message_tokens: 4 content + 4 envelope = 8.
    assert llm.count_message_tokens(messages) == 8


def test_estimate_message_tokens_counts_content_plus_envelope():
    llm = object.__new__(SlimeSGLangLLM)
    llm._tokenizer = _NoUserQueryTokenizer()

    messages = [
        {"role": "system", "content": "ab"},  # 2 + 4
        {"role": "assistant", "content": "xyz"},  # 3 + 4
    ]
    assert llm._estimate_message_tokens(messages) == (2 + 4) + (3 + 4)
