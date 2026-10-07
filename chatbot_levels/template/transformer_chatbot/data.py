"""Byte tokenizer and conversation encoding; no third-party dependencies."""
import json
import random
from pathlib import Path

PAD, BOS, EOS, USER, ASSISTANT = range(5)
OFFSET = 5
VOCAB_SIZE = 256 + OFFSET
IGNORE = -100


def encode(text):
    return [value + OFFSET for value in text.encode("utf-8")]


def decode(tokens):
    return bytes(token - OFFSET for token in tokens if OFFSET <= token < VOCAB_SIZE).decode(
        "utf-8", errors="replace"
    )


def encode_conversation(messages):
    """Return tokens and labels. Only assistant text and its EOS incur loss."""
    if not isinstance(messages, list) or not messages or len(messages) % 2:
        raise ValueError("Each conversation must contain complete user/assistant pairs.")
    tokens, labels = [BOS], [IGNORE]
    for index, message in enumerate(messages):
        expected = "user" if index % 2 == 0 else "assistant"
        if not isinstance(message, dict) or message.get("role") != expected:
            raise ValueError(f"Message {index + 1} must have role {expected!r}.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Message content must be a nonempty string.")
        body = encode(content) + [EOS]
        tokens.extend([USER if expected == "user" else ASSISTANT] + body)
        labels.extend([IGNORE] + (body if expected == "assistant" else [IGNORE] * len(body)))
    # Input t predicts label t+1. The assistant role marker predicts the first reply byte.
    return tokens[:-1], labels[1:]


def load_records(path, block_size):
    records, seen = [], set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            messages = json.loads(line)["messages"]
            x, y = encode_conversation(messages)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        if len(x) > block_size:
            raise ValueError(
                f"{path}:{number}: {len(x)} tokens exceeds block size {block_size}. "
                "Shorten the conversation or increase --block-size."
            )
        identity = tuple(x)
        if identity in seen:
            raise ValueError(f"{path}:{number}: duplicate conversation; remove it before splitting.")
        seen.add(identity)
        records.append((x, y))
    if len(records) < 2:
        raise ValueError("Provide at least two distinct conversations for train/validation splitting.")
    return records


def split_records(records, validation_fraction, seed):
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1.")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    count = min(len(shuffled) - 1, max(1, round(len(shuffled) * validation_fraction)))
    return shuffled[count:], shuffled[:count]


def build_prompt(history, user_text):
    tokens = [BOS]
    for user, assistant in history:
        tokens.extend([USER] + encode(user) + [EOS, ASSISTANT] + assistant + [EOS])
    return tokens + [USER] + encode(user_text) + [EOS, ASSISTANT]
