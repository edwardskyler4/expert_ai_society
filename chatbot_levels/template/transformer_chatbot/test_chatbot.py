"""Run with: python -m unittest -v. Tensor tests require PyTorch."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from data import ASSISTANT, BOS, EOS, IGNORE, OFFSET, PAD, USER, VOCAB_SIZE
from data import build_prompt, decode, encode, encode_conversation, load_records, split_records

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch.nn import functional as F
    from chat import generate, sample_token
    from model import Config, TransformerChatbot
    from train import collate


def pair(user="hi", assistant="hello"):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


class DataTests(unittest.TestCase):
    def test_unicode_roundtrip(self):
        text = "Hello, café! 🦉 你好"
        self.assertEqual(decode(encode(text)), text)
        self.assertTrue(all(OFFSET <= value < VOCAB_SIZE for value in encode(text)))

    def test_special_markers_are_not_parsed_from_user_text(self):
        self.assertNotIn(ASSISTANT, encode("<assistant>"))
        self.assertEqual(decode([BOS] + encode("hi") + [EOS]), "hi")

    def test_next_token_shift_and_assistant_mask(self):
        x, y = encode_conversation(pair("a", "b"))
        self.assertEqual(x, [BOS, USER, *encode("a"), EOS, ASSISTANT, *encode("b")])
        self.assertEqual(y, [IGNORE, IGNORE, IGNORE, IGNORE, *encode("b"), EOS])
        # The assistant marker predicts the first reply byte, not itself.
        self.assertEqual(y[x.index(ASSISTANT)], encode("b")[0])

    def test_multiturn_supervision(self):
        messages = pair("hi", "hello") + pair("bye", "goodbye")
        x, y = encode_conversation(messages)
        self.assertEqual(len(x), len(y))
        supervised = [value for value in y if value != IGNORE]
        self.assertEqual(supervised, encode("hello") + [EOS] + encode("goodbye") + [EOS])

    def test_prompt_matches_training_prefix(self):
        history = [("hi", encode("hello"))]
        prompt = build_prompt(history, "bye")
        x, _ = encode_conversation(pair("hi", "hello") + pair("bye", "goodbye"))
        self.assertEqual(x[:len(prompt)], prompt)

    def test_invalid_roles_rejected(self):
        with self.assertRaises(ValueError):
            encode_conversation(list(reversed(pair())))
        with self.assertRaises(ValueError):
            encode_conversation(pair()[:1])

    def test_split_is_disjoint_and_reproducible(self):
        records = [encode_conversation(pair(str(i), "answer")) for i in range(10)]
        train, val = split_records(records, 0.2, 42)
        self.assertEqual((train, val), split_records(records, 0.2, 42))
        self.assertFalse({tuple(x) for x, _ in train} & {tuple(x) for x, _ in val})
        self.assertEqual((len(train), len(val)), (8, 2))

    def test_oversize_and_duplicate_records_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            line = json.dumps({"messages": pair()})
            path.write_text(line + "\n" + line + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_records(path, 100)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                load_records(path, 2)


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.model = TransformerChatbot(Config(block_size=32, width=16, heads=2, layers=1, dropout=0))
        self.model.eval()

    def test_future_tokens_cannot_affect_past_logits(self):
        a = torch.tensor([[BOS, USER, 20, 21, 22]])
        b = a.clone()
        b[0, -1] = 100
        with torch.no_grad():
            torch.testing.assert_close(self.model(a)[:, :-1], self.model(b)[:, :-1])

    def test_right_padding_does_not_change_supervised_loss(self):
        record = encode_conversation(pair("a", "b"))
        x, y = collate([record])
        padded_x = F.pad(x, (0, 3), value=PAD)
        padded_y = F.pad(y, (0, 3), value=IGNORE)
        with torch.no_grad():
            def loss(inputs, labels):
                logits = self.model(inputs)
                return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE)
            torch.testing.assert_close(loss(x, y), loss(padded_x, padded_y))

    def test_gradient_updates_reduce_loss_on_one_example(self):
        x, y = collate([encode_conversation(pair("a", "b"))])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.02)
        losses = []
        for _ in range(40):
            logits = self.model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=IGNORE)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0] * 0.5)

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(self.model.state_dict(), path)
            copy = TransformerChatbot(self.model.config).eval()
            copy.load_state_dict(torch.load(path, weights_only=True))
            x = torch.tensor([[BOS, USER, 20]])
            with torch.no_grad():
                torch.testing.assert_close(self.model(x), copy(x))

    def test_sampling_suppresses_role_markers(self):
        logits = torch.zeros(VOCAB_SIZE)
        logits[[PAD, BOS, USER, ASSISTANT]] = 1000
        logits[OFFSET + ord("a")] = 100
        for temperature in [0, 0.7]:
            self.assertEqual(sample_token(logits, temperature, 1, 0.9), OFFSET + ord("a"))

    def test_generation_is_bounded_and_handles_long_prompts(self):
        prompt = build_prompt([], "a" * 100)
        reply = generate(self.model, prompt, "cpu", max_tokens=5, temperature=0)
        self.assertLessEqual(len(reply), 5)
        self.assertTrue(all(OFFSET <= value < VOCAB_SIZE for value in reply))


if __name__ == "__main__":
    unittest.main()
