"""Generate replies using a trained checkpoint and a sliding conversation window."""
import argparse
from pathlib import Path

import torch

from data import ASSISTANT, BOS, EOS, PAD, USER, build_prompt, decode
from model import Config, TransformerChatbot


def sample_token(logits, temperature, top_k, top_p):
    logits = logits.clone()
    # A reply may contain UTF-8 bytes or EOS, but never a new role or padding.
    logits[[PAD, BOS, USER, ASSISTANT]] = float("-inf")
    if temperature == 0:
        return int(logits.argmax().item())
    logits /= temperature
    if top_k:
        cutoff = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits[logits < cutoff] = float("-inf")
    ordered, indices = torch.sort(logits, descending=True)
    cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
    remove = cumulative > top_p
    # Keep the token that crosses the nucleus threshold.
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    logits[indices[remove]] = float("-inf")
    return int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())


@torch.no_grad()
def generate(model, prompt, device, max_tokens=160, temperature=0.7, top_k=40, top_p=0.9):
    if temperature < 0 or top_k < 0 or not 0 < top_p <= 1 or max_tokens < 1:
        raise ValueError("Invalid generation settings.")
    model.eval()
    tokens, reply = list(prompt), []
    for _ in range(max_tokens):
        context = tokens[-model.config.block_size:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        next_token = sample_token(model(x)[0, -1], temperature, top_k, top_p)
        if next_token == EOS:
            break
        tokens.append(next_token)
        reply.append(next_token)
    return reply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--history-turns", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.history_turns < 0 or args.threads < 1 or args.temperature < 0 or args.top_k < 0 \
            or not 0 < args.top_p <= 1 or args.max_tokens < 1:
        parser.error("Invalid generation or history settings.")
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "byte_transformer_chatbot_v1":
        raise ValueError("Checkpoint format is not supported.")
    model = TransformerChatbot(Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(args.device).eval()
    history = []
    print("Commands: /reset clears history; /quit exits.")
    print(f"Context window: {model.config.block_size} byte/special tokens; older tokens are discarded.")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if user == "/quit":
            break
        if user == "/reset":
            history.clear()
            print("History cleared.")
            continue
        if not user:
            continue
        prompt = build_prompt(history, user)
        reply = generate(model, prompt, args.device, args.max_tokens, args.temperature, args.top_k, args.top_p)
        print("Bot:", decode(reply) or "[The model generated an empty reply.]")
        if args.history_turns:
            history.append((user, reply))
            history = history[-args.history_turns:]


if __name__ == "__main__":
    main()
