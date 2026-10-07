"""A tiny intent chatbot trained from scratch with PyTorch.

Install: python3 -m pip install torch
Run:     python3 chatbot.py

Edit INTENTS to teach it new example phrases and responses. It trains again
each time it starts. No pretrained model, API key, or GPU is required.
"""

import random
import re

import torch
from torch import nn


# 1. Training data: each intent is a category of messages.
INTENTS = {
    "greeting": {
        "patterns": ["hi", "hello", "hey", "good morning", "good evening"],
        "responses": ["Hello!", "Hi there!", "Hey! How can I help?"],
    },
    "goodbye": {
        "patterns": ["bye", "goodbye", "see you later", "take care"],
        "responses": ["Goodbye!", "See you later!"],
    },
    "thanks": {
        "patterns": ["thanks", "thank you", "thanks a lot", "much appreciated"],
        "responses": ["You're welcome!", "Happy to help!"],
    },
    "name": {
        "patterns": ["what is your name", "who are you", "tell me your name"],
        "responses": ["I'm TinyBot, a chatbot you trained with PyTorch."],
    },
    "help": {
        "patterns": ["help", "help me", "what can you do", "how can you help"],
        "responses": ["I can greet you, introduce myself, and respond to thanks."],
    },
}
FALLBACK = "I'm not sure what you mean. Try saying 'hello' or 'help'."


def tokenize(text):
    """Lowercase text and split it into words, ignoring punctuation."""
    return re.findall(r"[a-z]+", text.lower())


def vectorize(text, vocabulary):
    """One number per vocabulary word: 1 if present, otherwise 0."""
    words = set(tokenize(text))
    return torch.tensor([float(word in words) for word in vocabulary])


# 2. The model: word features -> hidden layer -> one score per intent.
class ChatbotModel(nn.Module):
    def __init__(self, vocabulary_size, intent_count):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(vocabulary_size, 16),
            nn.ReLU(),
            nn.Linear(16, intent_count),
        )

    def forward(self, x):
        return self.network(x)


# 3. Learn weights from our labeled examples.
def train(epochs=500):
    torch.manual_seed(42)
    # This dataset is so small that extra CPU threads add overhead.
    torch.set_num_threads(1)
    tags = list(INTENTS)
    vocabulary = sorted({
        word
        for intent in INTENTS.values()
        for pattern in intent["patterns"]
        for word in tokenize(pattern)
    })
    examples = [
        (pattern, label)
        for label, tag in enumerate(tags)
        for pattern in INTENTS[tag]["patterns"]
    ]
    x = torch.stack([vectorize(pattern, vocabulary) for pattern, _ in examples])
    y = torch.tensor([label for _, label in examples], dtype=torch.long)

    model = ChatbotModel(len(vocabulary), len(tags))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()       # Clear gradients from the previous step.
        logits = model(x)           # Predict raw intent scores.
        loss = loss_fn(logits, y)   # Compare scores with the correct labels.
        loss.backward()            # Calculate gradients using autograd.
        optimizer.step()           # Update the model's weights.
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | loss: {loss.item():.4f}")

    model.eval()
    return model, vocabulary, tags


# 4. Classify new messages and select a handwritten response.
@torch.no_grad()
def respond(text, model, vocabulary, tags, threshold=0.75):
    features = vectorize(text, vocabulary)
    if features.sum().item() == 0:
        return FALLBACK  # Empty input or no recognized words.
    probabilities = torch.softmax(model(features.unsqueeze(0)), dim=1)
    confidence, index = probabilities.max(dim=1)
    if confidence.item() < threshold:
        return FALLBACK
    tag = tags[index.item()]
    return random.choice(INTENTS[tag]["responses"])


def main():
    print("Training TinyBot...")
    model, vocabulary, tags = train()
    print("\nTinyBot is ready. Type /quit to exit.")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTinyBot: Goodbye!")
            break
        if message.lower() == "/quit":
            print("TinyBot: Goodbye!")
            break
        print("TinyBot:", respond(message, model, vocabulary, tags))


if __name__ == "__main__":
    main()
