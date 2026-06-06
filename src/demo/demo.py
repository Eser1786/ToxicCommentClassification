import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH     = 'models/distilbert_toxic.pt'
TOKENIZER_PATH = 'models/tokenizer'
LABEL_COLS     = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
THRESHOLD      = 0.5
MAX_LEN        = 128

# ── Model definition (giống hệt Colab) ────────────────────────────────────
class DistilBertToxic(nn.Module):
    def __init__(self, n_labels=6):
        super().__init__()
        self.bert       = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout    = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, n_labels)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

# ── Load model ─────────────────────────────────────────────────────────────
print("Đang load model...")
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = DistilBertTokenizerFast.from_pretrained(TOKENIZER_PATH)
model     = DistilBertToxic()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print(f"Load xong — đang dùng: {device}\n")

# ── Predict function ───────────────────────────────────────────────────────
def predict(text):
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids      = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs  = torch.sigmoid(logits).cpu().numpy()[0]

    return probs

# ── Hiển thị kết quả ──────────────────────────────────────────────────────
def display(text, probs):
    print("\nKết quả phân tích:")
    print("─" * 45)

    any_detected = False
    for label, prob in zip(LABEL_COLS, probs):
        bar      = '█' * int(prob * 10)
        pad      = '░' * (10 - int(prob * 10))
        detected = prob >= THRESHOLD
        status   = '⚠ DETECTED' if detected else '✓ clean'
        if detected:
            any_detected = True
        print(f"  {label:<15} {bar}{pad}  {prob:.2f}  {status}")

    print("─" * 45)

    # Mức độ tổng thể
    max_prob = max(probs)
    if max_prob >= 0.8:
        level = "🔴 RẤT ĐỘC HẠI"
    elif max_prob >= 0.5:
        level = "🟠 ĐỘC HẠI"
    elif max_prob >= 0.3:
        level = "🟡 CÓ DẤU HIỆU"
    else:
        level = "🟢 BÌNH THƯỜNG"

    print(f"  Mức độ tổng thể: {level}")
    print()

# ── Main loop ─────────────────────────────────────────────────────────────
print("=" * 45)
print("      TOXIC COMMENT DETECTOR")
print("      Gõ 'quit' để thoát")
print("=" * 45)

while True:
    text = input("\nNhập comment: ").strip()

    if text.lower() == 'quit':
        print("Tạm biệt!")
        break

    if not text:
        print("Comment không được để trống.")
        continue

    probs = predict(text)
    display(text, probs)