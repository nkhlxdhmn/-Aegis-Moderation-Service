from transformers import AutoTokenizer, pipeline

MODEL_DIR = "/home/ubuntu/Desktop/muril_test/muril_test/models/muril_abuse_final"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, fix_mistral_regex=True)
except TypeError:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

classifier = pipeline(
    "text-classification",
    model=MODEL_DIR,
    tokenizer=tokenizer,
    truncation=True,
    max_length=128,
    device=0,
)

texts = [
    "You are an idiot and your ideas are worthless.",
    "aaj mausam bahut accha hai",
    "teri ma ki aankh tu kuch nahi kar sakta",
    "Good morning everyone, have a nice day",
    "saala kamine bakwaas band kar",
    "please share your feedback with us",
]

print(f"\n{'Label':<15} {'Score':>6}  Text")
print("-" * 70)
for text in texts:
    result = classifier(text)[0]
    flag = "✅" if result["label"] == "non_abusive" else "🚨"
    print(f"{flag} {result['label']:<13} {result['score']:.3f}  {text}")
