from transformers import AutoTokenizer

from training.config import MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

TEST_STRINGS = [
    ("Pure Bengali script", "আজকে ঢাকায় অনেক জ্যাম ছিল, সত্যিই অনেক কষ্ট হয়েছে"),
    ("Colloquial Banglish", "bhai seriously কষ্ট হইছে, amader ki hobe 😭"),
    ("Romanized Bengali", "vai eta ki hoise, amio same rokom vabsi ekdom"),
    ("Heavy code-switching", "bhai এইটা দেইখা হাসতে হাসতে শেষ 💀 seriously amader ki hobe"),
    ("Pure English", "bhai seriously what is happening in this country today"),
    ("Political Bengali", "সরকারের দুর্নীতি আর সহ্য করা যাচ্ছে না, জনগণ এখন রাস্তায়"),
]

print(f"Model: {MODEL_NAME}\n")
print(f"{'Test':<25} {'Chars':>6} {'Tokens':>7} {'Ratio':>7} {'Assessment'}")
print("─" * 70)

for label, text in TEST_STRINGS:
    tokens = tokenizer.encode(text)
    ratio = len(tokens) / len(text)

    if ratio < 1.5:
        assessment = "✅ Excellent"
    elif ratio < 2.5:
        assessment = "⚠ Acceptable"
    else:
        assessment = "❌ Inefficient"

    print(f"{label:<25} {len(text):>6} {len(tokens):>7} {ratio:>7.2f} {assessment}")
