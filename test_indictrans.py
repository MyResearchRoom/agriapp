import os
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor


MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    token=os.environ.get("HF_TOKEN")
)

print("Loading model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    token=os.environ.get("HF_TOKEN")
).to(DEVICE)

model.eval()

print("Model loaded.")

ip = IndicProcessor(inference=True)

src_lang = "eng_Latn"
tgt_lang = "hin_Deva"

text = (
    "The plant has leaf rust. "
    "Remove infected leaves and apply appropriate treatment."
)

print("Translating...")

batch = ip.preprocess_batch(
    [text],
    src_lang=src_lang,
    tgt_lang=tgt_lang
)

inputs = tokenizer(
    batch,
    truncation=True,
    padding="longest",
    return_tensors="pt"
).to(DEVICE)

with torch.no_grad():
    generated = model.generate(
        **inputs,
        num_beams=5,
        num_return_sequences=1,
        max_length=256
    )

decoded = tokenizer.batch_decode(
    generated,
    skip_special_tokens=True
)

result = ip.postprocess_batch(
    decoded,
    lang=tgt_lang
)[0]

print()
print("English:")
print(text)

print()
print("Hindi:")
print(result)

with open("translation_test.txt", "w", encoding="utf-8") as f:
    f.write("English:\n")
    f.write(text)
    f.write("\n\nHindi:\n")
    f.write(result)

print()
print("Translation saved to translation_test.txt")