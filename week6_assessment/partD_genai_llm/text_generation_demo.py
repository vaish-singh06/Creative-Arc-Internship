from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompts = [
    "AI will change education by",
    "The future of healthcare is",
    "Climate change can be solved by",
    "A student should learn AI because",
    "Neural networks work by"
]

for p in prompts:
    out = generator(p, max_length=50)
    print(out[0]["generated_text"])
