from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
print("Loading processor...")
p = AutoProcessor.from_pretrained("google/gemma-4-26b-a4b-it")
print("Processor OK")
print("Loading model...")
m = AutoModelForImageTextToText.from_pretrained(
    "google/gemma-4-26b-a4b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("Model OK, VRAM: %.0fMB" % (torch.cuda.memory_allocated() / 1e6))
