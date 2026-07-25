from transformers import AutoModel
model = AutoModel.from_pretrained("OpenGVLab/InternVL2_5-8B", trust_remote_code=True)
print("Forward signature:", model.forward.__code__.co_varnames)
