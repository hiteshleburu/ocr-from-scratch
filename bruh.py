import torch
from PIL import Image
from src.recognition import OCRModel, load_vocab, build_transform, generate

stoi, itos = load_vocab("src/vocab.json")   # your existing saved vocab file
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = OCRModel(vocab_size=39).to(device)
model.load_state_dict(torch.load("checkpoints/model_50k_v2_best.pt", map_location=device))
model.eval()

transform = build_transform(augment=False)
img = transform(Image.open("data/synth/eval_fresh/eval_00000.png").convert('RGB'))
pred = generate(model, img, itos, device)
print(pred)