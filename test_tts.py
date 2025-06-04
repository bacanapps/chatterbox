import torch
import torchaudio
from chatterbox import tts

# Patch torch.load to ensure it loads CUDA-trained weights on CPU
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(
    *args, **({**kwargs, "map_location": "cpu"} if "map_location" not in kwargs else kwargs)
)

# Step 1: Load model
print("🔄 Loading model...")
model = tts.ChatterboxTTS.from_pretrained(device="cpu")
print("✅ Model loaded.")

# Step 2: Generate speech
text = "Hoje é um ótimo dia para testar inteligência artificial."
print(f"🎙️ Generating speech for: '{text}'")
waveform = model.generate(text)
print("✅ Audio generated.")

# Step 3: Save audio
output_path = "output.wav"
print(f"💾 Saving to {output_path}")
torchaudio.save(output_path, waveform, sample_rate=model.sr)
print(f"🎉 Done! Audio saved to {output_path}")