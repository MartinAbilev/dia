from dia.model import Dia
import os
import datetime
import torch

# Set cache directory
cache_dir = "E:/huggingface_cache"
os.environ["HF_HOME"] = cache_dir  # Set Hugging Face cache directory

print(torch.__version__)  # Should print 2.6.0+cu124
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Should print your RTX GPU name
print(torch.version.cuda)  # Should print 12.4

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model with float16 precision and move to GPU
x = datetime.datetime.now()
try:
    model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16", device="cuda")
    print("Dia with CUDA")
except TypeError:
    print("Device parameter not supported, falling back to default.")
    model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")

# Input text
text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."

# Print start time
print("START:", x.date(), x.now())

# Generate output (ensure input is on GPU if required by the model)
output = model.generate(text, use_torch_compile=True, verbose=True)

# Save audio
model.save_audio("simple.mp3", output)

# Print end time
print("READY:", x.date(), x.now())

# Print GPU memory usage
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
