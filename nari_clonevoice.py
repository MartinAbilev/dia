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

# You should put the transcript of the voice you want to clone
# We will use the audio created by running simple.py as an example.
# Note that you will be REQUIRED TO RUN simple.py for the script to work as-is.
clone_from_text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
clone_from_audio = "simple.mp3"

# For your custom needs, replace above with below and add your audio file to this directory:
# clone_from_text = "[S1] ... [S2] ... [S1] ... corresponding to your_audio_name.mp3"
# clone_from_audio = "your_audio_name.mp3"

# Text to generate
text_to_generate = "[S1] Hello, how are you? [S2] I'm good, thank you. [S1] What's your name? [S2] My name is Dia. [S1] Nice to meet you. [S2] Nice to meet you too."

# It will only return the audio from the text_to_generate
output = model.generate(
    clone_from_text + text_to_generate, audio_prompt=clone_from_audio, use_torch_compile=True, verbose=True
)

model.save_audio("voice_clone.mp3", output)

# Print end time
print("READY:", x.date(), x.now())

# Print GPU memory usage
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")




