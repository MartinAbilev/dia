import io
import tempfile
from django.http import HttpResponse, StreamingHttpResponse
import os
import datetime
from django.shortcuts import render
import torch
from django.http import FileResponse, HttpResponse
from django.conf import settings
from dia.model import Dia

from django.template.loader import get_template
# def index(request):
#     try:
#         template = get_template('index.html')
#         return HttpResponse(f"Template found: {template.origin.name}")
#     except Exception as e:
#         return HttpResponse(f"Template error: {str(e)}")

def index(request):
    return render(request, 'index.html')

# Set cache directory for Hugging Face
cache_dir = "E:/huggingface_cache"
os.environ["HF_HOME"] = cache_dir

# Check PyTorch and CUDA setup
if not torch.cuda.is_available():
    print("CUDA is not available. Please check your GPU setup.", status=500)
print(torch.__version__)  # Should print 2.6.0+cu124
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Should print your RTX GPU name
print(torch.version.cuda)  # Should print 12.4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model with float16 precision
x = datetime.datetime.now()
try:
    model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16", device="cuda")
    print("Dia loaded with CUDA")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Failed to load model: {str(e)}", status=500)

def generate_audio_view(request):

    # Get text from query parameter, fallback to default
    text = request.GET.get('text')
    if not text:
        text = (
            "[S1] Dia is an open weights text to dialogue model. "
            "[S2] You get full control over scripts and voices. "
            "[S1] Wow. Amazing. (laughs) "
            "[S2] Try it now on Git hub or Hugging Face."
        )
    else:
        # Optionally validate or sanitize the text input
        text = text.strip()
        if not text:
            return HttpResponse("Text parameter cannot be empty.", status=400)

    # Define output file path
    output_file = os.path.join(settings.MEDIA_ROOT, "sample.mp3")

    try:
        # Generate audio output
        print("START:", x.date(), x.now())
        output = model.generate(
            text,
            use_torch_compile=True,
            verbose=True,
        )
        print("READY:", datetime.datetime.now().date(), datetime.datetime.now())

        # Use a temporary file to save the audio
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name
            model.save_audio(temp_path, output)  # Save to temporary file
            temp_file.seek(0)
            # Read the temporary file into a BytesIO buffer
            audio_buffer = io.BytesIO(temp_file.read())

        # Clean up the temporary file
        os.remove(temp_path)

        # Reset buffer position
        audio_buffer.seek(0)

        # Stream the audio data
        print("Creating Stream")
        response = StreamingHttpResponse(
            audio_buffer,
            content_type='audio/mpeg'
        )

        # Set Content-Disposition to inline for playback
        print("Sending buffer")
        response['Content-Disposition'] = 'inline; filename="sample.mp3"'
        return response
    finally:
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
