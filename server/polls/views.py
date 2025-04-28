from django.http import HttpResponse
import os
import datetime
import torch
from django.http import FileResponse, HttpResponse
from django.conf import settings
from dia.model import Dia

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


# Set cache directory for Hugging Face
cache_dir = "E:/huggingface_cache"
os.environ["HF_HOME"] = cache_dir

# Check PyTorch and CUDA setup
# if not torch.cuda.is_available():
#     return HttpResponse("CUDA is not available. Please check your GPU setup.", status=500)
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
    # return HttpResponse(f"Failed to load model: {str(e)}", status=500)

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
        print("Generating text:", text)
        output = model.generate(
            text,
            use_torch_compile=True,
            verbose=True
        )

        # Save the generated audio
        model.save_audio(output_file, output)
        print(f"Audio saved to: {output_file}")

    except Exception as e:
        print(f"Error generating audio: {e}")
        return HttpResponse(f"Failed to generate audio: {str(e)}", status=500)

    # Serve the generated audio file
    try:
        if os.path.exists(output_file):
            # Open the file and keep it open for FileResponse
            file_obj = open(output_file, 'rb')
            response = FileResponse(
                file_obj,
                content_type='audio/mpeg',
                as_attachment=True,
                filename="sample.mp3"
            )
            # Note: File will be closed automatically by FileResponse
            return response
        else:
            return HttpResponse("Generated audio file not found.", status=500)
    except Exception as e:
        print(f"Error serving file: {e}")
        return HttpResponse(f"Error serving audio file: {str(e)}", status=500)
    finally:
        # Print timing and GPU memory usage
        print("START:", x.date(), x.now())
        print("READY:", datetime.datetime.now().date(), datetime.datetime.now())
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
