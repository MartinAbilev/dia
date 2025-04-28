torch-2.5.1+cu118

# Install directly from GitHub
pip install git+https://github.com/nari-labs/dia.git

# change huging chache  dir
Step-by-Step Fix
Verify and Create the Target Directory (E:\huggingface_cache):
Open File Explorer and navigate to your E: drive.
Check if the folder E:\huggingface_cache exists. If it doesn’t:
Create it manually in File Explorer (right-click > New > Folder, name it huggingface_cache).
Alternatively, use Command Prompt (as Administrator):
cmd

Copy
mkdir E:\huggingface_cache
Ensure the E: drive is accessible and has enough space for your models (some models are 10GB+).
Check the Source Path (C:\Users\{user}\.cache\huggingface):
Open File Explorer and navigate to C:\Users\{user}\.cache.
If .cache doesn’t exist, create it:
cmd

Copy
mkdir C:\Users\{user}\.cache
If huggingface exists inside .cache, move or delete it:
Move: If it contains models you want to keep, move it to E:\huggingface_cache:
cmd

Copy
move C:\Users\{user}\.cache\huggingface E:\huggingface_cache
Delete: If you don’t need the existing cache, delete it (after backing up if necessary):
cmd

Copy
rmdir /S /Q C:\Users\{user}\.cache\huggingface
Ensure C:\Users\{user}\.cache\huggingface does not exist before running mklink, as mklink can’t create a junction if the source path already exists.
Run mklink Again in Command Prompt as Administrator:
Open Command Prompt as Administrator:
Press Win + S, type cmd, right-click Command Prompt, and select Run as administrator.
Run the mklink command:
cmd

Copy
mklink /J "C:\Users\{user}\.cache\huggingface" "E:\huggingface_cache"
If successful, you’ll see:
text

Copy
Junction created for C:\Users\{user}\.cache\huggingface <<===>> E:\huggingface_cache
Verify the Junction:
In File Explorer, go to C:\Users\{user}\.cache\huggingface. It should appear with a shortcut icon and show the contents of E:\huggingface_cache.
Test by downloading a small model in Python:
python

Copy
from transformers import AutoModel
AutoModel.from_pretrained("distilbert-base-uncased")


# test cuda

import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Should display your RTX GPU name

# if false

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install python-decouple
