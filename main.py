from transformers import pipeline
from PIL import Image
import requests
import torch

local_cache_path = "./my_local_models"
device_type = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-1.5-4b-it",
    torch_dtype=torch.bfloat16,
    device=device_type,
    model_kwargs={"cache_dir": local_cache_path}
)

