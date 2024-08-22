import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Load the model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")

# Download and prepare the image
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Define the question
question = "how many dogs are in the picture?"

# Process the inputs
inputs = processor(raw_image, question, return_tensors="pt").to("cuda")

# Get the model output
out = model.generate(**inputs)

# Decode and print the answer
print(processor.decode(out[0], skip_special_tokens=True))
