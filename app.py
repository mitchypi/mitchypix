import os
from flask import Flask, request, render_template, redirect, url_for
import torch
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
import time

app = Flask(__name__)
app.secret_key = 'hi'

ITEMS_PER_PAGE = 20
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CLIP model
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Load BLIP model for image captioning
captioning_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
captioning_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# File to store embeddings and paths
EMBEDDINGS_FILE = 'embeddings.npy'
PATHS_FILE = 'image_paths.npy'
CAPTIONS_FILE = 'image_captions.npy'

# Load existing data
if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(PATHS_FILE) and os.path.exists(CAPTIONS_FILE):
    embeddings = np.load(EMBEDDINGS_FILE)
    image_paths = np.load(PATHS_FILE)
    image_captions = np.load(CAPTIONS_FILE)
else:
    embeddings = np.array([])
    image_paths = np.array([])
    image_captions = np.array([])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    global embeddings, image_paths, image_captions
    if 'files' not in request.files:
        return 'No file part'
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return 'No selected file'
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static', 'uploads', filename)
            file.save(file_path)
            file_path = file_path.replace('\\', '/')
            
            total_start_time = time.time()
            
            caption_start_time = time.time()
            caption = generate_caption(file_path)
            caption_end_time = time.time()
            caption_time = caption_end_time - caption_start_time
            
            embedding_start_time = time.time()
            image = Image.open(file_path)
            inputs = clip_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
            embedding_end_time = time.time()
            embedding_time = embedding_end_time - embedding_start_time
            
            total_end_time = time.time()
            total_time = total_end_time - total_start_time
            
            caption_ratio = caption_time / total_time
            embedding_ratio = embedding_time / total_time
            
            print(f"Time taken for captioning: {caption_time:.2f} seconds")
            print(f"Time taken for creating embedding: {embedding_time:.2f} seconds")
            print(f"Total processing time: {total_time:.2f} seconds")
            print(f"Captioning time ratio: {caption_ratio:.2f}")
            print(f"Embedding time ratio: {embedding_ratio:.2f}")
            
            new_embedding = image_features.cpu().numpy()
            if embeddings.size == 0:
                embeddings = new_embedding
            else:
                embeddings = np.vstack((embeddings, new_embedding))
            image_paths = np.append(image_paths, os.path.join('uploads', filename).replace('\\', '/'))
            image_captions = np.append(image_captions, caption)
    
    np.save(EMBEDDINGS_FILE, embeddings)
    np.save(PATHS_FILE, image_paths)
    np.save(CAPTIONS_FILE, image_captions)
    
    return redirect(url_for('home'))

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    top_k = int(request.form.get('top_k', 5)) 
    
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    
    embeddings_tensor = torch.from_numpy(embeddings).to(device)
    similarities = torch.cosine_similarity(text_features, embeddings_tensor, dim=1)
    
    sorted_indices = similarities.cpu().argsort(descending=True)
    top_k_indices = sorted_indices[:top_k]
    results = [(image_paths[i], image_captions[i], similarities[i].item()) for i in top_k_indices]
    
    return render_template('results.html', results=results, query=query)
@app.route('/gallery')
def gallery():
    images_with_captions = list(zip(image_paths, image_captions))
    return render_template('gallery.html', images=images_with_captions)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = captioning_processor(images=image, return_tensors="pt").to(device)
    outputs = captioning_model.generate(**inputs)
    caption = captioning_processor.decode(outputs[0], skip_special_tokens=True)
    return caption

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)