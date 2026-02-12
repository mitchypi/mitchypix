import os
import json
import re
from datetime import datetime
from string import Template
from flask import Flask, request, render_template, render_template_string, redirect, url_for, jsonify, flash, abort
import torch
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
import time
from tqdm import tqdm

app = Flask(__name__)
app.secret_key = 'hi'

ITEMS_PER_PAGE = 50
BLOG_POSTS_FILE = 'blog_posts.json'
SLUG_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
DEFAULT_BLOG_POSTS = [
    {"slug": "09292025", "title": "09292025", "publish_date": "2025-09-29"},
    {"slug": "09112025", "title": "09112025", "publish_date": "2025-09-11"},
    {"slug": "08262025", "title": "08262025", "publish_date": "2025-08-26"},
]
ABOUT_CONTENT_FILE = 'about_content.json'
DEFAULT_ABOUT_CONTENT = """<h1>About</h1>
    
    <p><a href="/" class="button">Back to Home</a></p>
    
    <p>Movies I like:</p>
    <ul>
        <li>Yi Yi</li>
        <li>Robot Dreams</li>
        <li>Nebraska</li>
        <li>Blade Runner 2049</li>
        <li>Donnie Darko</li>
        <li>Everything Everywhere all at once</li>
        <li>Apocalypse Now</li>
        <li>Stalingrad</li>
        <li>Leviathan</li>
        <li>Full Metal Jacket</li>
        <li>City of God</li>
    </ul>
    
    <p>Games I like:</p>
    <ul>
        <li>Halo Reach</li>
        <li>Life is Strange</li>
        <li>Deus Ex</li>
        <li>SOMA</li>
        <li>jstris</li>
        <li>Spec Ops the Line</li>
        <li>Knights of the Old Republic</li>
        <li>Fallout New Vegas</li>
    </ul>
    
    <p>Music I like:</p>
    <ul>
        <li>Jadu Heart</li>
        <li>Superheaven</li>
        <li>Slow Pulp</li>
        <li>Deftones</li>
        <li>Brian Jonestown Massacre</li>
        <li>Yung Hurn</li>
        <li>Split Chain</li>
        <li>Valium Aggelein (Duster)</li>
        <li>Santigold</li>
        <li>Scarlet House</li>
        <li>The Black Angels</li>
        <li>Arauchi Yu</li>
        <li>Mirele</li>
        <li>Chevelle</li>
        <li>Dean Blunt</li>
        <li>Chinese Cigarettes</li>
        <li>Yung Lean</li>
        <li>Mogwai</li>
        <li>Whirr</li>
        <li>Duster</li>
        <li>sniper2004</li>
        <li>Black Kray</li>
        <li>Bar Italia</li>
        <li>Eiafuawn</li>
        <li>Julie</li>
        <li>Bassvictim</li>
        
        <li><a href="https://www.last.fm/user/pimpledkey2200" class="button">go to my last.fm</a></li>
    </ul>
    
    <p>Books:</p>
    <ul>
        <li>All Max Frisch books</li>
        <li>Colm Toibin</li>
        <li>Dispatches</li>
        <li>/u/befriendjamin reddit comments</li>
    </ul>
    
    <p>Mitch likes bar trivia</p>"""
BLOG_HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>$title</title>
    <style>
        body {
            font-size: 100%;
            background-color: beige;
            font-family: monospace;
            text-align: center;
        }
        html {
            -webkit-text-size-adjust: 100%;
        }
        table, td, tr{
            border: 1px solid black;
            border-collapse: collapse;
            font-family: monospace;
            text-align: center;
        }
        .wn{
            text-align: left;
        }
        a { 
            text-decoration: none;
            text-align: center;
        }
        .button {
            font-family: monospace;
            padding: 8px 16px;
            border: 1px solid black;
            background-color: white;
            cursor: pointer;
            display: inline-block;
            margin: 5px;
        }
        .button:hover {
            background-color: #f0f0f0;
        }
        ul {
            display: inline-block;
            text-align: left;
            vertical-align: top;
        }
    </style>
</head>
<body>
    <p>$display_date</p>
    
    <p><a href="/" class="button">Back to Home</a></p>

    $content_html
    
{% with messages = get_flashed_messages() %}
  {% if messages %}
    <ul class="flashes">
    {% for message in messages %}
      <li>{{ message }}</li>
    {% endfor %}
    </ul>
  {% endif %}
{% endwith %}
</body>
</html>
""")


def ensure_blog_posts_file():
    if not os.path.exists(BLOG_POSTS_FILE):
        save_blog_posts(DEFAULT_BLOG_POSTS)


def ensure_about_content_file():
    if not os.path.exists(ABOUT_CONTENT_FILE):
        save_about_content(DEFAULT_ABOUT_CONTENT)


def load_blog_posts():
    ensure_blog_posts_file()
    try:
        with open(BLOG_POSTS_FILE, 'r', encoding='utf-8') as handle:
            posts = json.load(handle)
            return posts if isinstance(posts, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_blog_posts(posts):
    with open(BLOG_POSTS_FILE, 'w', encoding='utf-8') as handle:
        json.dump(posts, handle, indent=2)


def load_about_content():
    ensure_about_content_file()
    try:
        with open(ABOUT_CONTENT_FILE, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
            if isinstance(payload, dict):
                content = payload.get('content')
                if isinstance(content, str):
                    return content
    except (json.JSONDecodeError, OSError):
        pass
    return DEFAULT_ABOUT_CONTENT


def save_about_content(content):
    with open(ABOUT_CONTENT_FILE, 'w', encoding='utf-8') as handle:
        json.dump({"content": content}, handle, indent=2)


def sort_blog_posts(posts):
    return sorted(
        posts,
        key=lambda post: post.get('publish_date', ''),
        reverse=True,
    )


def normalize_publish_date(slug, publish_date):
    if publish_date:
        try:
            dt = datetime.strptime(publish_date, "%Y-%m-%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None
    if len(slug) == 8 and slug.isdigit():
        try:
            dt = datetime.strptime(slug, "%m%d%Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None
    return None


def format_display_date(publish_date, fallback):
    if publish_date:
        try:
            dt = datetime.strptime(publish_date, "%Y-%m-%d")
            return dt.strftime("%m/%d/%Y")
        except ValueError:
            pass
    if len(fallback) == 8 and fallback.isdigit():
        return f"{fallback[:2]}/{fallback[2:4]}/{fallback[4:]}"
    return fallback


def convert_content_to_html(raw_content):
    paragraphs = [line.strip() for line in raw_content.splitlines() if line.strip()]
    if not paragraphs:
        return "<p></p>"
    return "\n    ".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)


def is_local_request():
    host = request.environ.get('HTTP_HOST', '') or ''
    remote_addr = (request.remote_addr or '').strip()
    forwarded_for_hdr = request.headers.get('X-Forwarded-For', '').strip()
    forwarded_for = forwarded_for_hdr.split(',')[0].strip() if forwarded_for_hdr else ''
    local_hosts = ('127.0.0.1', 'localhost', '0.0.0.0')
    return (
        any(host.startswith(h) for h in local_hosts)
        or remote_addr in ('127.0.0.1', '::1')
        or forwarded_for in ('127.0.0.1', '::1')
    )
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
    blog_posts = sort_blog_posts(load_blog_posts())
    about_content = load_about_content()
    return render_template('home.html', blog_posts=blog_posts, about_content_raw=about_content)

@app.route('/about')
def about():
    about_content = load_about_content()
    return render_template('about.html', about_content=about_content)


@app.route('/about/edit')
def edit_about():
    if not is_local_request():
        flash('About editing is only available from localhost.')
        return redirect(url_for('about'))
    content = load_about_content()
    return render_template('about_edit.html', content=content)

@app.route('/blog/<slug>')
def serve_blog_post(slug):
    blog_posts = load_blog_posts()
    valid_slugs = {post['slug'] for post in blog_posts}
    if slug not in valid_slugs:
        abort(404)
    template_name = f'{slug}.html'
    template_path = os.path.join(
        app.root_path,
        app.template_folder or 'templates',
        template_name,
    )
    if not os.path.exists(template_path):
        abort(404)
    # Inject a small edit link when local so user can edit the post
    if is_local_request():
        try:
            with open(template_path, 'r', encoding='utf-8') as fh:
                html = fh.read()
            edit_url = url_for('edit_blog_post', slug=slug)
            delete_url = url_for('delete_blog_post', slug=slug)
            injection = (
                "\n<div style=\"position:fixed;bottom:10px;right:10px;z-index:9999;display:flex;gap:8px;align-items:center;\">"
                f"\n  <a href=\"{edit_url}\" class=\"button\">Edit This Post</a>"
                f"\n  <form action=\"{delete_url}\" method=\"post\" style=\"margin:0;\" onsubmit=\"return confirm('Delete this blog post? This cannot be undone.');\">"
                "\n    <button type=\"submit\" class=\"button\">Delete This Post</button>"
                "\n  </form>"
                "\n</div>\n"
            )
            if '</body>' in html:
                html = html.replace('</body>', injection + '</body>')
            else:
                html = html + injection
            return render_template_string(html)
        except Exception:
            # Fallback to normal rendering if injection fails
            return render_template(template_name)
    return render_template(template_name)


@app.route('/blog/<slug>/edit')
def edit_blog_post(slug):
    if not is_local_request():
        flash('Blog editing is only available from localhost.')
        return redirect(url_for('serve_blog_post', slug=slug))

    blog_posts = load_blog_posts()
    valid_slugs = {post['slug'] for post in blog_posts}
    if slug not in valid_slugs:
        abort(404)
    template_name = f'{slug}.html'
    template_path = os.path.join(
        app.root_path,
        app.template_folder or 'templates',
        template_name,
    )
    if not os.path.exists(template_path):
        abort(404)

    with open(template_path, 'r', encoding='utf-8') as fh:
        content = fh.read()
    return render_template('blog_edit.html', slug=slug, content=content)


@app.route('/blog/<slug>/update', methods=['POST'])
def update_blog_post(slug):
    if not is_local_request():
        flash('Blog editing is only available from localhost.')
        return redirect(url_for('serve_blog_post', slug=slug))

    blog_posts = load_blog_posts()
    valid_slugs = {post['slug'] for post in blog_posts}
    if slug not in valid_slugs:
        abort(404)
    template_name = f'{slug}.html'
    template_path = os.path.join(
        app.root_path,
        app.template_folder or 'templates',
        template_name,
    )
    if not os.path.exists(template_path):
        abort(404)

    new_content = request.form.get('content', '')
    if not new_content.strip():
        flash('Content cannot be empty.')
        return redirect(url_for('edit_blog_post', slug=slug))
    normalized_content = new_content.replace('\r\n', '\n').replace('\r', '\n')
    try:
        with open(template_path, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(normalized_content)
        flash('Blog post updated.')
    except OSError as exc:
        flash(f'Failed to update blog post: {exc}')
        return redirect(url_for('edit_blog_post', slug=slug))
    return redirect(url_for('serve_blog_post', slug=slug))


@app.route('/blog/<slug>/delete', methods=['POST'])
def delete_blog_post(slug):
    if not is_local_request():
        flash('Blog deletion is only available from localhost.')
        return redirect(url_for('serve_blog_post', slug=slug))

    blog_posts = load_blog_posts()
    updated_posts = [post for post in blog_posts if post.get('slug') != slug]
    if len(updated_posts) == len(blog_posts):
        abort(404)

    template_name = f'{slug}.html'
    template_path = os.path.join(
        app.root_path,
        app.template_folder or 'templates',
        template_name,
    )

    try:
        if os.path.exists(template_path):
            os.remove(template_path)
        save_blog_posts(sort_blog_posts(updated_posts))
        flash(f'Deleted blog post "{slug}".')
    except OSError as exc:
        flash(f'Failed to delete blog post: {exc}')
        return redirect(url_for('serve_blog_post', slug=slug))
    return redirect(url_for('home'))


@app.route('/blog/create', methods=['POST'])
def create_blog_post():
    if not is_local_request():
        flash('Blog post creation is only available from localhost.')
        return redirect(url_for('home'))

    slug = request.form.get('slug', '').strip()
    title = request.form.get('title', '').strip()
    publish_date = request.form.get('publish_date', '').strip()
    raw_content = request.form.get('content', '').strip()

    if not slug:
        flash('Slug is required.')
        return redirect(url_for('home'))
    if not SLUG_PATTERN.match(slug):
        flash('Slug may only contain letters, numbers, hyphens, and underscores.')
        return redirect(url_for('home'))

    blog_posts = load_blog_posts()
    if any(post['slug'] == slug for post in blog_posts):
        flash(f'A blog post with slug "{slug}" already exists.')
        return redirect(url_for('home'))

    normalized_date = normalize_publish_date(slug, publish_date)
    if publish_date and not normalized_date:
        flash('Publish date must be in YYYY-MM-DD format.')
        return redirect(url_for('home'))

    display_date = format_display_date(normalized_date, slug)
    title = title or slug
    content_html = convert_content_to_html(raw_content)

    template_name = f'{slug}.html'
    template_path = os.path.join(
        app.root_path,
        app.template_folder or 'templates',
        template_name,
    )
    if os.path.exists(template_path):
        flash(f'Template for "{slug}" already exists.')
        return redirect(url_for('home'))

    try:
        rendered = BLOG_HTML_TEMPLATE.substitute(
            title=title,
            display_date=display_date,
            content_html=content_html,
        )
        with open(template_path, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(rendered)
    except OSError as exc:
        flash(f'Failed to write blog post: {exc}')
        return redirect(url_for('home'))

    blog_posts.append(
        {
            "slug": slug,
            "title": title,
            "publish_date": normalized_date or "",
        }
    )
    save_blog_posts(sort_blog_posts(blog_posts))
    flash(f'Created blog post "{title}".')
    return redirect(url_for('serve_blog_post', slug=slug))


@app.route('/about/update', methods=['POST'])
def update_about():
    if not is_local_request():
        flash('About page updates are only available from localhost.')
        return redirect(url_for('home'))

    content = request.form.get('content', '').strip()
    if not content:
        flash('Content is required to update the About page.')
        return redirect(url_for('home'))

    save_about_content(content)
    flash('Updated About page.')
    return redirect(url_for('about'))


@app.route('/upload', methods=['POST'])
def upload():
    global embeddings, image_paths, image_captions
    if 'files' not in request.files:
        return 'No file part'
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return 'No selected file'
    
    for file in tqdm(files, desc="Uploading images"):
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static', 'uploads', filename)
            relative_path = os.path.join('uploads', filename).replace('\\', '/')

            if os.path.exists(file_path) or relative_path in image_paths:
                print(f"Skipping existing image: {filename}")
                continue
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
            image_paths = np.append(image_paths, relative_path)
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
    try:
        top_k = int(request.form.get('top_k', 5)) 
    except ValueError:
        top_k = 5
    
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
    page = request.args.get('page', 1, type=int)
    
    total_images = len(image_paths)
    
    # Reverse the lists
    reversed_image_paths = list(reversed(image_paths))
    reversed_image_captions = list(reversed(image_captions))
    
    start = (page - 1) * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE
    
    images_with_captions = list(zip(reversed_image_paths[start:end], reversed_image_captions[start:end]))
    
    total_pages = (total_images + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    
    # Calculate the range of pages to display
    page_range = range(max(1, page-2), min(total_pages, page+2) + 1)
    
    return render_template('gallery.html', 
                           images=images_with_captions, 
                           page=page, 
                           total_pages=total_pages,
                           total_images=total_images,
                           page_range=page_range)

@app.route('/update_captions', methods=['POST'])
def update_captions():
    # Only allow updates from localhost
    if not (request.environ.get('HTTP_HOST', '').startswith('127.0.0.1') or 
            request.environ.get('HTTP_HOST', '').startswith('localhost')):
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    try:
        global image_captions, image_paths
        
        caption_updates = request.get_json()
        
        if not caption_updates:
            return jsonify({'success': False, 'error': 'No caption data received'})
        
        # Update captions in the array
        for image_path, new_caption in caption_updates.items():
            # Find the index of the image path in the array
            indices = np.where(image_paths == image_path)[0]
            if len(indices) > 0:
                image_captions[indices[0]] = new_caption
        
        # Save updated captions to file
        np.save(CAPTIONS_FILE, image_captions)
        
        return jsonify({'success': True, 'message': f'Updated {len(caption_updates)} captions'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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
    app.run(host='0.0.0.0', port=8080)
