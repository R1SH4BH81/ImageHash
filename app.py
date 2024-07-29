import os
import numpy as np
from PIL import Image
from math import cos, pi
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.secret_key = "3f4e7b9d8a2c45f1b93d2c4b5d8f7a6e" 

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Helper function to perform the Discrete Cosine Transform (DCT)
def dct_2d(image):
    h, w = image.shape
    dct_matrix = np.zeros((h, w))
    for u in range(h):
        for v in range(w):
            sum_val = 0.0
            for i in range(h):
                for j in range(w):
                    sum_val += image[i, j] * cos(pi * u * (2 * i + 1) / (2 * h)) * cos(pi * v * (2 * j + 1) / (2 * w))
            c_u = 1.0 if u != 0 else 1 / np.sqrt(2)
            c_v = 1.0 if v != 0 else 1 / np.sqrt(2)
            dct_matrix[u, v] = sum_val * c_u * c_v / np.sqrt(h * w)
    return dct_matrix

# Helper function to perform the Inverse Discrete Cosine Transform (IDCT)
def idct_2d(dct_matrix):
    h, w = dct_matrix.shape
    idct_matrix = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            sum_val = 0.0
            for u in range(h):
                for v in range(w):
                    c_u = 1.0 if u != 0 else 1 / np.sqrt(2)
                    c_v = 1.0 if v != 0 else 1 / np.sqrt(2)
                    sum_val += c_u * c_v * dct_matrix[u, v] * cos(pi * u * (2 * i + 1) / (2 * h)) * cos(pi * v * (2 * j + 1) / (2 * w))
            idct_matrix[i, j] = sum_val / np.sqrt(h * w)
    return idct_matrix

# Function to encode an image into a BlurHash-like string
def encode_image(image_path, components_x=4, components_y=4):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((components_x, components_y))
    pixels = np.array(image) / 255.0  # Normalize pixel values

    # Compute DCT for each channel
    r_dct = dct_2d(pixels[:,:,0])
    g_dct = dct_2d(pixels[:,:,1])
    b_dct = dct_2d(pixels[:,:,2])

    # Quantize the DCT coefficients
    r_quant = np.round(r_dct * 100).astype(int)
    g_quant = np.round(g_dct * 100).astype(int)
    b_quant = np.round(b_dct * 100).astype(int)

    # Ensure no invalid or empty values by filling gaps with zeros
    r_quant = np.nan_to_num(r_quant)
    g_quant = np.nan_to_num(g_quant)
    b_quant = np.nan_to_num(b_quant)

    # Encode the quantized coefficients into a string
    hash_string = f'{components_x}x{components_y}-' + \
                  '-'.join([str(v) for v in r_quant.flatten()]) + '-' + \
                  '-'.join([str(v) for v in g_quant.flatten()]) + '-' + \
                  '-'.join([str(v) for v in b_quant.flatten()])

    return hash_string

# Function to decode a BlurHash-like string into an image
def decode_hash(hash_string, width, height):
    parts = hash_string.split('-')
    components_x, components_y = map(int, parts[0].split('x'))
    
    expected_size = components_x * components_y

    # Debugging outputs
    print(f"Hash parts: {parts}")
    print(f"Expected components: {components_x}x{components_y}")
    print(f"Expected size per component: {expected_size}")

    # Validate if there are enough components in the hash
    if len(parts) < 1 + 3 * expected_size:
        raise ValueError("Invalid hash: missing or incomplete components")

    # Parse the quantized values from the hash string
    r_values_str = parts[1:1 + expected_size]
    g_values_str = parts[1 + expected_size:1 + 2 * expected_size]
    b_values_str = parts[1 + 2 * expected_size:1 + 3 * expected_size]

    # Debugging outputs for parsed strings
    print(f"R values string: {r_values_str}")
    print(f"G values string: {g_values_str}")
    print(f"B values string: {b_values_str}")
    print(f"Actual size R: {len(r_values_str)}, G: {len(g_values_str)}, B: {len(b_values_str)}")

    # Ensure the strings have correct numbers of elements
    if len(r_values_str) != expected_size or len(g_values_str) != expected_size or len(b_values_str) != expected_size:
        raise ValueError("Invalid hash: incorrect number of coefficients")

    # Convert to integers safely, skip any empty strings
    try:
        r_values = np.array([int(x) if x != '' else 0 for x in r_values_str]).reshape((components_y, components_x))
        g_values = np.array([int(x) if x != '' else 0 for x in g_values_str]).reshape((components_y, components_x))
        b_values = np.array([int(x) if x != '' else 0 for x in b_values_str]).reshape((components_y, components_x))
    except ValueError as e:
        print("Error parsing values:", e)
        raise ValueError("Invalid hash: unable to parse coefficients") from e

    # Dequantize the values
    r_dequant = r_values / 100.0
    g_dequant = g_values / 100.0
    b_dequant = b_values / 100.0

    # Compute IDCT for each channel
    r_idct = idct_2d(r_dequant)
    g_idct = idct_2d(g_dequant)
    b_idct = idct_2d(b_dequant)

    # Clip values to valid range and convert to 0-255
    r_channel = np.clip(r_idct * 255.0, 0, 255).astype(np.uint8)
    g_channel = np.clip(g_idct * 255.0, 0, 255).astype(np.uint8)
    b_channel = np.clip(b_idct * 255.0, 0, 255).astype(np.uint8)

    # Combine the channels into an image
    image = np.stack((r_channel, g_channel, b_channel), axis=-1)

    # Resize the image to the target size
    decoded_image = Image.fromarray(image).resize((width, height), Image.BICUBIC)

    return decoded_image

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image_to_hash' in request.form:
            # Image to Hash operation
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Generate the hash
                try:
                    blurhash = encode_image(file_path)
                    flash(f'Generated BlurHash: {blurhash}')
                except Exception as e:
                    flash(f'Error processing image: {e}')
                return redirect(request.url)

        elif 'hash_to_image' in request.form:
            # Hash to Image operation
            blurhash = request.form['blurhash']
            width = int(request.form['width'])
            height = int(request.form['height'])

            try:
                decoded_image = decode_hash(blurhash, width, height)
                decoded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'decoded_image.png')
                decoded_image.save(decoded_image_path)
                flash('Image generated from hash successfully!')
                return render_template('index.html', decoded_image_url=url_for('static', filename='uploads/decoded_image.png'))
            except Exception as e:
                flash(f'Error decoding hash: {e}')
                return redirect(request.url)

    return render_template('index.html', decoded_image_url=None)

# Route to display uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Create the upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
