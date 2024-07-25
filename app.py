import numpy as np
import cv2

# Constants for Base83 encoding
BASE83_CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~"


def encode_image_to_string(image_path, components_x=4, components_y=3):
    """
    Encodes an image into a compact string representation using a DCT-based method.

    Args:
        image_path (str): The path to the image file.
        components_x (int): The number of horizontal components.
        components_y (int): The number of vertical components.

    Returns:
        str: The encoded string representation of the image.
    """
    # Load the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    height, width, _ = image.shape

    # Normalize the image to a range of [0, 1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

    # Initialize frequency components
    frequency_components = []

    # Calculate average color
    avg_color = np.mean(image, axis=(0, 1))
    frequency_components.append(avg_color)

    # Apply DCT to capture frequency components
    for y in range(components_y):
        for x in range(components_x):
            if x == 0 and y == 0:
                continue
            dct_value = apply_dct(image, x, y, width, height)
            frequency_components.append(dct_value)

    # Encode frequency components into a Base83 string
    encoded_string = encode_base83(frequency_components, components_x,
                                   components_y)

    return encoded_string


def apply_dct(image, x, y, width, height):
    """
    Applies Discrete Cosine Transform (DCT) to extract frequency components.

    Args:
        image (np.ndarray): The image data.
        x (int): Horizontal component index.
        y (int): Vertical component index.
        width (int): Image width.
        height (int): Image height.

    Returns:
        np.ndarray: The frequency component for the given indices.
    """
    # Calculate the cosine basis functions
    cosines = np.cos(np.pi * np.arange(width) * x / width)[:, None] * \
              np.cos(np.pi * np.arange(height) * y / height)[None, :]

    # Scale factor
    scale = np.sqrt(
        (2 if x != 0 else 1) * (2 if y != 0 else 1) / (width * height))

    # Compute the DCT value
    image_y = image[:, :, 0]  # Red channel
    image_y = image_y - np.mean(image_y)  # Zero-mean
    dct_value = np.sum(image_y * cosines) * scale

    # Assuming the DCT is applied to the Y channel only, return the result
    return np.array([dct_value, dct_value,
                     dct_value])  # Return same value for RGB channels


def encode_base83(frequency_components, components_x, components_y):
    """
    Encodes frequency components into a Base83 string representation.

    Args:
        frequency_components (list): The list of frequency components.
        components_x (int): The number of horizontal components.
        components_y (int): The number of vertical components.

    Returns:
        str: The encoded Base83 string.
    """
    # Encode component dimensions and quantized AC components
    ac_encoded = encode_ac(frequency_components[1:], components_x,
                           components_y)

    # Encode DC component
    dc_encoded = encode_dc(frequency_components[0])

    return dc_encoded + ac_encoded


def encode_dc(dc):
    """
    Encodes the DC component of the frequency data into a Base83 string.

    Args:
        dc (np.ndarray): The DC component.

    Returns:
        str: The encoded DC component as a Base83 string.
    """
    dc_value = (int(dc[0] * 255) << 16) + (int(dc[1] * 255) << 8) + int(
        dc[2] * 255)
    return encode_base83_value(dc_value, 4)


def encode_ac(ac_components, components_x, components_y):
    """
    Encodes AC components of the frequency data into a Base83 string.

    Args:
        ac_components (list): The AC components.
        components_x (int): The number of horizontal components.
        components_y (int): The number of vertical components.

    Returns:
        str: The encoded AC components as a Base83 string.
    """
    max_ac = max(np.linalg.norm(ac, ord=2) for ac in ac_components)
    quantized_max_ac = min(82, max(0, int(max_ac * 166 - 0.5)))
    ac_encoded = encode_base83_value(quantized_max_ac, 1)

    for ac in ac_components:
        if quantized_max_ac > 0:
            quantized_ac = (ac / max_ac) * 9
            quantized_ac = np.clip(quantized_ac, -9, 9)
            ac_value = ((int(quantized_ac[0] * 9 + 9.5) * 19 * 19) +
                        (int(quantized_ac[1] * 9 + 9.5) * 19) +
                        int(quantized_ac[2] * 9 + 9.5))
            ac_encoded += encode_base83_value(ac_value, 2)

    return ac_encoded


def encode_base83_value(value, length):
    """
    Encodes an integer value into a Base83 string of a given length.

    Args:
        value (int): The value to encode.
        length (int): The length of the Base83 string.

    Returns:
        str: The encoded Base83 string.
    """
    base83 = ''
    for _ in range(length):
        base83 = BASE83_CHARACTERS[value % 83] + base83
        value //= 83
    return base83


def decode_string_to_image(encoded_string,
                           width,
                           height,
                           components_x=4,
                           components_y=3):
    """
    Decodes a compact string representation back into an image.

    Args:
        encoded_string (str): The encoded string representation of the image.
        width (int): The width of the output image.
        height (int): The height of the output image.
        components_x (int): The number of horizontal components.
        components_y (int): The number of vertical components.

    Returns:
        np.ndarray: The decoded image data.
    """
    # Decode components from Base83 string
    frequency_components = decode_base83(encoded_string, components_x,
                                         components_y)

    # Reconstruct the image using IDCT
    image = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            color = frequency_components[0]
            for j in range(components_y):
                for i in range(components_x):
                    if i == 0 and j == 0:
                        continue
                    basis = np.cos(np.pi * x * i / width) * np.cos(
                        np.pi * y * j / height)
                    color += frequency_components[j * components_x + i] * basis
            image[y, x, :] = np.clip(color, 0, 1)

    # Convert image back to uint8
    image = (image * 255).astype(np.uint8)

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def decode_base83(encoded_string, components_x, components_y):
    """
    Decodes frequency components from a Base83 string.

    Args:
        encoded_string (str): The encoded Base83 string.
        components_x (int): The number of horizontal components.
        components_y (int): The number of vertical components.

    Returns:
        list: The decoded frequency components.
    """
    # Decode DC component
    dc_value = decode_base83_value(encoded_string[:4])
    dc_component = np.array([((dc_value >> 16) & 255) / 255.0,
                             ((dc_value >> 8) & 255) / 255.0,
                             (dc_value & 255) / 255.0])

    # Decode AC components
    max_ac = (decode_base83_value(encoded_string[4:5]) + 1) / 166
    ac_components = []

    for i in range(components_x * components_y - 1):
        ac_value = decode_base83_value(encoded_string[5 + 2 * i:7 + 2 * i])
        quantized_ac = np.array([(ac_value // (19 * 19)) - 9,
                                 ((ac_value // 19) % 19) - 9,
                                 (ac_value % 19) - 9])
        ac_components.append((quantized_ac / 9) * max_ac)

    return [dc_component] + ac_components


def decode_base83_value(base83_str):
    """
    Decodes a Base83 string into an integer value.

    Args:
        base83_str (str): The Base83 string.

    Returns:
        int: The decoded integer value.
    """
    value = 0
    for char in base83_str:
        value = value * 83 + BASE83_CHARACTERS.index(char)
    return value


# Example usage
if __name__ == "__main__":
    # Specify the path to the image you want to encode
    image_path = "example.jpg"

    # Encode the image into a string representation
    encoded_string = encode_image_to_string(image_path)

    # Decode the string back into an image
    decoded_image = decode_string_to_image(encoded_string,
                                           width=128,
                                           height=128)

    # Save the decoded image
    cv2.imwrite("decoded_image.jpg", decoded_image)

    print("Encoding and decoding completed successfully.")
