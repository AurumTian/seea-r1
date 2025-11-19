import cv2
import base64


def resize_image(img, resize_factor):
    assert resize_factor > 0
    new_width = int(img.shape[1] * resize_factor)
    new_height = int(img.shape[0] * resize_factor)
    new_size = (new_width, new_height)
    print(f"new_size: {new_size}")
    new_img = cv2.resize(img, new_size)
    return new_img


def encode_base64(img, ext='.jpg'):
    # Encode the image into a byte stream (e.g., .jpg format)
    _, encoded_image = cv2.imencode(ext, img)

    # Convert the encoded image byte data to Base64 encoding
    image_bytes = encoded_image.tobytes()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    return image_base64


def to_byte(img, ext='.jpg'):
    # Encode the image into a byte stream
    success, encoded_image = cv2.imencode(ext, img)
    if not success:
        print("Image encoding failed")
        return None
    else:
        # Convert byte data to a binary stream
        return encoded_image.tobytes()
