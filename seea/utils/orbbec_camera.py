import requests
import cv2
import os
import numpy as np
import time
import json
import zerorpc
from seea.utils.common import detection


def fetch_latest_image(server_url):
    try:
        # Send request to get the latest image
        response = requests.get(server_url, timeout=5)
        response.raise_for_status()  # Check if the request was successful

        # Read image data from the response
        image_data = response.content

        # Convert image data to OpenCV image
        image_np = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        return image
    except requests.exceptions.RequestException as e:
        print(f"Request image from server failed: {e}\n{server_url}")
        return None


def process_images(rgb_bytes, depth_bytes, width=640, height=480, channel=3):
    # Encode the bytes into a NumPy array.
    rgb_image = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(height, width, channel)
    depth_image = np.frombuffer(depth_bytes, dtype=np.float64).reshape(height, width)
    return rgb_image, depth_image


def request_data(camera_client: zerorpc.Client, width=640, height=480, channel=3):
    rgb_bytes, depth_bytes = camera_client.get_latest_images()
    rgb_image, depth_image = process_images(rgb_bytes, depth_bytes, width, height, channel)
    rgb_array = np.asarray(rgb_image)
    depth_array = np.asarray(depth_image)
    return rgb_array, depth_array


def save_data(prefix, rgb_array, depth_array, add_time=True):
    time_str = time.time() if add_time else "latest"
    save_path = f"{prefix}_rgb_image_{time_str}.jpg"
    cv2.imwrite(save_path, rgb_array)
    np.save(f"{prefix}_depth_image_{time_str}.npy", depth_array)
    return save_path


class OrbbecCameraClient:
    def __init__(self, 
                 server_ip, 
                 default_view='front', 
                 default_port='4577', 
                 width=1280, 
                 height=720,
                 save_folder="assets/realtime_images"):
        self.default_view = default_view
        self.default_port = default_port
        self.server_ip = server_ip
        self.width = width
        self.height = height
        self.save_folder = save_folder
        
        os.makedirs(self.save_folder, exist_ok=True)
    
    def get_realtime_imagepath(self, view=None, port=None, add_time=True):
        if view is None:
            view = self.default_view
        if port is None:
            port = self.default_port
            
        front_camera_zerorpc_client = zerorpc.Client()
        front_camera_zerorpc_client.connect(f"tcp://{self.server_ip}:{port}")
        time_suffix = time.time() if add_time else ""
        imagepath = os.path.join(self.save_folder, f"{view}_rgb_image_{time_suffix}.jpg")
        rgb_array, depth_array = request_data(front_camera_zerorpc_client, width=self.width, height=self.height)
        front_camera_zerorpc_client.close()
        cv2.imwrite(imagepath, rgb_array)
        # np.save(f"front_depth_image_{time.time()}.npy", depth_array)
        return imagepath

    def get_realtime_det_imagepath(self, view=None, port=None, visualization=True, label=False, bbox=False):
        """
        Fetches the real-time image path and performs visualization based on the specified options.

        Args:
            visualization (bool): Whether to visualize using SoM (Set-of-Mark).
            label (bool): Whether to include labels in the visualization.
            bbox (bool): Whether to include bounding boxes in the visualization.

        Returns:
            tuple: Visualization image path and corresponding detection results.
        """
        image_path = self.get_realtime_imagepath(view, port)
        
        # Perform detection with visualization enabled
        predictions, vis_image_path = detection(image_path, visualization=visualization)
        # Extract specific components based on label and bbox flags
        if predictions:
            labels = predictions.get('label_names', []) if label else []
            boxes = predictions.get('boxes', []) if bbox else []

            if label and not bbox:
                results = list(set(labels))  # Return unique labels
            elif label and bbox:
                results = [
                    {"label": lbl, "bbox": box} for lbl, box in zip(labels, boxes)
                ]
            else:
                results = []
        else:
            results = []

        predictions = json.dumps(results)
        return vis_image_path, predictions
