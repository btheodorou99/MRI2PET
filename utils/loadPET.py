import pydicom
import numpy as np
from PIL import Image

def dicom_to_jpeg(dicom_file_path, jpeg_file_path):
    # Load the DICOM file
    ds = pydicom.dcmread(dicom_file_path)

    # Convert the pixel array to a numpy array
    pixel_array_numpy = ds.pixel_array

    # Normalize the image to the range [0,255]
    pixel_array_numpy = pixel_array_numpy - np.min(pixel_array_numpy)
    if np.max(pixel_array_numpy) != 0:
        pixel_array_numpy = pixel_array_numpy / np.max(pixel_array_numpy)
    pixel_array_numpy = (pixel_array_numpy * 255).astype(np.uint8)

    # Convert to PIL Image and save as JPEG
    image = Image.fromarray(pixel_array_numpy)
    image.save(jpeg_file_path)

# Example usage
dicom_file_path = './data/test.dcm'
jpeg_file_path = './data/test.jpg'
dicom_to_jpeg(dicom_file_path, jpeg_file_path)