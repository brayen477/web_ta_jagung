import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model_mobilenet_daun_jagung.h5'
CLASS_NAMES = [
    'Bercak Daun Helminthosporium', 
    'Bercak Daun Mata Ikan', 
    'Daun Bercak Abu-Abu', 
    'Daun Bulai', 
    'Daun Hawar', 
    'Daun Karat', 
    'Daun Sehat'
]

def analyze_last_upload():
    # Find most recently modified file in uploads
    files = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    if not files:
        with open('test_results.txt', 'w') as f:
            f.write("No files found in uploads.")
        return
        
    last_file = max(files, key=os.path.getmtime)
    
    # Analyze it
    try:
        img = image.load_img(last_file, target_size=(100, 100))
        rgb_pixels = np.array(img.convert('RGB'))
        hsv_img = img.convert('HSV')
        pixels = np.array(hsv_img)
        
        global_leaf_pixels = 0
        roi_leaf_pixels = 0
        total_hue_sum = 0
        total_sat_sum = 0
        total_val_sum = 0
        
        roi_start, roi_end = 20, 80
        roi_total_pixels = (roi_end - roi_start) * (roi_end - roi_start)
        total_pixels = 100 * 100
        
        for i in range(100):
            for j in range(100):
                h, s, v = pixels[i, j]
                h, s, v = int(h), int(s), int(v)
                
                if s > 20 and v > 20:
                    if 5 <= h <= 125:
                        global_leaf_pixels += 1
                        total_hue_sum += h
                        total_sat_sum += s
                        total_val_sum += v
                        
                        if roi_start <= i < roi_end and roi_start <= j < roi_end:
                            roi_leaf_pixels += 1
                            
        global_ratio = (global_leaf_pixels / total_pixels) * 100
        roi_ratio = (roi_leaf_pixels / roi_total_pixels) * 100
        mean_hue = total_hue_sum / max(1, global_leaf_pixels)
        mean_sat = total_sat_sum / max(1, global_leaf_pixels)
        mean_val = total_val_sum / max(1, global_leaf_pixels)
        
        # Structure Tensor Coherence
        roi_pixels_rgb = rgb_pixels[roi_start:roi_end, roi_start:roi_end]
        gray_roi = 0.299 * roi_pixels_rgb[:, :, 0] + 0.587 * roi_pixels_rgb[:, :, 1] + 0.114 * roi_pixels_rgb[:, :, 2]
        
        dx = gray_roi[:, 1:] - gray_roi[:, :-1]
        dy = gray_roi[1:, :] - gray_roi[:-1, :]
        
        h_roi, w_roi = gray_roi.shape
        dx_pad = np.zeros((h_roi, w_roi))
        dx_pad[:, :w_roi-1] = dx
        dy_pad = np.zeros((h_roi, w_roi))
        dy_pad[:h_roi-1, :] = dy
        
        Ixx = dx_pad ** 2
        Iyy = dy_pad ** 2
        Ixy = dx_pad * dy_pad
        
        mean_Ixx = np.mean(Ixx)
        mean_Iyy = np.mean(Iyy)
        mean_Ixy = np.mean(Ixy)
        
        trace = mean_Ixx + mean_Iyy
        det = mean_Ixx * mean_Iyy - mean_Ixy ** 2
        discriminant = np.sqrt(max(0.0, trace ** 2 - 4 * det))
        
        lambda_1 = 0.5 * (trace + discriminant)
        lambda_2 = 0.5 * (trace - discriminant)
        
        coherence = (lambda_1 - lambda_2) / (lambda_1 + lambda_2 + 1e-5)
        
        # Load and predict with MobileNetV2
        model = load_model(MODEL_PATH)
        img_pred = image.load_img(last_file, target_size=(224, 224))
        x = image.img_to_array(img_pred)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        classes = model.predict(x)
        predicted_class_index = np.argmax(classes[0])
        predicted_label = CLASS_NAMES[predicted_class_index]
        confidence = np.max(classes[0]) * 100
        
        sorted_probs = np.sort(classes[0])[::-1]
        margin = (sorted_probs[0] - sorted_probs[1]) * 100
        
        # Write results
        with open('test_results.txt', 'w') as f:
            f.write(f"Analyzed File: {last_file}\n")
            f.write(f"Global Ratio: {global_ratio:.2f}%\n")
            f.write(f"ROI Ratio: {roi_ratio:.2f}%\n")
            f.write(f"Mean Hue: {mean_hue:.2f}\n")
            f.write(f"Mean Saturation: {mean_sat:.2f}\n")
            f.write(f"Mean Value: {mean_val:.2f}\n")
            f.write(f"Coherence: {coherence:.4f}\n")
            f.write(f"AI Class: {predicted_label}\n")
            f.write(f"AI Confidence: {confidence:.2f}%\n")
            f.write(f"AI Margin: {margin:.2f}%\n")
            f.write(f"All Probabilities:\n")
            for c, p in zip(CLASS_NAMES, classes[0]):
                f.write(f"  {c}: {p*100:.2f}%\n")
                
    except Exception as e:
        with open('test_results.txt', 'w') as f:
            f.write(f"Error: {e}\n")

if __name__ == '__main__':
    analyze_last_upload()
