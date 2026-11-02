import pandas as pd
import math

# 1. Load the locked static calibration data
df = pd.read_csv("calibration_pose.csv")

# 2. Define the subject's known height and the dimensions of the calibration photo
REAL_HEIGHT_MM = 1800.0

# IMPORTANT: You must replace these with the actual pixel dimensions of the image you uploaded!
IMAGE_WIDTH = 1920  
IMAGE_HEIGHT = 1080 

def get_pixel_coords(joint_id):
    row = df[df['Joint_ID'] == joint_id].iloc[0]
    # Multiply the 0.0-1.0 ratio by the image dimensions to get true pixels
    pixel_x = float(row['X']) * IMAGE_WIDTH
    pixel_y = float(row['Y']) * IMAGE_HEIGHT
    return pixel_x, pixel_y

def calculate_2d_length_mm(joint1, joint2, scale_factor):
    x1, y1 = get_pixel_coords(joint1)
    x2, y2 = get_pixel_coords(joint2)
    
    # Strict 2D Pythagorean theorem (ignoring noisy Z-depth for a flat, straight-on photo)
    pixel_dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return pixel_dist * scale_factor

# 3. Calculate the digital height in pixels (Nose [0] to Left Heel [29])
digital_height_pixels = calculate_2d_length_mm(0, 29, 1.0) 
scale = REAL_HEIGHT_MM / digital_height_pixels

# 4. Calculate final physical segment lengths in mm
upper_arm_mm = calculate_2d_length_mm(11, 13, scale) # Left Shoulder to Left Elbow
forearm_mm = calculate_2d_length_mm(13, 15, scale)   # Left Elbow to Left Wrist
thigh_mm = calculate_2d_length_mm(23, 25, scale)     # Left Hip to Left Knee

print(f"Upper Arm Length: {upper_arm_mm:.1f} mm")
print(f"Forearm Length: {forearm_mm:.1f} mm")
print(f"Thigh Length: {thigh_mm:.1f} mm")