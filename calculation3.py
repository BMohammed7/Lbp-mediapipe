import pandas as pd
import math

df = pd.read_csv("calibration_pose (1).csv")
REAL_HEIGHT_MM = 1800.0

# Replace with your actual image dimensions
IMAGE_WIDTH = 1920  
IMAGE_HEIGHT = 1080 

def get_pixel_coords(joint_id):
    row = df[df['Joint_ID'] == joint_id].iloc[0]
    return float(row['X']) * IMAGE_WIDTH, float(row['Y']) * IMAGE_HEIGHT

def calc_mm(joint1, joint2, scale):
    x1, y1 = get_pixel_coords(joint1)
    x2, y2 = get_pixel_coords(joint2)
    pixel_dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return pixel_dist * scale

# --- NEW: Custom Spine Calculation ---
def calc_spine_mm(scale):
    l_sh_x, l_sh_y = get_pixel_coords(11)
    r_sh_x, r_sh_y = get_pixel_coords(12)
    l_hip_x, l_hip_y = get_pixel_coords(23)
    r_hip_x, r_hip_y = get_pixel_coords(24)
    
    # Find the exact pixel midpoints
    mid_sh_x, mid_sh_y = (l_sh_x + r_sh_x) / 2, (l_sh_y + r_sh_y) / 2
    mid_hip_x, mid_hip_y = (l_hip_x + r_hip_x) / 2, (l_hip_y + r_hip_y) / 2
    
    # Calculate 2D pixel distance between those midpoints
    pixel_dist = math.sqrt((mid_hip_x - mid_sh_x)**2 + (mid_hip_y - mid_sh_y)**2)
    return pixel_dist * scale
# --------------------------------------

digital_height = calc_mm(0, 29, 1.0) 
scale = REAL_HEIGHT_MM / digital_height

segments = [
    ("Left Upper Arm", 11, 13),
    ("Right Upper Arm", 12, 14),
    ("Left Forearm", 13, 15),
    ("Right Forearm", 14, 16),
    ("Left Torso Length", 11, 23),
    ("Right Torso Length", 12, 24),
    ("Shoulder Width", 11, 12),
    ("Hip Width", 23, 24),
    ("Left Thigh", 23, 25),
    ("Right Thigh", 24, 26),
    ("Left Calf", 25, 27),
    ("Right Calf", 26, 28)
]

print(f"Calibration Scale Factor: {scale:.4f} mm per pixel\n")

# Print standard segments
for name, j1, j2 in segments:
    length = calc_mm(j1, j2, scale)
    print(f"Joint {j1} to {j2} ({name}): {length:.1f} mm")

# Print the new custom spine segment
spine_length = calc_spine_mm(scale)
print(f"\n---> Virtual Spinal Cord Length: {spine_length:.1f} mm <---")