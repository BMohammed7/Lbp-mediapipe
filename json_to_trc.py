import json
import glob

json_files = sorted(glob.glob("output_data/*.json"))
if not json_files:
    print("No JSON files found in output_data/")
    exit()

fps = 30.0  
num_markers = 37
output_file = "jumping_jacks_rajagopal.trc"

# Mapped precisely to the RajagopalLaiUhlrich2023 marker set
marker_names = [
    "Nose", "LEye", "REye", "LEar", "REar",               
    "LACR", "RACR", "LLEL", "RLEL", "LFAradius", "RFAradius", 
    "LHJC", "RHJC", "LKJC", "RKJC", "LAJC", "RAJC",   
    "C7", "T3", "T7", "T12", "L1", "L3", "L5", "Pelvis", "Sacrum", 
    "CLAV", "RASH", "LASH", "RPSH", "LPSH",               
    "Head", "Neck", "Jaw", "MidBack", "LCAL", "RCAL"    
]

with open(output_file, "w") as f:
    f.write("PathFileType\t4\t(X/Y/Z)\tjumping_jacks_rajagopal.trc\n")
    f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
    f.write(f"{fps}\t{fps}\t{len(json_files)}\t{num_markers}\tm\t{fps}\t1\t{len(json_files)}\n")
    
    row4 = ["Frame#", "Time"]
    for name in marker_names:
        row4.extend([name, "", ""]) 
    f.write("\t".join(row4) + "\n")
    
    row5 = ["", ""]
    for i in range(num_markers):
        row5.extend([f"X{i+1}", f"Y{i+1}", f"Z{i+1}"])
    f.write("\t".join(row5) + "\n\n")

    for frame_idx, j_file in enumerate(json_files):
        with open(j_file, 'r') as j:
            data = json.load(j)
        
        time = frame_idx / fps
        row = [str(frame_idx + 1), f"{time:.4f}"]
        
        if len(data['people']) > 0:
            kpts = data['people'][0]['pose_keypoints_3d']
            chunk = 4 if len(kpts) == num_markers * 4 else 3
            
            for i in range(num_markers):
                x = kpts[i*chunk]
                y = kpts[i*chunk + 1]
                z = kpts[i*chunk + 2]
                
                # REVERTED to right-side up, but LIFTED 1 meter out of the floor
                os_x = x
                os_y = y + 1.0  
                os_z = z   
                
                row.extend([f"{os_x:.4f}", f"{os_y:.4f}", f"{os_z:.4f}"])
        else:
            row.extend(["0.0000"] * (num_markers * 3))
            
        f.write("\t".join(row) + "\n")

print(f"Success! Created {output_file} - Floor offset applied.")