import os
import shutil

source_folder = '/Users/peteksener/Desktop/vision final project/VeRi/image_test'
target_folder = 'veri_data/gallery'

os.makedirs(target_folder, exist_ok=True)

for filename in os.listdir(source_folder):
    if filename.endswith('.jpg'):
        vehicle_id = filename.split('_')[0]  
        vehicle_folder = os.path.join(target_folder, vehicle_id)
        os.makedirs(vehicle_folder, exist_ok=True)
        shutil.copy(os.path.join(source_folder, filename),
                    os.path.join(vehicle_folder, filename))

print("✅ Done organizing images by vehicle ID!")