import os
import requests

folder_path = "ECG_Dataset/Unrecognized_Scan"
os.makedirs(folder_path, exist_ok=True)

print("Starting automatic download of 50 random images...")

for i in range(1, 101):
    try:
        
        url = "https://picsum.photos/224/224"
        response = requests.get(url)
        
        image_name = f"{folder_path}/random_noise_{i}.jpg"
        with open(image_name, "wb") as file:
            file.write(response.content)
            
        print(f"✅ Downloaded image {i}/100")
    except Exception as e:
        print(f"❌ Failed to download image {i}: {e}")

print("\nSuccess! random images saved in Unrecognized_Scan .")