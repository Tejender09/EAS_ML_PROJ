import urllib.request
import os

url = "https://raw.githubusercontent.com/nelson-wu/employee-attrition-ml/master/WA_Fn-UseC_-HR-Employee-Attrition.csv"
filename = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

if not os.path.exists(filename):
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download: {e}")
else:
    print(f"{filename} already exists.")
