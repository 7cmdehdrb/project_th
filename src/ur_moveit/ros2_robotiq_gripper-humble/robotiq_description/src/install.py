import os
import requests
import tqdm

# >>> Download meshes.zip >>>

install_path = os.path.join(os.path.dirname(__file__), "..", "meshes")

# Download object model file
server_url = "http://7cmdehdrb.iptime.org:13670/api/public/dl/woKjVGJe"
filename = "meshes.zip"

local_filepath = os.path.join(install_path, filename)

# Check if the folder already exists
if not os.path.exists(install_path):
    os.makedirs(install_path)
    print(f"Created a folder {install_path}")

# 파일이 이미 존재하는지 확인
if len(os.listdir(install_path)) > 0:
    print(f"Files or folders already exist in {install_path}. Stop downloading.")

else:
    print(f"Downloading {filename} from {server_url} ...")
    response: requests.models.Response = requests.get(server_url, stream=True)

    if response.status_code == 200:
        with open(local_filepath, "wb") as f:
            total_size = int(response.headers.get("content-length", 0))
            with tqdm.tqdm(
                total=total_size, unit="B", unit_scale=True, desc=filename
            ) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"Downloaded {filename} to {local_filepath}")

        # 압축 파일이면 풀기 (옵션)
        if filename.endswith(".zip"):
            import zipfile

            with zipfile.ZipFile(local_filepath, "r") as zip_ref:
                zip_ref.extractall(install_path)
            print(f"Extracted {filename} to {install_path}")

        # 실행 가능한 파일이면 권한 부여
        if filename.endswith((".sh", ".bin")):
            os.chmod(local_filepath, 0o755)
            print(f"Set executable permissions for {local_filepath}")

    else:
        print(f"Failed to download {filename}, Status Code: {response.status_code}")

# Remove models.zip after extraction
models_zip_path = os.path.join(install_path, "meshes.zip")
if os.path.exists(models_zip_path):
    os.remove(models_zip_path)
    print(f"Removed {models_zip_path}")

print("Installation complete.")

# <<< Download models.zip <<<
