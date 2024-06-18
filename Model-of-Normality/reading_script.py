import requests
from bs4 import BeautifulSoup
import paramiko
from paramiko import SFTPClient
from io import BytesIO
import os

directory = r"V:\staff-umbrella\EleniSalient\Script"
# print("Directory contents:")
# for filename in os.listdir(directory):
#     print(filename)

# file_path = r"V:\staff-umbrella\EleniSalient\312_P\312_CLNF_pose.txt"

# # Open the file for reading
# with open(file_path, "r") as file:
#     content = file.read()

# URL of the webpage containing the .zip files
url = 'https://dcapswoz.ict.usc.edu/wwwdaicwoz/'

def download_zip_files(url,directory):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    zip_files = [url + link.get('href') for link in soup.find_all('a') if link.get('href').endswith('.zip')]

    for zip_url in zip_files:
        file_name = os.path.basename(zip_url)
        print(f"Downloading {file_name}...")

        # Download the file 
        file_data = requests.get(zip_url).content
        
        # Upload to SFTP directory
        upload_to_directory(file_name, file_data,directory)

def upload_to_directory(file_name, file_data, directory):
    """
    Save the file data to a specified directory.

    Args:
    file_name (str): The name of the file to save.
    file_data (bytes): The binary content of the file.
    directory (str): The path to the directory where the file will be saved.
    """
    # Ensure the directory exists, create if it does not
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Full path where the file will be saved
    full_path = os.path.join(directory, file_name)

    # Writing the file data to the file
    with open(full_path, 'wb') as file:
        file.write(file_data)
    print(f"File {file_name} has been uploaded to {directory}")

if __name__ == "__main__":
    download_zip_files(url,directory)
