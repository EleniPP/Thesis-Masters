
import pysftp
import os

directory = r"V:\staff-umbrella\EleniSalient\312_P"
print("Directory contents:")
for filename in os.listdir(directory):
    print(filename)

file_path = r"V:\staff-umbrella\EleniSalient\312_P\312_CLNF_pose.txt"

# Open the file for reading
with open(file_path, "r") as file:
    content = file.read()


# filename = '336_P.zip'

# # localpath = os.path.join("C:", "Users", "eleni","Downloads")
# localpath = r'C:\Users\eleni\Downloads'
# # Components of the path
# drive = "V:"
# folder1 = "staff-umbrella"
# folder2 = "EleniSalient"

# localFile = os.path.join(localpath,filename)
# # Construct the path
# # remotepath = os.path.join(drive, folder1, folder2)
# remotedirectory = "/{}/{}/{}".format(drive.replace(":", ""), folder1, folder2)

# # cnopts = pysftp.CnOpts(knownhosts=os.path.join(remotepath, "keyfile"))
# cnopts = pysftp.CnOpts()
# cnopts.hostkeys = None

# # SFTP server details
# host = 'sftp.tudelft.nl'
# user = 'elenipapadopou'
# password = 'Aggeliki6980173069!'

# with pysftp.Connection(host=host, username=user, password=password, cnopts=cnopts) as sftp:
#     sftp.put(localFile, os.path.join(remotedirectory, filename)) 
#     # sftp.chdir(remotepath)  # Change to the remote directory
#     # sftp.put(localFile)  # Upload file, retains original filename
#     # sftp.put(localFile, remotepath)

#     # Example to download a file from the SFTP server, adjust 'fileOnServer' and 'locationOnPC'
#     # fileOnServer = 'path/to/remote/file'
#     # locationOnPC = 'path/to/local/directory'
#     # sftp.get(fileOnServer, locationOnPC)


# ----------------------------------------------------------------------------------------------------------------

# import os
# import requests
# from bs4 import BeautifulSoup
# import paramiko
# from paramiko import SFTPClient
# from io import BytesIO

# # Configuration for SFTP
# sftp_host = 'sftp.tudelft.nl'
# sftp_port = 22
# sftp_username = 'elenipapadopou'
# sftp_password = 'Aggeliki6980173069!'
# remote_directory = 'V:\staff-umbrella\EleniSalient\Script'

# # URL of the webpage containing the .zip files
# url = 'https://dcapswoz.ict.usc.edu/wwwdaicwoz/'

# def download_zip_files(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     zip_files = [url + link.get('href') for link in soup.find_all('a') if link.get('href').endswith('.zip')]

#     for zip_url in zip_files:
#         file_name = os.path.basename(zip_url)
#         print(f"Downloading {file_name}...")

#         # Download the file
#         file_data = requests.get(zip_url).content
        
#         # Upload to SFTP
#         upload_to_sftp(file_name, file_data)

# def upload_to_sftp(file_name, file_data):
#     # Establish SFTP connection
#     with paramiko.Transport((sftp_host, sftp_port)) as transport:
#         transport.connect(username=sftp_username, password=sftp_password)
#         sftp = SFTPClient.from_transport(transport)
        
#         # Upload the file
#         with BytesIO(file_data) as f:
#             remote_path = os.path.join(remote_directory, file_name)
#             print(f"Uploading {file_name} to {remote_path}...")
#             sftp.putfo(f, remote_path)
#         sftp.close()
#         print(f"{file_name} uploaded successfully.")

# if __name__ == "__main__":
#     download_zip_files(url)
