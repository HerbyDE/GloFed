import os
import requests
import shutil
import zipfile

from urllib.parse import urlparse
from typing import List, Dict


def download_file(path: os.path, url: str) -> os.path:
    """
    This function downloads a specified file to a given directory and returns an os.path object.
    :param url: String containing the download URL
    :param path: os.path object where the file will be stored
    :return: os.path object pointing to file location
    """

    data_url = url
    url_path = urlparse(data_url)
    filename = os.path.basename(url_path.path)
    file_path = os.path.join(path, filename)

    try:
        with open(file_path) as f:
            f.close()

        print(f"Data has already been downloaded. File path returned.")

    except:
        # sends the get request with a timeout of 60 minutes (in case of a slow internet connection)
        req = requests.get(url=data_url, stream=True, timeout=60 * 60)

        if not req.ok:
            raise ConnectionError(f"Download failed with code {req.status_code}: {req.text}")

        try:
            with open(file_path, "wb") as f:
                shutil.copyfileobj(req.raw, f)
        except FileNotFoundError:
            raise FileNotFoundError("Please make sure the specified file path (incl. directories) exists!")

    return file_path


def unzip_file(path: os.path or str) -> Dict:
    """
    Unzips a given file and returns the target directory and a list of unzipped files
    :param path: os.path to to the .
    :return:Dict(keys: directory, files)
    """
    file_directory = os.path.dirname(os.path.realpath(path))

    # Unzips the dataset
    zip_file = zipfile.ZipFile(path)
    zip_file.extractall(file_directory)

    file_paths = []
    for el in zip_file.namelist():
        file_paths.append(os.path.join(file_directory, el))

    return {"directory": file_directory + "/", "files": file_paths}


def remove_file(path) -> None:
    """
    Removes a file from disk.
    :param path: os.path or str pointing to a file location
    :return: None
    """
    os.remove(path)
    return