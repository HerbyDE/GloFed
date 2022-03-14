import os
import requests
import zipfile
import shutil

from utils.file_loader import download_file, unzip_file


class DataHandler(object):

    def __init__(self):
        pass

    def download_dataset(self):
        data_url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
        current_dir = os.path.dirname(os.path.realpath(__file__))
        download_dir = os.path.join(current_dir, "download")
        zip_file = download_file(path=download_dir, url=data_url)

        return unzip_file(zip_file)

    def download_embedding(self):
        data_url = "https://nlp.stanford.edu/data/glove.twitter.27B.zip"
        current_dir = os.path.dirname(os.path.realpath(__file__))
        download_dir = os.path.join(current_dir, "download")
        zip_file = download_file(path=download_dir, url=data_url)

        return unzip_file(zip_file)
