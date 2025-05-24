import zipfile
import shutil
import os


def open_zipfile(file: str) -> None:
    if os.path.exists('repository/'):
        shutil.rmtree('repository/')

    with zipfile.ZipFile(file, 'r') as f:
        f.extractall(path='repository/')