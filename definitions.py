import os

ROOT_DIRECTORY = os.path.dirname(__file__)

def get_full_path(*path):
    return os.path.join(ROOT_DIRECTORY, *path)
