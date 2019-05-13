import sys
import os


def make_if_not_exist(path):
    try:
        if not os.path.isdir(path):
            print('[INFO] Creating "{}"...'.format(path))
            os.makedirs(path)
    except OSError:
        sys.exit('Fatal: the directory "{}" does not exist and cannot be created'.format(path))
