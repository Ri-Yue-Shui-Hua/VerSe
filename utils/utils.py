# -*- coding : UTF-8 -*-
# @file   : utils.py
# @Time   : 2024/4/9 0009 21:19
# @Author : Administrator
import os


def get_files(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files if file.endswith(suffix)]

