from os import *
import pandas as pd
from shutil import copyfile, copytree

def deal_with_data(data_type: str):
    folder = "data/" + data_type + "_folder"
    try:
        mkdir(folder)
        for i in range(3):
            mkdir(folder + "/" + str(i))
    except Exception:
        pass
    content = pd.read_csv("data/" + data_type + ".txt", header=None)
    for i in range(len(content)):
        name, label = content[0][i], content[1][i]
        copyfile("data/" + data_type + "/" + name, folder + "/" + str(label) + "/" + name)

deal_with_data("train")
deal_with_data("validation")

try:
    mkdir("test_folder")
    copytree("test", "test_folder/test")
except Exception:
    pass