import numpy as np
import glob
import os
import pandas as pd

import random
import tensorflow as tf
import re
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def read_data(cup_material,liquid,path_name):
    print(cup_material+liquid)

    source='./data/microwave_dataset'
    materials = glob.glob(source + '/*/')
    class_count = 0
    data = []
    for material in materials:
        class_count = class_count + 1
        liquids = glob.glob(material + '/*/')

        for liquid in liquids:
            positions = glob.glob(liquid + '/*/')
            for position in positions:
                runs = sorted(glob.glob(position + '*.csv'),  key=numericalSort)
                for run in runs:
                    load = pd.read_csv(run, skiprows=6)
                    load = load.drop(index=load.index[-1], axis=0)

                    data[class_count].append(load.to_numpy())



    return data


mat = "plastic"
liq = "ethanol"


dataset = read_data(mat,liq,"a_1.csv")
x=1
dataset = np.asarray(dataset)
love = np.size(dataset)
calc = 1