import torch
from torchvision import datasets
from util.ImageFolderWithPaths import ImageFolderWithPaths
from random import seed, randint
import os

def random_split(PATH1:str, PATH2:str, data:ImageFolderWithPaths, split_percent:list = [0.8, 0.1, 0.1], rand_seed = 8) -> tuple:
    
    
    assert sum(split_percent) == 1 and len(split_percent) == 3
    
    seed(rand_seed)
    
    list_of_patients = [f1.path for f1 in os.scandir(PATH1) if f1.is_dir()] + [f2.path for f2 in os.scandir(PATH2) if f2.is_dir()]
        
    train_labels, val_labels, test_labels = get_labels(data, split_percent, list_of_patients)
    
    train_arr, val_arr, test_arr = [], [], []
    
    for i, tup in enumerate(data):
        if i%5000 == 0:
            print(i)
        cur_label = tup[2][:54]
        if cur_label in train_labels:
            train_arr.append(i)
        elif cur_label in val_labels:
            val_arr.append(i)
        elif cur_label in test_labels:
            test_arr.append(i)
        else:
            raise KeyError(f"The dataset label isn't in the label lists {tup[2]=}\n{cur_label=}")
        
    train_set = torch.utils.data.Subset(data, train_arr)
    val_set = torch.utils.data.Subset(data, val_arr)
    test_set = torch.utils.data.Subset(data, test_arr)
    
    print('Returning')

    return (train_set, val_set, test_set)
    
    
        
def get_labels(data:ImageFolderWithPaths, split_percent: list, list_of_patients: list) -> tuple:
    
   
    val_amt = int(len(list_of_patients) * split_percent[1])
    test_amt = int(len(list_of_patients) * split_percent[2])
    train_amt = len(list_of_patients) - val_amt - test_amt
    
    
    val_labels, test_labels = set(), set()
    
    for x in range(val_amt):
        index = randint(0, len(list_of_patients)-1)
        val_labels.add(list_of_patients[index])
        list_of_patients.pop(index)
    
    for y in range(test_amt):
        index = randint(0, len(list_of_patients)-1)
        test_labels.add(list_of_patients[index])
        list_of_patients.pop(index)
        
    train_labels = set(list_of_patients)
    
    assert len(val_labels) == val_amt and len(test_labels) == test_amt and len(train_labels) == train_amt
    
    print("Finished getting labels")
    return (train_labels, val_labels, test_labels)
