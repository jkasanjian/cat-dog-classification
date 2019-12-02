from shutil import copyfile
import numpy as np 

SOURCE_DIR  = 'data/all_data/'
TRAIN_DIR = 'data/train/'
VALIDATION_DIR = 'data/validation/'
NUM_TRAIN = 10000
NUM_VALIDATION = 2500
# file names: 0 - 12499


def split_data():
    dog_files = []
    cat_files = []

    for i in range(12500):
        dog_files.append('dog.' + str(i) + '.jpg')
        cat_files.append('cat.' + str(i) + '.jpg')
    
    for i in range(NUM_VALIDATION):
        index_dog = np.random.randint(0, len(dog_files))
        index_cat = np.random.randint(0, len(cat_files))
        dog_file = dog_files.pop(index_dog)
        cat_file = cat_files.pop(index_dog)
        copyfile(SOURCE_DIR + dog_file, VALIDATION_DIR + 'dogs/' + dog_file)
        copyfile(SOURCE_DIR + cat_file, VALIDATION_DIR + 'cats/' + cat_file)

    for i in range(NUM_TRAIN):
        dog_file = dog_files.pop()
        cat_file = cat_files.pop()
        copyfile(SOURCE_DIR + dog_file, TRAIN_DIR + 'dogs/' + dog_file)
        copyfile(SOURCE_DIR + cat_file, TRAIN_DIR + 'cats/' + cat_file)


# split_data()