from os import listdir
from os.path import join, isfile

directory_path = "/home/username/anypath"
contents = listdir(directory_path)

files = filter(lambda f: isfile(join(directory_path,f)), contents)

# print(files) # <filter object at 0x10a5203a0>
print(list(files)) # [list of files]