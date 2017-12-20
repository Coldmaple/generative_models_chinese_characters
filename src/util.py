import os
import os.path

par_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
char_path = os.path.abspath(os.path.join(par_path, os.pardir)) + '/Gan_chinese_characters/character_images_labels/'

def getCharByte():
    
    fpath = char_path + "chars.txt"
    out = []
    with open(fpath,'rb') as f:
        while True:
            c = f.readline()
            if not c:
              break
            out = out+c.split()
    return out

def getCharInd():
    chars = getCharByte()

    fpath = char_path + "labels.txt"
    i = 0
    out = []
    with open(fpath,'rb') as f:
        while True:
            c = f.read(2)
            if not c:
                break
            if c in chars:
                out.append(i)
            i += 1
    return out
