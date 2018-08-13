import xml.etree.ElementTree as ET
from os import getcwd

import random

sets=[('raccoon_train'), ('raccoon_test')]

classes = ["raccoon"]


def convert_annotation(image_id, list_file):
    in_file = open('Raccoon_dataset/annotations/raccoon-%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()   # current working path

# 1 arrange train, val, test dataset:
ids = list(range(1,201))
random.shuffle(ids)
train = ids[:160]
test = ids[160:]
dataset = [train, test]
k = 0
for image_set in dataset:
    subset_name = sets[k]
    k += 1
    image_ids = open('Raccoon_dataset/%s.txt'%(subset_name), 'w')
    for i in range(len(image_set)):
        image_ids.write('%d\n'%(image_set[i]))
    image_ids.close()


# 2 produce train, val, test data:
for image_set in sets:
    image_ids = open('Raccoon_dataset/%s.txt'%(image_set)).read().strip().split()
    list_file = open('Raccoon_dataset/%s_data.txt'%(image_set), 'w')
    for image_id in image_ids:
        list_file.write('Raccoon_dataset/images/raccoon-%s.jpg'%(image_id))
        convert_annotation(image_id, list_file)
        list_file.write('\n')
    list_file.close()

