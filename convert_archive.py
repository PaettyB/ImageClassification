import os
import json

train_annotations = json.load(open("archive/train_annotations"))
train_labels = [l["category_id"] for l in train_annotations]
# print(train_labels)

valid_annotations = json.load(open("archive/valid_annotations"))
valid_labels = [l["category_id"] for l in valid_annotations]

# for i in range(len(train_labels)):
#     cla = "penguin" if train_labels[i] == 1 else "turtle"
#     print("archive/train/image_id_{:03d}.jpg".format(i))
#     os.rename("archive/train/image_id_{:03d}.jpg".format(i), "archive/train/{}/image_id_{:03d}.jpg".format(cla, i))


for i in range(len(valid_labels)):
    cla = "penguin" if valid_labels[i] == 1 else "turtle"
    print("archive/valid/image_id_{:03d}.jpg".format(i))
    os.rename("archive/valid/image_id_{:03d}.jpg".format(i), "archive/valid/{}/image_id_{:03d}.jpg".format(cla, i))