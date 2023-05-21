import tensorflow as tf
from keras import layers
from keras.models import Sequential
import json
import sys
import os
import numpy as np
import PIL
import pathlib
import matplotlib.pyplot as plt

# print(tf.config.list_physical_devices('GPU'))

batch_size = 32
img_height = 640
img_width = 640


def main():
    global train_labels, valid_labels, train_ds, val_ds, class_names
    
    training_dir = pathlib.Path("archive/train")
    valid_dir = pathlib.Path("archive/valid")

    train_annotations = json.load(open("archive/train_annotations"))
    train_labels = [l["category_id"] for l in train_annotations]
    # print(len(list(training_dir.glob("*.jpg"))))

    valid_annotations = json.load(open("archive/valid_annotations"))
    valid_labels = [l["category_id"] for l in valid_annotations]
    # print(len(list(valid_dir.glob("*.jpg"))))
    # PIL.Image.open(str(training_data[0])).show()

    train_ds = tf.keras.utils.image_dataset_from_directory(
        training_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )


    val_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names

    #PERFORMANCE OPTIMIZATION
    if True:
        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    run()

    cont_input = input("Continue? ([y]/n)")
    while(cont_input == "y" or cont_input == ""):
        run()


def run():
    global epochs, model
    if 'model' not in globals():
        if (input("Load existing model? (y/n) \n") == "y"):
            print("Loading model")
            model = tf.keras.models.load_model("model")
        else:
            print("Creating model")
            model = create_model()

    epochs = 4
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )
    
    plot_training_results(history)

    save_input = input("Save Model? ([y]/n)\n")
    if(save_input == "y" or save_input == ""):
        print("Saving model")
        model.save("model")
    else :
        print("Dismissing model")

def create_model():
    num_classes = len(class_names)

    data_augmentation = Sequential([
            layers.RandomFlip("horizontal",
                            input_shape=(img_height,
                                        img_width,
                                        3)),
            # layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    return model

def plot_training_example():
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(train_ds.class_names[labels[i]])
        plt.axis("off")
    plt.show()

def plot_training_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


#Shows the shape of the image batch (32 images 640x640x3)
def print_shape():
    for image_batch, labels_batch in train_ds:
      print(image_batch.shape)
      print(labels_batch.shape)
      break

if __name__ == "__main__":
    main()