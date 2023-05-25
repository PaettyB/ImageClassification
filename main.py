import tensorflow as tf
import keras_tuner as kt
from keras import layers
from keras.models import Sequential
import json
import sys
import os
import numpy as np
import PIL
import pathlib
import matplotlib.pyplot as plt


batch_size = 5
img_height = 400
img_width = 400


def main():
    global train_labels, valid_labels, train_ds, val_ds, class_names
    
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

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
    while True:
        c = input("Continue? ([y]/n): ")
        if c == "y" or c == "":
            run()
        else:
            break


def run():
    global epochs, model
    if 'model' not in globals():
        if (input("Load existing model? (y/[n]): ") == "y"):
            print("Loading model")
            model = tf.keras.models.load_model("model")
        else:
            n = input("Create tuned model? ([y]/n): ")
            if (n == "y" or n ==""):
                print("Creating tuned model")
                tuner = kt.Hyperband(create_tuned_model_v2,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=2,
                     directory='tune',
                     project_name='non_softmax')
                stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4)
                epochs = 40
                tuner.search(train_ds, validation_data=val_ds, batch_size=batch_size, callbacks=[stop_early])
                best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
                print(best_hps)
                print("-----------")
                print("Filters: ", best_hps.get("filters_1"), best_hps.get("filters_2"), best_hps.get("filters_1"))
                print("Kernels: ", best_hps.get("kernel_1"), best_hps.get("kernel_2"), best_hps.get("kernel_1"))
                print("Activation: ", best_hps.get("activation"))
                print("Droprate: ", best_hps.get("drop"))
                print("Dense units: ", best_hps.get("dense_units"))
                print("Learning rate: ", best_hps.get("learning_rate"))
                best_model = tuner.hypermodel.build(best_hps)
                best_history = best_model.fit(train_ds, validation_data=val_ds, epochs=epochs)
                plot_training_results(best_history)
                best_model.save("best_model")
                sys.exit()

            else:
                print("Creating normal model")
                model = create_model()

    epochs = 4
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        batch_size=batch_size
    )
    
    plot_training_results(history)

    save_input = input("Save Model? ([y]/n): ")
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
            # layers.RandomZoom(0.1),
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

def create_tuned_model(hp: kt.HyperParameters):
    num_classes = len(class_names)
    model = Sequential()
    data_augmentation = Sequential([
            layers.RandomFlip("horizontal",
                            input_shape=(img_height,
                                        img_width,
                                        3)),
            # layers.RandomRotation(0.1),
            # layers.RandomZoom(0.1),
        ])
    model.add(data_augmentation)
    model.add(layers.Rescaling(1./255))

    hp_filters_1 = hp.Choice("filters_1", values=[8,16,32, 64])
    hp_filters_2 = hp.Choice("filters_2", values=[8,16,32,64,128])
    hp_filters_3 = hp.Choice("filters_3", values=[8,16,32,64,128])
    hp_activation = hp.Choice("activation", values=["relu", "tanh", "sigmoid"])

    hp_drop = hp.Float("drop", min_value=0.05, max_value=0.3, step=0.05)
    hp_dense = hp.Int("dense_units", min_value=32, max_value=512, step=64)

    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.add(layers.Conv2D(hp_filters_1, 3, padding="same", activation=hp_activation))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(hp_filters_2, 3, padding="same", activation=hp_activation))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(hp_filters_3, 3, padding="same", activation=hp_activation))
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(hp_drop))
    model.add(layers.Flatten())
    model.add(layers.Dense(hp_dense, activation=hp_activation))
    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    return model

def create_tuned_model_v2(hp: kt.HyperParameters):
    num_classes = len(class_names)
    model = Sequential()
    data_augmentation = Sequential([
            layers.RandomFlip("horizontal",
                            input_shape=(img_height,
                                        img_width,
                                        3)),
            # layers.RandomRotation(0.1),
            # layers.RandomZoom(0.1),
        ])
    model.add(data_augmentation)
    model.add(layers.Rescaling(1./255))

    hp_filters_1 = hp.Choice("filters_1", values=[3,8,16,32,64])
    hp_filters_2 = hp.Choice("filters_2", values=[3,8,16,32,64])
    hp_filters_3 = hp.Choice("filters_3", values=[3,8,16,32,64])
    
    hp_kernel_1 = hp.Choice("kernel_1", values=[3,5,10])
    hp_kernel_2 = hp.Choice("kernel_2", values=[3,5,10])
    hp_kernel_3 = hp.Choice("kernel_3", values=[3,5,10])
    # hp_activation = hp.Choice("activation", values=["relu", "tanh"])

    hp_drop = hp.Float("drop", min_value=0.05, max_value=0.3, step=0.05)
    hp_dense = hp.Int("dense_units", min_value=128, max_value=1024, step=64)

    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.add(layers.Conv2D(hp_filters_1, hp_kernel_1, padding="same", activation="relu"))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(hp_filters_2, hp_kernel_2, padding="same", activation="relu"))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(hp_filters_3, hp_kernel_3, padding="same", activation="relu"))
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(hp_drop))
    model.add(layers.Flatten())
    model.add(layers.Dense(hp_dense, activation="relu"))
    model.add(layers.Dense(num_classes))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
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