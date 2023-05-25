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


img_height = 28
img_width = 28
input_shape = (img_width, img_height, 1)

def main():
    global train_images, train_labels, test_images, test_labels, class_names
    
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    print(train_images.shape)
    train_images = train_images.reshape(60000, img_width, img_height, 1)
    test_images = test_images.reshape(10000, img_width, img_height, 1)

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
            model = tf.keras.models.load_model("fashion_model")
        else:
            n = input("Create tuned model? ([y]/n): ")
            if (n == "y" or n ==""):
                print("Creating tuned model")
                tuner = kt.Hyperband(create_tuned_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='fashion_tune_conv',
                     project_name='01')
                stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4)
                epochs = 40
                tuner.search(train_images, train_labels, validation_split=0.2, callbacks=[stop_early])
                best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
                print("-----------------------------")
                # print("Dense units: ", best_hps.get("dense_units_1"), best_hps.get("dense_units_2"), best_hps.get("dense_units_3"))
                print("Filters: ", best_hps.get("filters_1"), best_hps.get("filters_2"), best_hps.get("filters_1"))
                print("Kernels: ", best_hps.get("kernel_1"), best_hps.get("kernel_2"), best_hps.get("kernel_1"))
                print("Dense Units: ", best_hps.get("dense_units"))
                print("Droprate: ", best_hps.get("drop"))
                print("Learning rate: ", best_hps.get("learning_rate"))
                print("-----------------------------")
                best_model = tuner.hypermodel.build(best_hps)
                best_history = best_model.fit(train_images, train_labels, validation_split=0.2, epochs=epochs)
                eval_result = best_model.evaluate(test_images, test_labels)
                print("[test loss, test accuracy]:", eval_result)
                plot_training_results(best_history)
                sys.exit()

            else:
                print("Creating normal model")
                model = create_model()

    epochs = 4
    history = model.fit(
        train_images,
        train_labels,
        validation_split=0.2,
        epochs=epochs
    )
    
    plot_training_results(history)

    save_input = input("Save Model? ([y]/n): ")
    if(save_input == "y" or save_input == ""):
        print("Saving model")
        model.save("fashtion_model")
    else :
        print("Dismissing model")

def create_model():
    num_classes = len(class_names)

    data_augmentation = Sequential([
            layers.RandomFlip("horizontal", input_shape=(img_height, img_width)),
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

    model.add(layers.Conv2D(hp_filters_1, hp_kernel_1, padding="same", activation="relu", input_shape=input_shape))
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

def create_tuned_dense_model(hp: kt.HyperParameters):
    num_classes = len(class_names)
    model = Sequential()
    
    hp_dense_1 = hp.Int("dense_units_1", min_value=32, max_value=1024, step=64)
    hp_dense_2 = hp.Int("dense_units_2", min_value=32, max_value=1024, step=64)
    hp_dense_3 = hp.Int("dense_units_3", min_value=32, max_value=1024, step=64)
    
    hp_activation = hp.Choice("activation", values=["relu", "tanh"])


    hp_drop = hp.Float("drop", min_value=0.05, max_value=0.3, step=0.05)
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(hp_dense_1, activation=hp_activation))
    model.add(layers.Dense(hp_dense_2, activation=hp_activation))
    model.add(layers.Dense(hp_dense_3, activation=hp_activation))
    model.add(layers.Dropout(hp_drop))
    model.add(layers.Dense(num_classes))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    return model

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


if __name__ == "__main__":
    main()