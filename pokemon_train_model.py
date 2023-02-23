# %%
import pickle
import numpy as np
import pandas as pd
import os
import shutil
from keras.utils import image_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import ResNet50
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import Model
from sklearn.model_selection import train_test_split
from itertools import cycle
from PIL import Image
from matplotlib import pyplot as plt

# %%
# Making new folder for the data
if not os.path.isdir('./res'):
    os.mkdir('./res')
PARENT = './PokemonData/'
DATA = './res/'
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_DIMS = (IMG_WIDTH, IMG_HEIGHT)
IMG_CHANNELS = 3

# %%
# Making folders for new train/test split
sets = ['X_train_img', 'X_test_img', 'X_train_aug_img']
for folder in sets:
    if not os.path.isdir(os.path.join(DATA, folder)):
        os.mkdir(os.path.join(DATA, folder))

# %%
# Putting folders of each class into the new folders
for set in sets:
    for label in os.listdir(PARENT):
        class_path = os.path.join(DATA, set, label)
        os.makedirs(class_path, exist_ok=True)

# %%
X = []
y = []
for folder in os.listdir(PARENT):
    folder_path = os.path.join(PARENT, folder)
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        X.append(filepath)
        y.append(folder)

# %%
# Making a train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=True,
    random_state=1410,
)

# %%
TRAIN_SET = sets[0]
TEST_SET = sets[1]


def copy_to_new_folders(data, labels, set):
    # If folders already have images don't copy again
    num_imgs = sum([len([filename for filename in os.listdir(
        os.path.join(DATA, set, label))]) for label in labels])
    if num_imgs > 0:
        print(f"Folders in {DATA + set} already contain {num_imgs} images.")
        print("If you wish to copy again remove the folder and rerun the program")
        return
    print(f"COPYING IMAGES to {DATA + set}...")
    for img, label in zip(data, labels):
        filename = img.split('/')[-1]
        src = img
        dst = os.path.join(DATA, set, label, filename)
        shutil.copy(src, dst)


copy_to_new_folders(X_train, y_train, TRAIN_SET)
copy_to_new_folders(X_test, y_test, TEST_SET)

# %%


def count_instances(data_path):
    pokemon_all = os.listdir(data_path)
    max = 0
    min = float('inf')
    avg = 0
    for pokemon in pokemon_all:
        p = os.path.join(data_path, pokemon)
        num_pokemon = len(os.listdir(p))
        if num_pokemon > max:
            max = num_pokemon
        if num_pokemon < min:
            min = num_pokemon
        avg += num_pokemon
        print(pokemon + ' count is: ' + str(num_pokemon))
    avg = avg/len(pokemon_all)
    print(f"Max: {max} Min: {min} Avg: {avg}")
    return max


most_imgs = count_instances(os.path.join(DATA, TRAIN_SET))

# %%
# Create ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(rescale=1.0/255,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             rotation_range=30,
                             fill_mode='nearest'
                             )

# %%


def augment_data(data_path):
    for folder in os.listdir(data_path):
        folder_contents = os.listdir(os.path.join(DATA, TRAIN_SET, folder))
        folder_size = len(folder_contents)
        imgs_to_augment = most_imgs - folder_size
        imgs_augmented = 0
        for img in cycle(folder_contents):
            if imgs_to_augment <= imgs_augmented:
                print(f"Augmented {imgs_augmented} images for {folder}")
                break
            img_path = os.path.join(data_path, folder, img)
            aug_src_img = Image.open(img_path)
            aug_src_img = aug_src_img.resize(IMG_DIMS)
            aug_src_img = aug_src_img.convert('RGB')
            aug_src_img = np.array(aug_src_img)
            aug_img = datagen.apply_transform(
                aug_src_img,
                datagen.get_random_transform(aug_src_img.shape))
            filename = f'aug_{folder}_{imgs_augmented}.jpg'
            savepath = os.path.join(data_path, folder, filename)
            aug_img = Image.fromarray(aug_img)
            aug_img.save(savepath)
            imgs_augmented += 1


augment_data(os.path.join(DATA, TRAIN_SET))

# %%
count_instances(os.path.join(DATA, TRAIN_SET))

# %%
print('TRAIN: ')
train_generator = datagen.flow_from_directory(
    directory=DATA + TRAIN_SET,
    class_mode='categorical',
    batch_size=64,
    shuffle=True,
    target_size=IMG_DIMS,
)
print('TEST: ')
val_generator = datagen.flow_from_directory(
    directory=DATA + TEST_SET,
    class_mode='categorical',
    batch_size=64,
    shuffle=True,
    target_size=IMG_DIMS,
)

# %%
# Download pretrained Resnet50
resnet = ResNet50(include_top=False, weights='imagenet',
                  input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

# %%
# Add layers to the Resnet50
layer1 = GlobalAveragePooling2D()(resnet.output)
layer2 = Dense(1024, activation='relu')(layer1)
layer3 = Dense(512, activation='relu')(layer2)
layer_out = Dense(149, activation='softmax')(layer3)

model = Model(inputs=resnet.input, outputs=layer_out)
model.summary()

# %%
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
# Train
epochs = 15
hist = model.fit(train_generator, epochs=epochs, validation_data=val_generator)

# %%
# Visualise training data


def create_graphs(history, epochs, name):
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
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.title('Training and Validation Loss')
    plt.show()
    plt.savefig(f'{name}.png')


# %%
# Save history and model and evaluation
with open(f'history_{epochs}_epochs.pkl', 'wb') as f:
    pickle.dump(hist.history, f)

model.save(f'resnet50_model_{epochs}_epochs.h5')
create_graphs(hist, epochs, f'resnet50_model_{epochs}_epochs')

test_loss, test_accuracy = model.evaluate(val_generator)
with open(f'evaluation_results_{epochs}_epochs.txt', 'w') as f:
    f.write(f'Test Loss: {test_loss:.4f}\n')
    f.write(f'Test Accuracy: {test_accuracy:.4f}')
