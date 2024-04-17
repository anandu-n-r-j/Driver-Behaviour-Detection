import pandas as pd
from PIL import Image
import cv2
import os
import random
import matplotlib.pyplot as plt
from skimage.io import imread
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

def find_unreadable_images_path(folder_path):
    unreadable_images_path = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        try:
            with Image.open(file_path) as img:
                pass
        except Exception as e:
            unreadable_images_path.append(file_path)

    return unreadable_images_path

dataset_path = r'C:\Users\anand\OneDrive\Documents\Driver_Behaviour_Detection\Revitsone-5classes'

other_activities_images = []
safe_driving_images = []
talking_phone_images = []
texting_phone_images = []
turning_images = []

class_folders = ['other_activities', 'safe_driving', 'talking_phone', 'texting_phone', 'turning']

for folder in class_folders:
    class_folder_path = os.path.join(dataset_path, folder)
    for file in os.listdir(class_folder_path):
        file_path = os.path.join(class_folder_path, file)
        if file.endswith(".png") or file.endswith(".jpg"):
            if folder == 'other_activities':
                other_activities_images.append(file_path)
            elif folder == 'safe_driving':
                safe_driving_images.append(file_path)
            elif folder == 'talking_phone':
                talking_phone_images.append(file_path)
            elif folder == 'texting_phone':
                texting_phone_images.append(file_path)
            elif folder == 'turning':
                turning_images.append(file_path)

folder_path_other = r'C:\Users\anand\OneDrive\Documents\Driver_Behaviour_Detection\Revitsone-5classes\other_activities'
paths_to_remove_other = find_unreadable_images_path(folder_path_other)
folder_path_turn = r'C:\Users\anand\OneDrive\Documents\Driver_Behaviour_Detection\Revitsone-5classes\turning'
paths_to_remove_turn = find_unreadable_images_path(folder_path_turn)

for path_to_remove in paths_to_remove_other:
    if path_to_remove in other_activities_images:
        other_activities_images.remove(path_to_remove)

for path_to_remove in paths_to_remove_turn:
    if path_to_remove in turning_images:
        turning_images.remove(path_to_remove)

plt.figure(1, figsize=(10, 10), facecolor='white')
plt.axis('off')

plt.suptitle("Random Images of People Talking on the Phone",
             fontsize=25)

for i in range(4):
    random_img = random.choice(talking_phone_images)
    imgs = imread(random_img)
    plt.subplot(2, 2, i + 1)
    plt.imshow(imgs)

plt.show()

num_other = len(other_activities_images)
num_safe = len(safe_driving_images)
num_talking = len(talking_phone_images)
num_text = len(texting_phone_images)
num_turn = len(turning_images)

print(f"Number of samples in Class 'Other': {num_other}")
print(f"Number of samples in Class 'Safe Driving': {num_safe}")
print(f"Number of samples in Class 'Talking Phone': {num_talking}")
print(f"Number of samples in Class 'Texting Phone': {num_text}")
print(f"Number of samples in Class 'Turning': {num_turn}")

split_ratio = {
    'Other': (.75 * num_other, .15 * num_other, .1 * num_other),
    'Safe Driving': (.75 * num_safe, .15 * num_safe, .1 * num_safe),
    'Talking Phone': (.75 * num_talking, .15 * num_talking, .1 * num_talking),
    'Texting Phone': (.75 * num_text, .15 * num_text, .1 * num_text),
    'Turning': (.75 * num_turn, .15 * num_turn, .1 * num_turn)
}

for class_name, (train_count, val_count, test_count) in split_ratio.items():
    print(f"For class '{class_name}':")
    print(f"Number of samples for training: {train_count}")
    print(f"Number of samples for validation: {val_count}")
    print(f"Number of samples for testing: {test_count}\n")

train_other = other_activities_images[:1596]
test_other = other_activities_images[1596:1916]
valid_other = other_activities_images[1916:]

print("For class 'Other':")
print("Train:", len(train_other))
print("Test:", len(test_other))
print("Valid:", len(valid_other))

train_safe = safe_driving_images[:1652]
test_safe = safe_driving_images[1652:1982]
valid_safe = safe_driving_images[1982:]

print("\nFor class 'Safe Driving':")
print("Train:", len(train_safe))
print("Test:", len(test_safe))
print("Valid:", len(valid_safe))

train_talking = talking_phone_images[:1627]
test_talking = talking_phone_images[1627:1952]
valid_talking = talking_phone_images[1952:]

print("\nFor class 'Talking Phone':")
print("Train:", len(train_talking))
print("Test:", len(test_talking))
print("Valid:", len(valid_talking))

train_text = texting_phone_images[:1652]
test_text = texting_phone_images[1652:1982]
valid_text = texting_phone_images[1982:]

print("\nFor class 'Texting Phone':")
print("Train:", len(train_text))
print("Test:", len(test_text))
print("Valid:", len(valid_text))

train_turn = turning_images[:1543]
test_turn = turning_images[1543:1848]
valid_turn = turning_images[1848:]

print("\nFor class 'Turning':")
print("Train:", len(train_turn))
print("Test:", len(test_turn))
print("Valid:", len(valid_turn))

train_other_df = pd.DataFrame({'image': train_other, 'label': 'Other'})
train_safe_df = pd.DataFrame({'image': train_safe, 'label': 'Safe Driving'})
train_talking_df = pd.DataFrame({'image': train_talking, 'label': 'Talking Phone'})
train_text_df = pd.DataFrame({'image': train_text, 'label': 'Texting Phone'})
train_turn_df = pd.DataFrame({'image': train_turn, 'label': 'Turning'})

test_other_df = pd.DataFrame({'image': test_other, 'label': 'Other'})
test_safe_df = pd.DataFrame({'image': test_safe, 'label': 'Safe Driving'})
test_talking_df = pd.DataFrame({'image': test_talking, 'label': 'Talking Phone'})
test_text_df = pd.DataFrame({'image': test_text, 'label': 'Texting Phone'})
test_turn_df = pd.DataFrame({'image': test_turn, 'label': 'Turning'})
valid_other_df = pd.DataFrame({'image': valid_other, 'label': 'Other'})
valid_safe_df = pd.DataFrame({'image': valid_safe, 'label': 'Safe Driving'})
valid_talking_df = pd.DataFrame({'image': valid_talking, 'label': 'Talking Phone'})
valid_text_df = pd.DataFrame({'image': valid_text, 'label': 'Texting Phone'})
valid_turn_df = pd.DataFrame({'image': valid_turn, 'label': 'Turning'})

train_df = pd.concat([train_other_df, train_safe_df, train_talking_df, train_text_df, train_turn_df], ignore_index=True)
test_df = pd.concat([test_other_df, test_safe_df, test_talking_df, test_text_df, test_turn_df], ignore_index=True)
val_df = pd.concat([valid_other_df, valid_safe_df, valid_talking_df, valid_text_df, valid_turn_df], ignore_index=True)

print("Number of samples in the training dataframe:", len(train_df))
print("Number of samples in the testing dataframe:", len(test_df))
print("Number of samples in the validation dataframe:", len(val_df))

random_height = random.choice(train_other)
Image= cv2.imread(random_height)

h, w= Image.shape[:2]

print("The height is ", h)
print("The width is ", w)

Batch_size = 64
Img_height = 224
Img_width = 224

trainGenerator = ImageDataGenerator(rescale=1./255.)
valGenerator = ImageDataGenerator(rescale=1./255.)
testGenerator = ImageDataGenerator(rescale=1./255.)

trainDataset = trainGenerator.flow_from_dataframe(
  dataframe=train_df,
  class_mode="categorical",
  x_col="image",
  y_col="label",
  batch_size=Batch_size,
  seed=42,
  shuffle=True,
  target_size=(Img_height,Img_width)
)

testDataset = testGenerator.flow_from_dataframe(
  dataframe=test_df,
  class_mode='categorical',
  x_col="image",
  y_col="label",
  batch_size=Batch_size,
  seed=42,
  shuffle=True,
  target_size=(Img_height,Img_width)
)

valDataset = valGenerator.flow_from_dataframe(
  dataframe=val_df,
  class_mode='categorical',
  x_col="image",
  y_col="label",
  batch_size=Batch_size,
  seed=42,
  shuffle=True,
  target_size=(Img_height,Img_width)
)

def InceptionV1():
    # Input layer with shape (224, 224, 3)
    inp = layers.Input((224, 224, 3))

    # Convolutional layers
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inp)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(192, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Inception modules
    tower_1 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_3 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(tower_3)
    x = layers.concatenate([tower_1, tower_2, tower_3], axis=3)

    tower_1 = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(192, (3, 3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(96, (1, 1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(208, (5, 5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_3 = layers.Conv2D(16, (1, 1), padding='same', activation='relu')(tower_3)
    x = layers.concatenate([tower_1, tower_2, tower_3], axis=3)

    # Global average pooling layer
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(5, activation='softmax')(x)  # Assuming 5 classes

    # Create the modified InceptionV1 model
    Inception = models.Model(inputs=inp, outputs=x)

    return Inception

Inception = InceptionV1()

Inception.summary()

Inception.compile(loss=BinaryCrossentropy(),
                         optimizer=Adam(learning_rate=0.001),
                         metrics=['accuracy'])

history = Inception.fit(trainDataset, epochs=20, validation_data=valDataset)

# Save the model
Inception.save("inception_model.h5")
