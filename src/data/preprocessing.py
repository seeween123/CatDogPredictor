import os
import random
from sklearn.model_selection import train_test_split
import shutil
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def copy_images(imgs,lbls,split,base_dir: str = '../dataset/Processed/cats_dogs_split'):
    for img,l in zip(imgs,lbls):
        cls='cat' if l==0 else 'dog'
        try:
            shutil.copy(img, os.path.join(base_dir,split,cls,os.path.basename(img)))
        except:
            pass

def preprocess_data(
    source_dir: str = '../dataset/Raw/PetImages',
    base_dir: str = '../dataset/Processed/cats_dogs_split'
) -> tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
    """
    Data cleaning for the UCI Heart Disease dataset.
    """

    cats=[os.path.join(source_dir,'Cat',f) for f in os.listdir(os.path.join(source_dir,'Cat')) if f.endswith('.jpg')]
    dogs=[os.path.join(source_dir,'Dog',f) for f in os.listdir(os.path.join(source_dir,'Dog')) if f.endswith('.jpg')]

    # Set desired number of images per class
    N = 1000
    random.seed(42)
    cats = random.sample(cats, min(N, len(cats)))
    dogs = random.sample(dogs, min(N, len(dogs)))

    images=cats+dogs
    labels=[0]*len(cats)+[1]*len(dogs)
    X_train,X_temp,y_train,y_temp=train_test_split(images,labels,test_size=0.2,stratify=labels,random_state=42)
    X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=0.5,stratify=y_temp,random_state=42)

    for split in ['train','val','test']:
        for cls in ['cat','dog']:
            os.makedirs(os.path.join(base_dir,split,cls),exist_ok=True)

    # Copy images to respective directories
    copy_images(X_train,y_train,'train')
    copy_images(X_val,y_val,'val')
    copy_images(X_test,y_test,'test')

    # Data augmentation
    IMG_SIZE=(224,224)
    BATCH_SIZE=32
    train_datagen=ImageDataGenerator(rescale=1./255,rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
    test_datagen=ImageDataGenerator(rescale=1./255)

    train_generator=train_datagen.flow_from_directory(os.path.join(base_dir,'train'),target_size=IMG_SIZE,batch_size=BATCH_SIZE,class_mode='binary')
    val_generator=test_datagen.flow_from_directory(os.path.join(base_dir,'val'),target_size=IMG_SIZE,batch_size=BATCH_SIZE,class_mode='binary')
    test_generator=test_datagen.flow_from_directory(os.path.join(base_dir,'test'),target_size=IMG_SIZE,batch_size=BATCH_SIZE,class_mode='binary',shuffle=False)

    return train_generator, val_generator, test_generator