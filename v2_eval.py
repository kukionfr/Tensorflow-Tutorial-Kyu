import tensorflow as tf
from tensorflow.keras.layers import average, maximum
import tensorflow_hub as hub
import os
import numpy as np
import tensorflow_hub as hub
import pathlib

def read_and_label(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    label = get_label(file_path)
    return img, label


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return tf.reshape(tf.where(parts[-4] == CLASS_NAMES), [])

IMG_HEIGHT = 100
IMG_WIDTH = 100
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

test_data_dir = r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\test'
test_data_dir = pathlib.Path(test_data_dir)
CLASS_NAMES = np.array(
    [item.name for item in test_data_dir.glob('*') if item.name != "LICENSE.txt" and item.name != ".DS_store"])
test_image_count = len(list(test_data_dir.glob('*\*\image\*.jpg')))
test_list_ds = tf.data.Dataset.list_files(str(test_data_dir / '*\*\image\*'))
test_labeled_ds = test_list_ds.map(read_and_label, num_parallel_calls=AUTOTUNE)
test_ds = (test_labeled_ds
           .cache("./cache/fibro_test.tfcache")
           .repeat()
           .batch(BATCH_SIZE)
           .prefetch(buffer_size=AUTOTUNE)  # time it takes to produce next element
           )


lr = 'AdamW'
model_cnnA = tf.keras.models.load_model('cnn/A/'+lr+'/full_model.h5', compile=False)
# model_cnnB = tf.keras.models.load_model('cnn/B/'+lr+'/full_model.h5', compile=False)
resnetv2 = tf.keras.models.load_model('cnn/resnetv2/'+lr+'/full_model.h5', compile=False, custom_objects={'KerasLayer':hub.KerasLayer})
inceptionv3 = tf.keras.models.load_model('cnn/inceptionv3/'+lr+'/full_model.h5', compile=False, custom_objects={'KerasLayer':hub.KerasLayer})
inceptionv3nat = tf.keras.models.load_model('cnn/inceptionv3nat/'+lr+'/full_model.h5', compile=False, custom_objects={'KerasLayer':hub.KerasLayer})
Xception = tf.keras.models.load_model('cnn/Xception/'+lr+'/full_model.h5', compile=False)

model_cnnA.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
# model_cnnB.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               optimizer=tf.keras.optimizers.Adam(),
#               metrics=['accuracy'])
resnetv2.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
inceptionv3.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
inceptionv3nat.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
Xception.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
#
# inputs = tf.keras.Input(shape=(100, 100, 3))
# y2 = model_cnnB(inputs)
# y4 = mobilenetv2_train(inputs)
# y5 = maximum([y2,y4])
# outputs = tf.keras.layers.Softmax()(y5)
#
# ensemble_model_max = tf.keras.Model(inputs=inputs, outputs=outputs)
#
# ensemble_model_max.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                        optimizer=tf.keras.optimizers.Adam(),
#                        metrics=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),'accuracy'])
#
# inputs = tf.keras.Input(shape=(100, 100, 3))
# y2 = model_cnnB(inputs)
# y4 = mobilenetv2_train(inputs)
# y5 = average([y2,y4])
# outputs = tf.keras.layers.Softmax()(y5)
#
# ensemble_model_avg = tf.keras.Model(inputs=inputs, outputs=outputs)
#
# ensemble_model_avg.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                        optimizer=tf.keras.optimizers.Adam(),
#                        metrics=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),'accuracy'])

def load_dataset(dataset_dir):
    dataset_dir = pathlib.Path(dataset_dir)
    test_image_count2 = len(list(test_data_dir.glob('image\*.jpg')))
    list_ds = tf.data.Dataset.list_files(str(dataset_dir / 'image/*.jpg'))
    for f in list_ds.take(5):
        print(f.numpy())
    labeled_ds = list_ds.map(read_and_label, num_parallel_calls=AUTOTUNE)
    return labeled_ds, test_image_count2

def evalmodels(path):
    datasett, datasettsize = load_dataset(path)
    # with tf.device('/device:GPU:1'):
    #     results = model_cnnA.evaluate(datasett.batch(1000))
    # print(os.path.basename(path), results[-1] * 100)
    results = model_cnnA.evaluate(datasett.batch(1000))
    print(os.path.basename(path), results[-1] * 100)
    # results = model_cnnB.evaluate(datasett.batch(1000))
    # print(os.path.basename(path), results[-1] * 100)
    results = resnetv2.evaluate(datasett.batch(1000))
    print(os.path.basename(path), results[-1] * 100)
    results = inceptionv3.evaluate(datasett.batch(1000))
    print(os.path.basename(path), results[-1] * 100)
    results = inceptionv3nat.evaluate(datasett.batch(1000))
    print(os.path.basename(path), results[-1] * 100)
    results = Xception.evaluate(datasett.batch(1000))
    print(os.path.basename(path), results[-1] * 100)


evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\train\young\sec001')
print('---------------------------------------------------------------------------')
evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\train\young\sec001')
print('---------------------------------------------------------------------------')
evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\train\young\sec001')
print('---------------------------------------------------------------------------')
evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\train\young\sec001')
print('---------------------------------------------------------------------------')


# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\test\young\sec023')
# print('---------------------------------------------------------------------------')
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\test\young\sec025')
# print('---------------------------------------------------------------------------')
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\test\old\sec031')
# print('---------------------------------------------------------------------------')
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\test\old\sec037')
# print('---------------------------------------------------------------------------')
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\test\old\sec045')
# print('---------------------------------------------------------------------------')
