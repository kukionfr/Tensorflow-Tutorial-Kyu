import tensorflow as tf
from tensorflow.keras.layers import average, maximum, minimum
import tensorflow_hub as hub
import os
import numpy as np
import tensorflow_hub as hub
import pathlib

print(tf.__version__)


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


def augment(image, label):
    image = tf.image.random_hue(image, max_delta=0.05, seed=5)
    image = tf.image.random_contrast(image, 0.95, 1.05, seed=5)  # tissue quality
    image = tf.image.random_saturation(image, 0.95, 1.05, seed=5)  # stain quality
    image = tf.image.random_brightness(image, max_delta=0.05)  # tissue thickness, glass transparency (clean)
    image = tf.image.random_flip_left_right(image, seed=5)  # cell orientation
    image = tf.image.random_flip_up_down(image, seed=5)  # cell orientation
    image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))  # cell orientation
    return image, label


IMG_HEIGHT = 100
IMG_WIDTH = 100
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64
shuffle_buffer_size = 1000000

test_data_dir = r'C:\Users\kuki\OneDrive - Johns Hopkins\Research\Skin\RCNN data\torefine'
test_data_dir = pathlib.Path(test_data_dir)
CLASS_NAMES = np.array(
    [item.name for item in test_data_dir.glob('*') if item.name != "LICENSE.txt" and item.name != ".DS_store"])


def load_compile(net, lr):
    model = tf.keras.models.load_model('cnn/' + net + '/' + lr + '/full_model.h5', compile=False,
                                       custom_objects={'KerasLayer': hub.KerasLayer}
                                       )
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model


ResV2 = load_compile('ResV2', 'e3')
IncV3 = load_compile('IncV3', 'e3')
IncV3n = load_compile('IncV3n', 'e3')


def load_dataset(dataset_dir):
    dataset_dir = pathlib.Path(dataset_dir)
    test_image_count2 = len(list(test_data_dir.glob('image/*.jpg')))
    list_ds = tf.data.Dataset.list_files(str(dataset_dir / 'image/*.jpg'))
    # for f in list_ds.take(5):
    #     print(f.numpy())
    labeled_ds = list_ds.map(read_and_label, num_parallel_calls=AUTOTUNE)
    return labeled_ds, test_image_count2


def evalmodels(path, model):
    datasett, datasettsize = load_dataset(path)
    with tf.device('/device:GPU:1'):
        results = model.evaluate(datasett.batch(1000))
    # print(os.path.basename(path), results[-1] * 100)
    aa.append(np.around(results[-1] * 100, decimals=1))


inputs = tf.keras.Input(shape=(100, 100, 3))
y1 = ResV2(inputs)
y2 = IncV3(inputs)
y3 = IncV3n(inputs)
y4 = maximum([y1, y2, y3])
outputs = tf.keras.layers.Softmax()(y4)

ensemble_model_max = tf.keras.Model(inputs=inputs, outputs=outputs)

ensemble_model_max.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'accuracy'])
#
# inputs2 = tf.keras.Input(shape=(100, 100, 3))
# y5 = ResV2(inputs2)
# y6 = IncV3(inputs2)
# y7 = IncV3n(inputs2)
# y8 = average([y5, y6, y7])
# outputs2 = tf.keras.layers.Softmax()(y8)
#
# ensemble_model_avg = tf.keras.Model(inputs=inputs2, outputs=outputs2)
#
# ensemble_model_avg.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                            optimizer=tf.keras.optimizers.Adam(),
#                            metrics=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'accuracy'])
#
# inputs3 = tf.keras.Input(shape=(100, 100, 3))
# y9 = ResV2(inputs3)
# y10 = IncV3(inputs3)
# y11 = IncV3n(inputs3)
# y12 = minimum([y9, y10, y11])
# outputs3 = tf.keras.layers.Softmax()(y12)
#
# ensemble_model_min = tf.keras.Model(inputs=inputs3, outputs=outputs3)
#
# ensemble_model_min.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                            optimizer=tf.keras.optimizers.Adam(),
#                            metrics=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'accuracy'])

model = ensemble_model_max

# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\train\young\sec001',model)
# # print('---------------------------------------------------------------------------')
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\train\young\sec003',model)
# # print('---------------------------------------------------------------------------')
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\train\young\sec007',model)
# # print('---------------------------------------------------------------------------')
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\train\young\sec010',model)
# # print('---------------------------------------------------------------------------')
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\train\young\sec016',model)
# # print('---------------------------------------------------------------------------')
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\train\young\sec019',model)
# # print('---------------------------------------------------------------------------')
#
#
#
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\test\young\sec023',model)
# # print('---------------------------------------------------------------------------')
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\test\young\sec025',model)
# # print('---------------------------------------------------------------------------')
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\test\young\sec029',model)
# # print('---------------------------------------------------------------------------')
#
#
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\train\old\sec031',model)
# # print('---------------------------------------------------------------------------')
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\train\old\sec037',model)
# # print('---------------------------------------------------------------------------')
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\train\old\sec041',model)
# # print('---------------------------------------------------------------------------')
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\train\old\sec045',model)
# # print('---------------------------------------------------------------------------')
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\train\old\sec049',model)
# # print('---------------------------------------------------------------------------')
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\train\old\sec062',model)
# # print('---------------------------------------------------------------------------')
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\train\old\sec068',model)
# # print('---------------------------------------------------------------------------')
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\train\old\sec070',model)
# # print('---------------------------------------------------------------------------')
#
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\test\old\sec076',model)
# # print('---------------------------------------------------------------------------')
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\test\old\sec078',model)
# # print('---------------------------------------------------------------------------')
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\test\old\sec082',model)
# # print('---------------------------------------------------------------------------')
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\test\old\sec088',model)
# # print('---------------------------------------------------------------------------')

aa = []
evalmodels(r'C:\Users\kuki\OneDrive - Johns Hopkins\Research\Skin\RCNN data\torefine\young\sec001', model)
evalmodels(r'C:\Users\kuki\OneDrive - Johns Hopkins\Research\Skin\RCNN data\torefine\young\sec003', model)
evalmodels(r'C:\Users\kuki\OneDrive - Johns Hopkins\Research\Skin\RCNN data\torefine\young\sec007', model)
print(aa)
