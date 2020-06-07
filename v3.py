import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten, GlobalAveragePooling2D, add, average, \
    maximum
import tensorflow_addons as tfa

from tensorflow_docs import modeling
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import shutil

tfds.disable_progress_bar()  # disable tqdm progress bar
AUTOTUNE = tf.data.experimental.AUTOTUNE

print("TensorFlow Version: ", tf.__version__)
print("Number of GPU available: ", len(tf.config.experimental.list_physical_devices("GPU")))


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
    # image = tf.image.random_hue(image, max_delta=0.01, seed=5)
    # image = tf.image.random_contrast(image, 0.95, 1.05, seed=5)  # tissue quality
    # image = tf.image.random_saturation(image, 0.95, 1.05, seed=5)  # stain quality
    # image = tf.image.random_brightness(image, max_delta=0.01)  # tissue thickness, glass transparency (clean)
    # image = tf.image.random_flip_left_right(image, seed=5)  # cell orientation
    # image = tf.image.random_flip_up_down(image, seed=5)  # cell orientation
    # image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))  # cell orientation
    return image, label


IMG_HEIGHT = 100
IMG_WIDTH = 100
BATCH_SIZE = 64
val_fraction = 30
shuffle_buffer_size = 1000000  # take first 100 from dataset and shuffle and pick one.
samplesize = [300, 400] #old, young
# list location of all training images
train_data_dir = '/home/kuki/Desktop/Research/cnn_dataset/train'
train_data_dir = pathlib.Path(train_data_dir)
CLASS_NAMES = np.array(
    [item.name for item in train_data_dir.glob('*') if item.name != "LICENSE.txt" and item.name != ".DS_store"])


def balance(data_dir):
    tmp = [0]
    for CLASS, n in zip(CLASS_NAMES, samplesize):
        secs = [_ for _ in data_dir.glob(CLASS+'/*')]
        for idx,sec in enumerate(secs):
            sec = os.path.join(sec, 'image/*.jpg')
            list_ds = tf.data.Dataset.list_files(sec)
            # subsample
            list_ds = (list_ds
                       .shuffle(shuffle_buffer_size)
                       .take(n)
                       )
            labeled_ds = list_ds.map(read_and_label, num_parallel_calls=AUTOTUNE)

            # add augment
            sampleN = len(list(labeled_ds))
            while sampleN < n:
                labeled_ds_aug = (labeled_ds
                                  .shuffle(shuffle_buffer_size)
                                  .take(n-sampleN)
                                  .map(augment,num_parallel_calls=AUTOTUNE)
                                  )
                labeled_ds = labeled_ds.concatenate(labeled_ds_aug)
                sampleN = len(list(labeled_ds))
            print('list_ds: ', len(list(labeled_ds)),CLASS)
            # append
            if tmp[0] == 0:
                tmp[idx] = labeled_ds
            else:
                labeled_ds = tmp[0].concatenate(labeled_ds)
                tmp[0] = labeled_ds
        print(CLASS, ': sample size =', len(list(tmp[0])))
    return tmp[0].shuffle(shuffle_buffer_size)


train_labeled_ds = balance(train_data_dir)
train_image_count = len(list(train_labeled_ds))
print('training set size : ', train_image_count)
val_image_count = train_image_count // 100 * val_fraction
print('validation size: ', val_image_count)
train_image_count2 = train_image_count-val_image_count
print('training set size after split : ', train_image_count2)
STEPS_PER_EPOCH = train_image_count2 // BATCH_SIZE
VALIDATION_STEPS = val_image_count // BATCH_SIZE

# plt.figure(figsize=(10,10))
# for idx, elem in enumerate(train_labeled_ds.take(25)):
#     img = elem[0]
#     label = elem[1]
#     ax = plt.subplot(5,5,idx+1)
#     plt.imshow(img)
#     plt.title(CLASS_NAMES[label].title())
#     plt.axis('off')
# plt.show()

train_ds = (train_labeled_ds
            .skip(val_image_count)
            .cache("./cache/fibro_train.tfcache")
            .shuffle(buffer_size=shuffle_buffer_size)
            .repeat()
            .batch(BATCH_SIZE)
            # .map(augment, num_parallel_calls=AUTOTUNE)  # always batch before mapping
            .prefetch(buffer_size=AUTOTUNE)
            )

val_ds = (train_labeled_ds
          .take(val_image_count)
          .cache("./cache/fibro_val.tfcache")
          .repeat()
          .batch(BATCH_SIZE)
          .prefetch(buffer_size=AUTOTUNE))

test_data_dir = '/home/kuki/Desktop/Research/cnn_dataset/test'
test_data_dir = pathlib.Path(test_data_dir)
test_labeled_ds = balance(test_data_dir)

plt.figure(figsize=(10,10))
for idx,elem in enumerate(test_labeled_ds.take(100)):
    img = elem[0]
    label = elem[1]
    ax = plt.subplot(10,10,idx+1)
    plt.imshow(img)
    plt.title(CLASS_NAMES[label].title())
    plt.axis('off')
plt.show()


test_ds = (test_labeled_ds
           .cache("./cache/fibro_test.tfcache")
           .shuffle(buffer_size=shuffle_buffer_size)
           .repeat()
           .batch(BATCH_SIZE)
           .prefetch(buffer_size=AUTOTUNE)  # time it takes to produce next element
           )
test_image_count = len(list(test_labeled_ds))
print('test set size : ', test_image_count)
TEST_STEPS = test_image_count // BATCH_SIZE

checkpoint_dir = "training_1"
# shutil.rmtree(checkpoint_dir, ignore_errors=True)


def get_callbacks(name):
    return [
        modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_crossentropy',
                                         patience=50, restore_best_weights=True),
        # tf.keras.callbacks.TensorBoard(log_dir/name, histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + "/{}/cp.ckpt".format(name),
                                           verbose=0,
                                           monitor='val_sparse_categorical_crossentropy',
                                           save_weights_only=True,
                                           save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_crossentropy',
                                             factor=0.1, patience=10, verbose=0, mode='auto',
                                             min_delta=0.0001, cooldown=0, min_lr=0),
    ]


# lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#     1e-4,
#     decay_steps=STEPS_PER_EPOCH * 100,
#     decay_rate=1,
#     staircase=False)


def compilefit(model, name, opt, max_epochs=1000):
    optimizer = opt
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'accuracy'])
    model_history = model.fit(val_ds,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              epochs=max_epochs,
                              verbose=0,
                              validation_data=test_ds,
                              callbacks=get_callbacks(name),
                              validation_steps=VALIDATION_STEPS,
                              use_multiprocessing=True
                              )
    namename = os.path.dirname(name)
    if not os.path.isdir(os.path.abspath(namename)):
        os.mkdir(os.path.abspath(namename))
    if not os.path.isdir(os.path.abspath(name)):
        os.mkdir(os.path.abspath(name))
    if not os.path.isfile(pathlib.Path(name) / 'full_model.h5'):
        try:
            model.save(pathlib.Path(name) / 'full_model.h5')
        except:
            print('model not saved?')

    return model_history


def plotdf(dfobj, condition, lr=None):
    pd.DataFrame(dfobj).plot(title=condition)
    dfobj.pop('loss')
    dfobj.pop('val_loss')
    dfobj1 = dfobj.copy()
    dfobj2 = dfobj.copy()
    dfobj.pop('lr')
    dfobj.pop('sparse_categorical_crossentropy')
    dfobj.pop('val_sparse_categorical_crossentropy')
    pd.DataFrame(dfobj).plot(title=condition)
    dfobj1.pop('lr')
    dfobj1.pop('accuracy')
    dfobj1.pop('val_accuracy')
    pd.DataFrame(dfobj1).plot(title=condition)
    if lr is not 'decay':
        dfobj2.pop('sparse_categorical_crossentropy')
        dfobj2.pop('val_sparse_categorical_crossentropy')
        dfobj2.pop('accuracy')
        dfobj2.pop('val_accuracy')
        pd.DataFrame(dfobj2).plot(title=condition)
    plt.show()


histories = {}

Alex = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
           , kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    MaxPool2D((2, 2)),
    Dropout(0.5),
    Conv2D(32, (3, 3), activation='relu',
           kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    MaxPool2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu',
           kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    Flatten(),
    Dense(512, activation='relu',
          kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dense(2)
])


inputs = tf.keras.Input(shape=(100, 100, 3), name='img')
x = Conv2D(32, 3, activation='relu', kernel_initializer='he_uniform')(inputs)
x = Conv2D(64, 3, activation='relu', kernel_initializer='he_uniform')(x)
block_1_output = MaxPool2D(3)(x)

x = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(block_1_output)
x = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(x)
block_2_output = add([x, block_1_output])

x = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(block_2_output)
x = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')(x)
block_3_output = add([x, block_2_output])

x = Conv2D(64, 3, activation='relu', kernel_initializer='he_uniform')(block_3_output)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
x = Dropout(0.5)(x)
outputs = Dense(2)(x)

ToyRes = tf.keras.Model(inputs, outputs, name='toy_resnet')

ResV2 = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/4",
                   trainable=True, arguments=dict(batch_norm_momentum=0.99)),  # Can be True, see below.
    tf.keras.layers.Dense(2, activation='softmax')
])
ResV2.build([None, 100, 100, 3])  # Batch input shape.

IncV3 = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4",
                   trainable=True, arguments=dict(batch_norm_momentum=0.99)),  # Can be True, see below.
    tf.keras.layers.Dense(2, activation='softmax')
])
IncV3.build([None, 100, 100, 3])  # Batch input shape.

IncV3n = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/4",
                   trainable=True, arguments=dict(batch_norm_momentum=0.99)),  # Can be True, see below.
    tf.keras.layers.Dense(2, activation='softmax')
])
IncV3n.build([None, 100, 100, 3])  # Batch input shape.

base_model = tf.keras.applications.Xception(input_shape=(100, 100, 3),
                                            pooling='avg',
                                            include_top=False,
                                            weights=None
                                            )
base_model.trainable = True
Xcep = tf.keras.Sequential([
    base_model,
    Dense(2)
])


def evaluateit(network,networkname, opt):
    # with tf.device('/device:GPU:1'):
    histories[networkname+namer] = compilefit(network, 'cnn/'+networkname+'/'+namer, opt, max_epochs=1000)
    results = network.evaluate(test_ds, steps=TEST_STEPS)
    plotdf(histories[networkname+namer].history, networkname + ' lr='+namer)
    print('test acc', results[-1] * 100)


# namers = ['AdamW_e3', 'SGD_e3','RMSprop_e3']
# opts = [tfa.optimizers.AdamW(lrr), tf.keras.optimizers.SGD(lrr),tf.keras.optimizers.RMSprop(lrr)]
lrr=1e-3

namers = ['Adam_t4']
opts = [tf.keras.optimizers.Adam(lrr)]


# for idx,opt in enumerate(opts):
#
#     namer = namers[idx]
#     # evaluateit(Alex,'Alex',opt)
#     # evaluateit(ToyRes,'ToyRes',opt)
#     print(namer)
#     print('ResV2')
#     evaluateit(ResV2,'ResV2',opt)
#     print('IncV3')
#     evaluateit(IncV3,'IncV3',opt)
#     print('IncV3n')
#     evaluateit(IncV3n,'IncV3n',opt)
#     # evaluateit(Xcep,'Xcep',opt)

# print('train_ds ')
# for img,label in train_ds.take(100).as_numpy_iterator():
#     print(label)
# print('val_ds ')
# for img, label in val_ds.take(100).as_numpy_iterator():
#     print(label)
# print('test_ds ')
# for img,label in test_ds.take(100).as_numpy_iterator():
#     print(label)
