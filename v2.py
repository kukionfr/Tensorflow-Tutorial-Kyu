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
BATCH_SIZE = 32
val_fraction = 10
# list location of all training images
data_dir = r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\train'
data_dir = pathlib.Path(data_dir)
train_image_count = len(list(data_dir.glob('*\*\image\*.jpg')))
CLASS_NAMES = np.array(
    [item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt" and item.name != ".DS_store"])
list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*/image/*'))
AUTOTUNE = tf.data.experimental.AUTOTUNE
labeled_ds = list_ds.map(read_and_label, num_parallel_calls=AUTOTUNE)
# plt.figure(figsize=(10,10))
# for idx,elem in enumerate(labeled_ds.take(25)):
#     img = elem[0]
#     label = elem[1]
#     ax = plt.subplot(5,5,idx+1)
#     plt.imshow(img)
#     plt.title(CLASS_NAMES[label].title())
#     plt.axis('off')


test_data_dir = r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\test'
test_data_dir = pathlib.Path(test_data_dir)
test_image_count = len(list(test_data_dir.glob('*\*\image\*.jpg')))
test_list_ds = tf.data.Dataset.list_files(str(test_data_dir / '*\*\image\*'))
test_labeled_ds = test_list_ds.map(read_and_label, num_parallel_calls=AUTOTUNE)

val_image_count = test_image_count // 100 * val_fraction  # // : floor division ex) 15/2 = 7.5 -> 7
STEPS_PER_EPOCH = train_image_count // BATCH_SIZE
TEST_STEPS = test_image_count // BATCH_SIZE
VALIDATION_STEPS = val_image_count // BATCH_SIZE

shuffle_buffer_size = 3000  # take first 100 from dataset and shuffle and pick one.
train_ds = (labeled_ds
            #             .skip(val_image_count)
            .cache("./cache/fibro_train.tfcache")
            .shuffle(buffer_size=shuffle_buffer_size)
            .repeat()
            .batch(BATCH_SIZE)
            .map(augment, num_parallel_calls=AUTOTUNE)  # always batch before mapping
            .prefetch(buffer_size=AUTOTUNE)
            )

# no shuffle, augment for validation and test dataset
val_ds = (test_labeled_ds
          .shuffle(buffer_size=shuffle_buffer_size)
          .take(val_image_count)
          .cache("./cache/fibro_val.tfcache")
          .repeat()
          .batch(BATCH_SIZE)
          .prefetch(buffer_size=AUTOTUNE))

test_ds = (test_labeled_ds
           .cache("./cache/fibro_test.tfcache")
           .repeat()
           .batch(BATCH_SIZE)
           .prefetch(buffer_size=AUTOTUNE)  # time it takes to produce next element
           )

checkpoint_dir = "training_1"
shutil.rmtree(checkpoint_dir, ignore_errors=True)


def get_callbacks(name):
    return [
        modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_crossentropy',
                                         patience=100, restore_best_weights=True),
        #     tf.keras.callbacks.TensorBoard(log_dir/name, histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + "/{}/cp.ckpt".format(name),
                                           verbose=0,
                                           monitor='val_sparse_categorical_crossentropy',
                                           save_weights_only=True,
                                           save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_crossentropy',
                                             factor=0.1, patience=50, verbose=0, mode='auto',
                                             min_delta=0.0001, cooldown=0, min_lr=0),
    ]


lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    1e-3,
    decay_steps=STEPS_PER_EPOCH * 100,
    decay_rate=1,
    staircase=False)


def compilefit(model, name, lr, max_epochs=1000):
    optimizer = tfa.optimizers.AdamW(lr)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'accuracy'])
    model_history = model.fit(train_ds,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              epochs=max_epochs,
                              verbose=0,
                              validation_data=val_ds,
                              callbacks=get_callbacks(name),
                              validation_steps=VALIDATION_STEPS,
                              use_multiprocessing=True
                              )
    namename = os.path.dirname(name)
    if not os.path.isdir(os.path.abspath(namename)):
        os.mkdir(os.path.abspath(namename))
    if not os.path.isdir(os.path.abspath(name)):
        os.mkdir(os.path.abspath(name))
    model.save(pathlib.Path(name) / 'full_model.h5')
    return model_history


def plotdf(dfobj, condition, lr=None):
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
'''
namel = ['e5','e4','e3','e2']
lrl = [1e-5,1e-4,1e-3,1e-2]
for namer,lrr in zip(namel,lrl):
    model_cnnA = Sequential([
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

    model_cnnB = tf.keras.Model(inputs, outputs, name='toy_resnet')

    resnetv2 = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/4",
                       trainable=True, arguments=dict(batch_norm_momentum=0.99)),  # Can be True, see below.
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    resnetv2.build([None, 100, 100, 3])  # Batch input shape.

    inceptionv3nat = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/4",
                       trainable=True, arguments=dict(batch_norm_momentum=0.99)),  # Can be True, see below.
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    inceptionv3nat.build([None, 100, 100, 3])  # Batch input shape.

    inceptionv3 = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4",
                       trainable=True, arguments=dict(batch_norm_momentum=0.99)),  # Can be True, see below.
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    inceptionv3.build([None, 100, 100, 3])  # Batch input shape.

    base_model = tf.keras.applications.Xception(input_shape=(100, 100, 3),
                                                pooling='avg',
                                                include_top=False,
                                                weights=None
                                                )
    base_model.trainable = True
    Xception = tf.keras.Sequential([
        base_model,
        Dense(2)
    ])


    histories['cnnA'+namer] = compilefit(model_cnnA, 'cnn/A/'+namer, lr = lrr, max_epochs=1000)
    plotdf(histories['cnnA'+namer].history, '7 layer CNN with initializer&normalizer '+namer)
    results = model_cnnA.evaluate(test_ds, steps=TEST_STEPS)
    print('test loss, test acc:', results)

    histories['cnnB'+namer] = compilefit(model_cnnB, 'cnn/B/'+namer, lr = lrr, max_epochs=1000)
    plotdf(histories['cnnB'+namer].history, 'toy resnet '+namer)
    results = model_cnnB.evaluate(test_ds, steps=TEST_STEPS)
    print('test loss, test acc:', results)

    histories['resnetv2'+namer] = compilefit(resnetv2, 'cnn/resnetv2/'+namer, lr = lrr, max_epochs=1000)
    plotdf(histories['resnetv2'+namer].history, 'resnetv2 '+namer)
    results = resnetv2.evaluate(test_ds, steps=TEST_STEPS)
    print('test loss, test acc:', results)

    histories['inceptionv3nat'+namer] = compilefit(resnetv2, 'cnn/inceptionv3nat/'+namer, lr = lrr, max_epochs=1000)
    plotdf(histories['inceptionv3nat'+namer].history, 'inceptionv3nat '+namer)
    results = resnetv2.evaluate(test_ds, steps=TEST_STEPS)
    print('test loss, test acc:', results)

    histories['inceptionv3'+namer] = compilefit(resnetv2, 'cnn/inceptionv3/'+namer , lr = lrr, max_epochs=1000)
    plotdf(histories['inceptionv3'+namer].history, 'inceptionv3 '+namer)
    results = resnetv2.evaluate(test_ds, steps=TEST_STEPS)
    print('test loss, test acc:', results)

    histories['Xception'+namer] = compilefit(Xception, 'cnn/Xception/'+namer, lr = lrr, max_epochs=1000)
    plotdf(histories['Xception'+namer].history, 'Xception '+namer)
    results = Xception.evaluate(test_ds, steps=TEST_STEPS)
    print('test loss, test acc:', results)
    del model_cnnA, model_cnnB, resnetv2, inceptionv3nat, inceptionv3, Xception
'''


model_cnnA = Sequential([
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

model_cnnB = tf.keras.Model(inputs, outputs, name='toy_resnet')

resnetv2 = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/4",
                   trainable=True, arguments=dict(batch_norm_momentum=0.99)),  # Can be True, see below.
    tf.keras.layers.Dense(2, activation='softmax')
])
resnetv2.build([None, 100, 100, 3])  # Batch input shape.

inceptionv3nat = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/4",
                   trainable=True, arguments=dict(batch_norm_momentum=0.99)),  # Can be True, see below.
    tf.keras.layers.Dense(2, activation='softmax')
])
inceptionv3nat.build([None, 100, 100, 3])  # Batch input shape.

inceptionv3 = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4",
                   trainable=True, arguments=dict(batch_norm_momentum=0.99)),  # Can be True, see below.
    tf.keras.layers.Dense(2, activation='softmax')
])
inceptionv3.build([None, 100, 100, 3])  # Batch input shape.

base_model = tf.keras.applications.Xception(input_shape=(100, 100, 3),
                                            pooling='avg',
                                            include_top=False,
                                            weights=None
                                            )
base_model.trainable = True
Xception = tf.keras.Sequential([
    base_model,
    Dense(2)
])

namer = 'AdamW'
histories['cnnA' + namer] = compilefit(model_cnnA, 'cnn/A/' + namer, lr=0.001, max_epochs=1000)
plotdf(histories['cnnA' + namer].history, '7 layer CNN with initializer&normalizer ' + namer)
results = model_cnnA.evaluate(test_ds, steps=TEST_STEPS)
print('test loss, test acc:', results)

histories['cnnB' + namer] = compilefit(model_cnnB, 'cnn/B/ ' + namer, lr=0.001, max_epochs=1000)
plotdf(histories['cnnB' + namer].history, 'toy resnet ' + namer)
results = model_cnnB.evaluate(test_ds, steps=TEST_STEPS)
print('test loss, test acc:', results)

histories['resnetv2' + namer] = compilefit(resnetv2, 'cnn/resnetv2/' + namer, lr=0.001, max_epochs=1000)
plotdf(histories['resnetv2' + namer].history, 'resnetv2 ' + namer)
results = resnetv2.evaluate(test_ds, steps=TEST_STEPS)
print('test loss, test acc:', results)

histories['inceptionv3nat' + namer] = compilefit(resnetv2, 'cnn/inceptionv3nat/' + namer, lr=0.001,
                                                 max_epochs=1000)
plotdf(histories['inceptionv3nat' + namer].history, 'inceptionv3nat ' + namer)
results = resnetv2.evaluate(test_ds, steps=TEST_STEPS)
print('test loss, test acc:', results)

histories['inceptionv3' + namer] = compilefit(resnetv2, 'cnn/inceptionv3/' + namer, lr=0.001, max_epochs=1000)
plotdf(histories['inceptionv3' + namer].history, 'inceptionv3 ' + namer)
results = resnetv2.evaluate(test_ds, steps=TEST_STEPS)
print('test loss, test acc:', results)

histories['Xception' + namer] = compilefit(Xception, 'cnn/Xception/' + namer, lr=0.001, max_epochs=1000)
plotdf(histories['Xception' + namer].history, 'Xception ' + namer)
results = Xception.evaluate(test_ds, steps=TEST_STEPS)
print('test loss, test acc:', results)

# model_cnnA = tf.keras.models.load_model('cnn/A/full_model.h5', compile=False)
# model_cnnB = tf.keras.models.load_model('cnn/B/full_model.h5', compile=False)
# mobilenetv2 = tf.keras.models.load_model('cnn/mobilenetv2/full_model.h5', compile=False, custom_objects={'KerasLayer':hub.KerasLayer})
# mobilenetv2_train = tf.keras.models.load_model('cnn/mobilenetv2_train/full_model.h5', compile=False, custom_objects={'KerasLayer':hub.KerasLayer})
#
# model_cnnA.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               optimizer=tf.keras.optimizers.Adam(),
#               metrics=['accuracy'])
# model_cnnB.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               optimizer=tf.keras.optimizers.Adam(),
#               metrics=['accuracy'])
# mobilenetv2.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               optimizer=tf.keras.optimizers.Adam(),
#               metrics=['accuracy'])
# mobilenetv2_train.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               optimizer=tf.keras.optimizers.Adam(),
#               metrics=['accuracy'])
#
# testacc = model_cnnA.evaluate(test_ds, steps=TEST_STEPS)
# print(testacc)
# testacc =model_cnnB.evaluate(test_ds, steps=TEST_STEPS)
# print(testacc)
# testacc =mobilenetv2.evaluate(test_ds, steps=TEST_STEPS)
# print(testacc)
# testacc =mobilenetv2_train.evaluate(test_ds, steps=TEST_STEPS)
# print(testacc)
#
# inputs = tf.keras.Input(shape=(100, 100, 3))
# # y1 = model_cnnA(inputs)
# y2 = model_cnnB(inputs)
# # y3 = mobilenetv2(inputs)
# y4 = mobilenetv2_train(inputs)
# # y5 = average([y2, y4])  # choose models to ensemble
# y5 = maximum([y2,y4])
# outputs = tf.keras.layers.Softmax()(y5)
#
# ensemble_model_avg = tf.keras.Model(inputs=inputs, outputs=outputs)
#
# ensemble_model_avg.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                        optimizer=tf.keras.optimizers.Adam(),
#                        metrics=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),'accuracy'])
# results = ensemble_model_avg.evaluate(test_ds, steps=TEST_STEPS)
# print('test loss, test acc:', results)
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
# results = ensemble_model_max.evaluate(test_ds, steps=TEST_STEPS)
# print('test loss, test acc:', results)
#
#
#
#
# def load_dataset(dataset_dir):
#     dataset_dir = pathlib.Path(dataset_dir)
#     test_image_count2 = len(list(test_data_dir.glob('image\*.jpg')))
#     list_ds = tf.data.Dataset.list_files(str(dataset_dir / 'image/*.jpg'))
#     for f in list_ds.take(5):
#         print(f.numpy())
#     labeled_ds = list_ds.map(read_and_label, num_parallel_calls=AUTOTUNE)
#     return labeled_ds, test_image_count2
#
# def evalmodels(path):
#     datasett, datasettsize = load_dataset(path)
#     results = model_cnnA.evaluate(datasett.batch(10000))
#     print(os.path.basename(path), results[-1] * 100)
#     results = model_cnnB.evaluate(datasett.batch(10000))
#     print(os.path.basename(path), results[-1] * 100)
#     results = mobilenetv2.evaluate(datasett.batch(10000))
#     print(os.path.basename(path), results[-1] * 100)
#     results = mobilenetv2_train.evaluate(datasett.batch(10000))
#     print(os.path.basename(path), results[-1] * 100)
#     results = ensemble_model_avg.evaluate(datasett.batch(10000))
#     print(os.path.basename(path), results[-1] * 100)
#     results = ensemble_model_max.evaluate(datasett.batch(10000))
#     print(os.path.basename(path), results[-1] * 100)
#
# evalmodels(r'C:\Users\kuki\Desktop\Research\Skin\RCNN data\test\young\sec001')
