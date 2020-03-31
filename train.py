import tensorflow as tf
import numpy as np 
import time

# loading picture
import pathlib
data_root = pathlib.Path('./data/trainpic/')
print(data_root)
print("----------------")
import random
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print(image_count)


def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)

# 元组被解压缩到映射函数的位置参数中
def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

# covert pic to tensor
def opera(path):
    img_raw = tf.io.read_file(path)
    img_tensor = tf.image.decode_image(img_raw)
    img_final = tf.image.resize(img_tensor, [28, 28])
    img_final = img_final/255.0
    img_final = tf.expand_dims(img_final,axis=0)
    return img_final

# make up lables one to one 
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

# setting model CNN
class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,             # 卷积层神经元（卷积核）数目
            kernel_size=[5, 5],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            activation=tf.nn.relu   # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=11)

    def call(self, inputs):
        x = self.conv1(inputs)                  # [batch_size, 28, 28, 32]
        x = self.pool1(x)                       # [batch_size, 14, 14, 32]
        x = self.conv2(x)                       # [batch_size, 14, 14, 64]
        x = self.pool2(x)                       # [batch_size, 7, 7, 64]
        x = self.flatten(x)                     # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)                      # [batch_size, 1024]
        x = self.dense2(x)                      # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output

# start = time.time()
start = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
print(start)

# start to training
learning_rate = 0.001
model = CNN()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

for i in range(0,image_count):
    print("step---"+str(i)+"--")
    X=all_image_paths[i]
    print(X)
    y=all_image_labels[i]
    print(y)
    with tf.GradientTape() as tape:
        y_pred = model(opera(X))
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("loss %f" % (loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))


sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
y_pred = model.predict(opera(all_image_paths[1050]))
sparse_categorical_accuracy.update_state(y_true=all_image_labels[1050], y_pred=y_pred)
print("test accuracy: %f" % sparse_categorical_accuracy.result())

end = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
total = int(end) - int(start)
print(total)

# saved model
tf.saved_model.save(model, "./save/2/")
