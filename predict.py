import tensorflow as tf
import numpy as np
import cv2

# mnist data
class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)      # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)        # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)    # [60000]
        self.test_label = self.test_label.astype(np.int32)      # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]

data_loader = MNISTLoader()
data_test = data_loader.test_data[0:9]

# loading model
model = tf.saved_model.load("./save/2")

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



# convert pic to tensor
img_raw = tf.io.read_file("./trainpic/0/*.jpg")
img_tensor = tf.image.decode_image(img_raw)
img_final = tf.image.resize(img_tensor, [28, 28])
img_final = img_final/255.0
img_final = tf.expand_dims(img_final,axis=0)
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())


# predict
pred_mypic = model(img_final)
print(pred_mypic.numpy())
print("------------------")
final_value = np.argmax(pred_mypic,axis=-1)
print(final_value)
print("/////////////////////////////////////////")
# predict mnist data by my model
pred = model(data_test)
print(pred.numpy())
print("------------------")
final_value = np.argmax(pred,axis=-1)
print(final_value)
print(data_loader.test_label[0:9])