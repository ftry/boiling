

![](https://pic2.zhimg.com/v2-56be56e21e223436aee1f1fe75bd1efb_1200x500.gif)

# $BOILING$

## TensorFlow2.0 本地印刷体数字０～９识别

- 几点说明:

  - 官方和许多教程的案例都是利用Mnist手写数字的识别。想要迁移到训练自己本地的图片时，颇费周折，对于初学者来说。这一两日我摸索出了迁移的方法。故，放于github。
  - 我的问题是：如何将本地图片数据喂入模型里训练。
  - 最好的理解和运用Tensorflow，还是多多专注查阅官方的文档和官方推荐。如：[《简单粗暴的TensorFlow 2》](https://tf.wiki/index.html)
  - 本案例是网上各种资料和本人自己对tf2的理解整合而来。

- 对Tensorflow2.0的一些认识:

  - Tensor张量是其基本的运算单位，可以理解为矩阵。我们在做文字数字识别的模型时，如何将本地数据喂入模型训练，关键就是将图片数据利用tf2转换成Tensor张量。
  - 模型就是函数。如：$y=ax+b$就可以说成是一个模型，不过是在tf2里面的模型函数要复杂得多，且是关于矩阵的函数。定义模型参考官方的[文档](https://www.tensorflow.org)。我认为初学者应先按自己的想法跑通自己的模型，这样可以收获大大的成就感，然后就会有巨大的动力去深入学习，再自定义模型。
  - 官方的案例按批次多张图片喂入模型训练。我这里是单张图片喂入模型训练，效果和官方的作用一致。
  - 喂入模型的图片和标签要一一对应。具体的操作方法官方文档tf.data有案例[说明](https://www.tensorflow.org/tutorials/load_data/images?hl=zh-cn)
  - tf2里有许多内置的函数和张量操作方法需要在实际操作中去摸索理解。

- 目前实现:

  - 单个印刷体数字0到９和一个字母A的识别。
  - 图片数据转化为Tensor张量。
  - 模型的保存与加载，这个在TF2里变得十分简单。

- 下一步:

  实现连续的数字字母识别。

  > 这里应该是先要对数字字母定位，然后切割，然后识别。不知道还有没有其他方法来识别连续的数字字母。

- 数据集地址:

   [The Chars74K dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/)

- 实现环境:

  Ubuntu/Centos等Linux系统、Python3+、GPU\CPU(TensorFlow2.０需要指定CPU或GPU)

- 使用方法:

  - Python train.py 训练模型并保存
  - Python predict.py 加载模型并预测

## License

[Apache License 2.0]( http://www.apache.org/licenses/LICENSE-2.0)

------

@ 山谷编辑

&copy; $\pi zy$



