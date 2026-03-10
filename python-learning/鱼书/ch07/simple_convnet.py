# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 父目录的文件导入的设置
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient

class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1}, 
                 hidden_size=100, 
                 output_size=10, 
                 weight_init_std=0.01):
        '''
        初始化卷积神经网络
        input_dim: 输入数据的维度
        conv_param: 卷积层的参数
        hidden_size: 隐藏层的神经元数量
        output_size: 输出层的神经元数量
        weight_init_std: 权重初始化标准差
        '''
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        # 计算池化层的输出大小,这里假设池化层是2x2的窗口，步长为2,所以输出大小为conv_output_size/2
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
        
        self.params = {}
        # 初始化权重,权重的形状为(filter_num, input_dim[0], filter_size, filter_size)，其中filter_num是卷积核的数量，input_dim[0]是输入的通道数，filter_size是卷积核的大小
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():    # 遍历所有层，进行前向传播
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """计算交叉熵损失

        前向传播得到预测值 y，再通过 SoftmaxWithLoss 层计算与真实标签 t 的交叉熵。
        训练时每次迭代都会调用此函数。

        Parameters
        ----------
        x : ndarray
            输入数据，形状 (N, 1, 28, 28)
        t : ndarray
            标签，one-hot 或类别索引

        Returns
        -------
        float
            标量损失值
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        """计算分类准确率

        为避免一次性加载全部数据导致内存溢出，按 batch 分批预测并累加正确数。
        支持 one-hot 和类别索引两种标签格式。

        Parameters
        ----------
        x : ndarray
            输入数据
        t : ndarray
            标签，若为 one-hot 则自动转为类别索引
        batch_size : int
            每批样本数，默认 100

        Returns
        -------
        float
            正确预测数 / 总样本数，范围 [0, 1]
        """
        if t.ndim != 1:
            t = np.argmax(t, axis=1)  # one-hot 转类别索引
        acc = 0.0
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = self.predict(x_batch)
            y_batch = np.argmax(y_batch, axis=1)  # 取概率最大的类别
            acc += np.sum(y_batch == t[i:i+batch_size])
        return acc / len(x)

    def numerical_gradient(self, x, t):
        """数值梯度法计算梯度（用于验证反向传播是否正确）

        对每个参数做微小扰动，用 (f(x+h)-f(x-h))/(2h) 近似导数。
        计算慢但实现简单，常用于梯度检查，实际训练应使用 gradient()。

        Parameters
        ----------
        x : ndarray
            输入数据
        t : ndarray
            标签

        Returns
        -------
        dict
            各参数对应的梯度，键为 'W1','b1','W2','b2','W3','b3'
        """
        loss_w = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_w, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_w, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_w, self.params['b2'])
        grads['W3'] = numerical_gradient(loss_w, self.params['W3'])
        grads['b3'] = numerical_gradient(loss_w, self.params['b3'])
        return grads

    def gradient(self, x, t):
        """反向传播法计算梯度（训练时使用，速度快）

        流程：1) 前向传播计算 loss；2) 从 loss 层开始反向传播；
        3) 各层计算并保存 dW、db；4) 从各层取出梯度汇总返回。

        Parameters
        ----------
        x : ndarray
            输入数据
        t : ndarray
            标签

        Returns
        -------
        dict
            各参数的梯度
        """
        self.loss(x, t)  # 前向传播，各层会保存中间结果供反向传播使用
        dout = 1  # 损失对自身的梯度为 1
        dout = self.last_layer.backward(dout)  # SoftmaxWithLoss 反向传播
        layers = list(self.layers.values())
        layers.reverse()  # 反向传播需从最后一层到第一层
        for layer in layers:
            dout = layer.backward(dout)  # 每层接收上游梯度，输出对输入的梯度
        grads = {}
        # 从各层取出已计算好的权重梯度（只有 Conv、Affine 有参数）
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db
        return grads

    def save_params(self, file_name='params.pkl'):
        """将模型参数保存到 pickle 文件

        保存 W1,b1,W2,b2,W3,b3，用于断点续训或模型部署。

        Parameters
        ----------
        file_name : str
            保存路径，默认 params.pkl
        """
        params = {}
        for key, value in self.params.items():
            params[key] = value
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name='params.pkl'):
        """从 pickle 文件加载模型参数

        加载后需同步更新各层的 W、b 引用，因为 layers 中的层持有的是
        self.params 的引用，直接修改 params 即可；但为保险起见，这里
        显式地重新赋给各层的 W 和 b。

        Parameters
        ----------
        file_name : str
            参数文件路径，默认 params.pkl
        """
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, value in params.items():
            self.params[key] = value
        # 将加载的权重同步到各层（Conv1、Affine1、Affine2 有参数）
        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]