import numpy as np
 
class Softmax:
    # A standard fully-connected layer with softmax activation.
 
    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of our initial values
        # input_len: 输入层的节点个数，池化层输出拉平之后的
        # nodes: 输出层的节点个数，这是最后分类的个数
        # 构建权重矩阵，初始化随机数，不能太大
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)
 
    def forward(self, input):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions.
        '''
        # 记录本次input各维度的长度，用于BP
        self.last_input_shape = input.shape
        
        # 3d to 1d，用来构建全连接网络
        input = input.flatten()# 将数据扁平化为1维数组

        # 记录本次input扁平化后的结果，用于BP
        self.last_input = input
 
        input_len, nodes = self.weights.shape
 
        # input: 13x13x8 = 1352
        # self.weights: (1352, 10)
        # 以上叉乘之后为 向量，1352个节点与对应的权重相乘再加上bias得到输出的节点
        # totals: 向量, 10
        totals = np.dot(input, self.weights) + self.biases # y = wx + b

        # 记录本次的y值，用于BP
        self.last_totals = totals

        # exp: 向量, 10
        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0) # softmax激活
    
    def backprop(self, d_L_d_out, learn_rate=0.01):
        '''
        Performs a backward pass of the softmax layer.\n
        Parameters
        ---
        - d_L_d_out is the loss gradient for this layer's outputs.\n
        - learn_rate is a float\n
        Returns
        ---
        Returns the loss gradient for this layer's inputs.
        '''
        # We know only 1 element of d_L_d_out will be nonzero
        for i, gradient in enumerate(d_L_d_out):
            # 找到 label 的值，就是 gradient 不为 0 的
            if gradient == 0:
                continue
 
            # e^totals
            t_exp = np.exp(self.last_totals)
 
            # Sum of all e^totals
            S = np.sum(t_exp)
 
            # Gradients of out[i] against totals
            # 初始化都设置为 非 c 的值
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)#d(out)/dt
            # 单独修改 c 的值，只有这一行值不相同
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of totals against weights/biases/input
            # d_t_d_w 的结果是 softmax 层的输入数据，1352 个元素的向量
            # 不是最终的结果，最终结果是 2d 矩阵，1352x10
            d_t_d_w = self.last_input
            d_t_d_b = 1
            # d_t_d_input 的结果是 weights 值，2d 矩阵，1352x10
            d_t_d_inputs = self.weights
 
            # Gradients of loss against totals
            # 向量，10
            d_L_d_t = gradient * d_out_d_t
 
            # Gradients of loss against weights/biases/input
            # np.newaxis 可以帮助一维向量变成二维矩阵
            # (1352, 1) @ (1, 10) to (1352, 10) @表示向量乘积
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]# .T表示矩阵或向量的转置
            d_L_d_b = d_L_d_t * d_t_d_b
            # (1352, 10) @ (10, 1) to (1352, 1)
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # Update weights / biases 梯度下降
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b
            # 将矩阵从 1d 转为 3d
            # 1352 to 13x13x8
            return d_L_d_inputs.reshape(self.last_input_shape)