import numpy as np
 
class Conv3x3:
    '''
    使用`3*3`卷积核的卷积层实现\n
    Attributes
    ------------
    - num_filters: 卷积核的个数
    '''
 
    def __init__(self, num_filters):
        '''
        初始化\n
        参数：
        -----------
        - num_filters: 过滤器（卷积核）的个数
        '''
        self.num_filters = num_filters
 
        # filters is a 3d array with dimensions (num_filters, 3, 3)
        # We divide by 9 to reduce the variance of our initial values
        self.filters = np.random.randn(num_filters, 3, 3) / 9
    
    def iterate_regions(self, image):
        '''
        Generates all possible 3x3 image regions using valid padding.\n
        Parameters:
        -------------
        - image: a 2d numpy array\n
        Returns:
        ------------
        im_region, i, j\n
        - im_region: 图片上当前迭代到的3*3区域\n
        - i: 输出的`行下标`\n
        - j: 输出的`列下标`\n
        Notes:
        ----------
        这是一个`生成器函数`，它通过yield关键字将每次迭代的值返回给接收它的变量，\n
        使用`next(iterate_regions)`来获取下一次迭代的值，\n
        使用`for element in iterate_regions(your_image)`来遍历每次迭代的值.
        '''
        h, w = image.shape # 图片的长、宽
 
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j
        # 将 im_region, i, j 以 tuple 形式存储到迭代器中
        # 以便后面遍历使用
 
    def forward(self, input):
        '''
        前向传播\n
        Parameters:
        -----
        - input: a 2d numpy array.\n
        Returns:
        -----
        - a 3d numpy array with dimensions (h, w, num_filters)
        '''
        # 存储输入用于BP
        self.last_input = input

        # input 为 image，即输入数据
        # output 为输出框架，默认都为 0，都为 1 也可以，反正后面会覆盖
        # input: h*w
        # output: (h-2)*(w-2)*self.num_filters -2是因为卷积核是3*3，卷积后矩阵大小会变小
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))
 
        for im_region, i, j in self.iterate_regions(input):# 调用生成器的值
            # 卷积运算，点乘再相加，ouput[i, j] 为向量，8 层 可以理解为维度1和维度2相同的项相加，结果为1维数组（保留的是维度0）
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))# axis表示沿着这些维度下标变化的方向操作 从0开始
        # 最后将输出数据返回，便于下一层的输入使用
        return output
    
    def backprop(self, d_L_d_out, learn_rate):
        '''
        Performs a backward pass of the conv layer.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        '''
        # 初始化一组为 0 的 gradient，3x3x8
        d_L_d_filters = np.zeros(self.filters.shape)
 
        # im_region，一个个 3x3 小矩阵
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                # 按 f 分层计算，一次算一层，然后累加起来
                # d_L_d_filters[f]: 3x3 matrix
                # d_L_d_out[i, j, f]: num
                # im_region: 3x3 matrix in image
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
 
        # Update filters
        self.filters -= learn_rate * d_L_d_filters
 
        # We aren't returning anything here since we use Conv3x3 as
        # the first layer in our CNN. Otherwise, we'd need to return
        # the loss gradient for this layer's inputs, just like every
        # other layer in our CNN.
        return None