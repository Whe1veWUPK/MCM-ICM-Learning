import numpy as np
 
class MaxPool2:
    '''
    `2*2`范围的`最大池化`
    '''
 
    def iterate_regions(self, image):
        '''
        生成`2*2`的池化区域\n
        Generates non-overlapping 2x2 image regions to pool over.\n
        Parameters
        ---
        - image is a 2d numpy array\n
        Returns
        ---
        生成器\n
        (im_region, i, j)\n
        - im_region: 二维NumPy数组，图片上当前迭代到的`2*2`区域\n
        - i: 输出的`行下标`\n
        - j: 输出的`列下标`\n  
        '''
        # image: 26x26x8
        h, w, _ = image.shape # '_'是用来接收第三个维度的，可能有多个卷积核
        new_h = h // 2
        new_w = w // 2
 
        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j
 
    def forward(self, input):
        '''
        Performs a forward pass of the maxpool layer using the given input.
        Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
        - input is a 3d numpy array with dimensions (h, w, num_filters)
        '''
        # 存储输入用于BP
        self.last_input = input

        # input: 卷积层的输出，池化层的输入
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))
 
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))
        return output
    
    def backprop(self, d_L_d_out):
        '''
        Performs a backward pass of the maxpool layer.
        Returns the loss gradient for this layer's inputs.
        - d_L_d_out is the loss gradient for this layer's outputs.
        '''
        # 池化层输入数据，26x26x8，默认初始化为 0
        d_L_d_input = np.zeros(self.last_input.shape)
 
        # 每一个 im_region 都是一个 3x3x8 的8层小矩阵
        # 修改 max 的部分，首先查找 max
        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            # 获取 im_region 里面最大值的索引向量，一叠的感觉
            amax = np.amax(im_region, axis=(0, 1))
 
            # 遍历整个 im_region，对于传递下去的像素点，修改 gradient 为 loss 对 output 的gradient
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If this pixel was the max value, copy the gradient to it. 修改最大值处的梯度值
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]
 
        return d_L_d_input