# 支持向量机

## 原理

详细可参考：https://zhuanlan.zhihu.com/p/77750026 讲的很清楚

它将一组**线性可分**的点用一个超平面正确划分，是一个**二分类**模型。如果超平面的方程是

$$
w^{T}x+ b= 0
$$

其中，w是一个n维列向量（待求参数），x是一个n维列向量（点的坐标），用x_i表示它的每个分量，b是一个实数，我们需要求得

$$
min\frac{1}{2} ||w||^{2} \space \space \space \space \space \space \space \space \space \space \space \space s.t. \space \space \space \space y_{i}(w^{T}x_{i}+ b)\ge 1
$$

即，使得这个超平面到所有点的**最小距离**最大（x和y可以分别理解成已知数据的输入和输出，它们相当于样本集；等式的左边理解成两侧支持向量的距离的倒数）。对于这一问题的转化在上面的帖子里有详细说明。

在实际场景中，严格线性可分的情况很少很少，因此引入了软间隔，即允许部分样本点出现在超平面附近的一个间隔带中（不满足约束条件）。为了实现这一点，为每个变量引入了一个**松弛变量**，它代表偏离约束条件的程度（原文4.1下有图）。

$$
\xi _{i}= max(0,1-y_{i}(w\cdot x_{i}+ b))
$$

## 算法
**输入**：训练数据集

$$
T= \{(x_{1},y_{1}),\cdots,(x_{n},y_{n})\}, \space  x_{i}\in R^{m}, y_{i}\in \{+1,-1\},i=1,2,\cdots ,n
$$

**输出**：超平面的方程及分类决策函数

**方法**：

1. 确定惩罚参数C>0，它是一个常数，数值越大表示对错误样本的惩罚越大，然后构造如下凸二次规划问题：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><munder><mrow><mi>m</mi><mi>a</mi><mi>x</mi></mrow><mi>λ</mi></munder><mo stretchy="false">[</mo><munderover><mo data-mjx-texclass="OP">∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>n</mi></mrow></munderover><msub><mi>λ</mi><mrow><mi>i</mi></mrow></msub><mo>−</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><munderover><mo data-mjx-texclass="OP">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>n</mi></mrow></munderover><munderover><mo data-mjx-texclass="OP">∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>n</mi></mrow></munderover><msub><mi>λ</mi><mrow><mi>i</mi></mrow></msub><msub><mi>λ</mi><mrow><mi>j</mi></mrow></msub><mo stretchy="false">(</mo><msub><mi>x</mi><mrow><mi>i</mi></mrow></msub><mo>⋅</mo><msub><mi>x</mi><mrow><mi>j</mi></mrow></msub><mo stretchy="false">)</mo><mo stretchy="false">]</mo></math>

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>s</mi><mo>.</mo><mi>t</mi><mo>.</mo><mtext>&nbsp;</mtext><mtext>&nbsp;</mtext><munderover><mo data-mjx-texclass="OP">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>n</mi></mrow></munderover><msub><mi>λ</mi><mrow><mi>i</mi></mrow></msub><msub><mi>y</mi><mrow><mi>i</mi></mrow></msub><mo>=</mo><mn>0</mn><mo>,</mo><mtext>&nbsp;</mtext><mtext>&nbsp;</mtext><msub><mi>λ</mi><mrow><mi>i</mi></mrow></msub><mo>≥</mo><mn>0</mn><mo>,</mo><mtext>&nbsp;</mtext><mtext>&nbsp;</mtext><mi>C</mi><mo>−</mo><msub><mi>λ</mi><mrow><mi>i</mi></mrow></msub><mo>−</mo><msub><mi>μ</mi><mrow><mi>i</mi></mrow></msub><mo>=</mo><mn>0</mn></math>

λ和μ都是拉格朗日乘子（见4.2），后者是应用于松弛变量的，并不会作为输出。

2. 用SMO算法（见下）求上面优化问题的所有λ
3. 用下面的方法计算w和b：（注意，w和x_i都是m维向量）

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>w</mi><mo>=</mo><munderover><mo data-mjx-texclass="OP">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>m</mi></mrow></munderover><msub><mi>λ</mi><mrow><mi>i</mi></mrow></msub><msub><mi>y</mi><mrow><mi>i</mi></mrow></msub><msub><mi>x</mi><mrow><mi>i</mi></mrow></msub></math>

选择一个λ_j，计算（也可以求平均）

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>b</mi><mo>=</mo><msub><mi>y</mi><mrow><mi>j</mi></mrow></msub><mo>−</mo><munderover><mo data-mjx-texclass="OP">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>n</mi></mrow></munderover><msub><mi>λ</mi><mrow><mi>i</mi></mrow></msub><msub><mi>y</mi><mrow><mi>i</mi></mrow></msub><mo stretchy="false">(</mo><msub><mi>x</mi><mrow><mi>i</mi></mrow></msub><mo>⋅</mo><msub><mi>x</mi><mrow><mi>j</mi></mrow></msub><mo stretchy="false">)</mo></math>

4. 求得超平面wx+b=0，分类决策函数为（sgn是符号函数）

$$
f(x)= sgn(wx+b)
$$

**SMO算法**（序列最小优化算法）：

针对前面问题中的约束条件

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><munderover><mo data-mjx-texclass="OP">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>n</mi></mrow></munderover><msub><mi>λ</mi><mrow><mi>i</mi></mrow></msub><msub><mi>y</mi><mrow><mi>i</mi></mrow></msub><mo>=</mo><mn>0</mn></math>

不断执行下列步骤，直到算法收敛。

1. 选择两个参数λ_i和λ_j作为待更新参数，其他参数视作常数
2. 在优化问题的目标函数中对λ_i求导，并令导数等于0，解出λ_i，然后根据下式解出λ_j

$$
\lambda _{j}= \frac{c-\lambda _{i}y_{i}}{y_{j}}
$$

其中

$$
c= \sum_{k\ne i,j} \lambda _{k}y_{k}
$$

## 线性不可分问题的解法
将样本点映射到高维空间，使其线性可分。这通过一个**核函数**来实现：

$$
k(m,n)= (\phi (m),\phi (n))
$$

它表示将向量m和n映射到更高维度

目标函数中，

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><munder><mrow><mi>m</mi><mi>a</mi><mi>x</mi></mrow><mi>λ</mi></munder><mo stretchy="false">[</mo><munderover><mo data-mjx-texclass="OP">∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>n</mi></mrow></munderover><msub><mi>λ</mi><mrow><mi>i</mi></mrow></msub><mo>−</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><munderover><mo data-mjx-texclass="OP">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>n</mi></mrow></munderover><munderover><mo data-mjx-texclass="OP">∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>n</mi></mrow></munderover><msub><mi>λ</mi><mrow><mi>i</mi></mrow></msub><msub><mi>λ</mi><mrow><mi>j</mi></mrow></msub><mo stretchy="false">(</mo><msub><mi>x</mi><mrow><mi>i</mi></mrow></msub><mo>⋅</mo><msub><mi>x</mi><mrow><mi>j</mi></mrow></msub><mo stretchy="false">)</mo><mo stretchy="false">]</mo></math>

(x_i·x_j)一项被替换为φ(x_i)·φ(x_j)，其余和线性情况相同。

常用的一个核函数：**高斯核函数**

$$
k(x,z)= exp(-\frac{||x-z||^{2}}{2\sigma ^{2}})
$$

这种情况下对应的SVM是高斯径向基函数分类器，决策函数为

$$
f(x)=sgn(\sum_{i=1}^{n} \lambda _{i}y_{i}exp(-\frac{||x-z||^{2}}{2\sigma ^{2}})+b)
$$

在线性可分的数据集上，线性核函数效果较好（ps：线性可分为什么还要用核函数？）；在非线性的数据集上，高斯核函数或多项式核函数效果较好。

## 优点
- 有严格的数学理论支持，可解释性强，不依靠统计方法，从而简化了通常的分类和回归问题
- 能找出对任务至关重要的关键样本（支持向量）
- 采用核技巧之后，可以处理非线性分类/回归任务
- 最终决策函数只由少数的支持向量所确定，计算的复杂性取决于支持向量的数目，而不是样本空间的维数，这在某种意义上避免了“维数灾难”

## 缺点
- 训练时间长，平方级
- 采用核技巧时，空间复杂度也是平方级
- 模型预测时，预测时间与支持向量的个数成正比。当支持向量的数量较大时，预测计算复杂度较高

（总结：只适用于小规模任务）

## 软件包
- LIBSVM: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
- LIBLINEAR: http://www.csie.ntu.edu.tw/~cjlin/liblinear/
- SVM_LIGHT,PERF,STRUCT: http://svmlight.joachims.org/svm_struct.html
- Pegasos: http://www.cs.huji.ac.il/~shais/code/index.html