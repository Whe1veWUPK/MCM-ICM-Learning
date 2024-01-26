import numpy as np
import statsmodels.api as sm

# 示例：估计 ARMA(2, 2) 模型的参数
# 这是一个随机生成的时间序列样本数据
np.random.seed(12345)
ar_params = np.array([0.75, -0.25])
ma_params = np.array([0.65, 0.35])
ar = np.r_[1, -ar_params]  # AR 参数
ma = np.r_[1, ma_params]   # MA 参数
y = sm.tsa.ArmaProcess(ar, ma).generate_sample(nsample=100)

# 使用 statsmodels 来拟合 ARMA 模型
model = sm.tsa.ARIMA(y, order=(2, 0, 2))  # ARMA(2, 2) 模型
results = model.fit()

# 输出模型拟合结果
print(results.summary())
