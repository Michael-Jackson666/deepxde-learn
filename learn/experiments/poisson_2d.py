"""
第一个DeepXDE实验：求解简单的Poisson方程

这是一个学习性质的实验，用来熟悉DeepXDE的基本使用流程。

PDE: -Δu = 1 在 [0,1]² 上
边界条件: u = 0 在边界上

理论解: u(x,y) = (x(1-x)y(1-y))/2
"""

import deepxde as dde
import numpy as np


def pde(x, y):
    """定义PDE：-Δu = 1"""
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    return -dy_xx - dy_yy - 1


def boundary(x, on_boundary):
    """定义边界"""
    return on_boundary


def func_zero(x):
    """边界条件：u = 0"""
    return 0


def main():
    # 定义几何区域：单位正方形
    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    
    # 定义边界条件
    bc = dde.icbc.DirichletBC(geom, func_zero, boundary)
    
    # 定义PDE问题
    data = dde.data.PDE(geom, pde, bc, num_domain=2540, num_boundary=80, solution=None, num_test=100)
    
    # 定义神经网络
    layer_size = [2] + [50] * 4 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)
    
    # 创建模型
    model = dde.Model(data, net)
    
    # 编译模型
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    
    # 训练模型
    losshistory, train_state = model.train(iterations=15000)
    
    # 保存和显示结果
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()