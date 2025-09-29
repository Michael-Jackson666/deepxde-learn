# PINNs学习笔记
PINNs，又称为physics-informed neural network，中文为物理信息神经网络，是用加入物理信息的神经网络来求解pde。是近十年来兴起的求解pde的方法，由于和现在最火热的人工智能关系很大，所以算是一个非常火的赛道。学习PINNs技术是为了结合传统与现代的方法，做出新的成果。

## 1. PINNs框架
### 1.1. PINNs数学理论基础
PINNs的核心思想就是将物理定律以微分方程的形式嵌入到神经网络中，并且使用==自动微分==。PINNs可以用于求解积分微分方程，分数阶偏微分方程，随机偏微分方程等。首先考虑含参数 $\boldsymbol{\lambda}$ 的偏微分方程，在域$\Omega \subset \mathbb{R}^d$上有解$u(\mathbf{x})$，且$\mathbf x = (x_1, \cdots, x_d)$，方程为
$$
f(\mathbf x; \frac{\partial u}{\partial x_1},\cdots,\frac{\partial u}{\partial x_d};\frac{\partial^2 u}{\partial x_1\partial x_2},\cdots\frac{\partial^2 u}{\partial x_1\partial x_d};\cdots;\boldsymbol{\lambda}) = 0, x\in \Omega \tag{1.1}
$$
有边界条件为
$$
\mathcal{B}(u,\mathbf{x}) = 0\ \text{on} \ \partial \Omega,
$$
其中$\mathcal{B}(u,\mathbf{x})$可能为Dirichlet，Neumann，Robin和周期性边界条件。简写(1.1)式子可以表示为
$$
\mathcal{F}[u](x) = 0, x \in \Omega
$$
都是表示微分方程的大类。


下面以一个简单的热方程的例子来讲解整个PINNs的实现过程。设扩散方程
$$
\frac{\partial u}{\partial t} = \boldsymbol \lambda \frac{\partial^2 u}{\partial x^2}
$$
有混合边界条件为
$$
\left \{  
\begin{align}
u(x,t) &= g_D(x,t), (x,t) \in \Gamma_D\subset \partial \Omega\\
\frac{u(x,t)}{\partial \mathbf{n}} &= g_R(x,t), (x,t) \in \Gamma_R\subset \partial \Omega
\end{align}
\right. 
$$
首先是构建神经网络$\hat{u}(\mathbf{x};\mathbf{\theta})$作为解$u(\mathbf{x})$的替代。这里$\mathbf \theta = \left\{\mathbf W ^l , \mathbf b^l\right\}_{1\leq l\leq L}$为神经网络$\hat u$权重矩阵和偏置向量的集合。在PINNs中用神经网络替代解的一个好处为，可以通过链式法则和神经网络的==自动微分==技术计算$\hat u$的导数。上述扩散方程的例子如下图所示

![[PINNs of diffusion equation.png]]

下一步是限制$\hat u$使其满足PDE和边界条件施加的物理性质。在实践中，我们将$\hat u$限制在一些分散点（例如，随机分布的点，或域中的聚点），即大小为$|\mathcal{T}|$的训练数据
$$
\mathcal{T} = \left \{ \mathcal x_1, \mathcal x_1, \cdots, \mathcal x_{|\mathcal T|}\right \}
$$
另外，$\mathcal T$还包含了在域中的点和在边界的点集，$\mathcal T_f \subset \Omega$和$\mathcal T_b \subset \partial \Omega$，并且称$\mathcal T_f$和$\mathcal T_b$为==残差点==集合。

为了测量网络和约束之间的误差，选择$L^2$范数下残差点的损失函数
$$
\mathcal{L}(\boldsymbol{\theta};\mathcal{T}) = w_f \mathcal{L}_f(\boldsymbol{\theta};\mathcal{T}_f) + w_b \mathcal{L}_b(\boldsymbol{\theta};\mathcal{T}_b),\tag{1.2}
$$
其中
$$
\begin{align}
\mathcal{L}_f(\boldsymbol{\theta};\mathcal{T}_f) &= \frac{1}{|\mathcal{T}_f|} \sum_{\mathbf{x} \in \mathcal{T}_f} \left\| f \left( \mathbf{x}; \frac{\partial \hat{u}}{\partial x_1}, \dots, \frac{\partial \hat{u}}{\partial x_d}, \frac{\partial^2 \hat{u}}{\partial x_1^2}, \dots, \frac{\partial^2 \hat{u}}{\partial x_1 \partial x_d}, \dots ; \boldsymbol{\lambda} \right) \right\|_2^2,\\
\mathcal{L}_b(\boldsymbol{\theta};\mathcal{T}_b) &= \frac{1}{|\mathcal{T}_b|} \sum_{\mathbf{x} \in \mathcal{T}_b} \|\mathcal{B}(\hat{u}, \mathbf{x})\|_2^2.
\end{align}
$$
而$w_f$和$w_b$表示权重。

最后，通过训练最小化误差$\mathcal L(\boldsymbol \theta; \mathcal T)$来寻找一个好的参数$\boldsymbol \theta$。由于损失函数具有高非线性性和非凸性，这里使用基于梯度下降的优化器比如Adam和L-BFGS。但是注意，在不同问题背景下不同的优化器会有不同的表现，比如光滑的PDE下L-BFGS表现优于Adam，因为前者使用的是损失的二阶导数，而Adam则为一阶。然而对于刚性问题，L-BFGS会卡在局部最小。还要注意的是PINNs计算的数值解可能会收敛到不同的解根据优化器的不同选择。

总结步骤包括一下四点
1. 构造含参数$\boldsymbol \theta$的神经网络$\hat u(\mathbf x;\boldsymbol \theta)$。
2. 选取两个训练集合 $\mathcal L_f$ 和 $\mathcal L_b$ 表示方程和初边值条件。
3. 通过对偏微分方程和边界条件残差的加权 $L^2$ 范数求和来指定损失函数。
4. 通过最小化损失函数 $\mathcal L(\boldsymbol \theta; \mathcal T)$ 来训练网络，以找到最优参数 $\boldsymbol \theta^*$。

在算法的具体实践过程中，除了最小化边界损失函数$\mathcal L_b$的软约束，在边界较为简单的情形，还可以采用强制约束方法。比如边界条件$u(0) = u(1) = 0, \Omega = [0,1]$，可以取网络$\hat u = x(x-1)\mathcal N (x)$，其中$\mathcal N(x)$是一个神经网络。这样可以自动满足边界条件。

#### 残差点的选取
残差点，也就是数据点的选取非常重要，这里给出三种策略
1. 我们可以在训练开始时指定残差点，可以是晶格上的网格点，也可以是随机点，并且在训练过程中永远不要改变它们。
2. 在每次优化迭代中，我们可以随机选择不同的残差点。
3. 我们可以在训练过程中自适应地改进残差点的位置，后面会提到自适应方法。

### 1.2. 逼近论和PINNs的误差估计
要完善PINNs理论部分，必须回答是否存在一个神经网络同时满足PDE和边界条件，并且可以一致逼近函数其他偏导数。首先定义记号，$\mathbb Z_{+}^d$为$d$维非负整数向量空间。对于$\mathbf m = (m_1,m_2,\cdots, m_d)\in \mathbb Z_{+}^d$，并规定$|\mathbf m| := m_1 + \cdots + m_d$，且有微分算子
$$
D^{\mathbf m} := \frac{\partial^{|\mathbf m|}}{\partial x_1^{m_1}\cdots \partial x_d^{m_d}}.
$$
称$f \in C^{\mathbf m}(\mathbb R^d)$若$D^{\mathbf k}f \in C(\mathbb R^d)$对于所有的$\mathbf k \leq \mathbf m, \mathbf k \mathbb Z_+^d$，其中$C(\mathbb R^d) = \left\{f: \mathbb{R}^d \to \mathbb R| f \text{连续}\right\}$ 为连续函数空间。有定理如下

 **<font color=pink>定理</font> 1.1.（神经网络逼近函数定理）** 设 $\mathbf{m}^i \in \mathbb{Z}^d_+$，其中 $i = 1, \dots, s$，并设 $m = \max_{i=1, \dots, s} |\mathbf{m}^i|$。假设 $\sigma \in C^m(\mathbb{R})$ 且 $\sigma$ **不是**一个多项式。那么，**单隐层神经网络**的空间
$$
\mathcal{M}(\sigma) := \text{span} \{\sigma(\mathbf{w} \cdot \mathbf{x} + b) : \mathbf{w} \in \mathbb{R}^d, b \in \mathbb{R}\}
$$
在空间
$$
C^{\mathbf{m}^1, \dots, \mathbf{m}^s}(\mathbb{R}^d) := \bigcap_{i=1}^s C^{\mathbf{m}^i}(\mathbb{R}^d)
$$
中是**稠密**的。
即，对于任意 $f \in C^{\mathbf{m}^1, \dots, \mathbf{m}^s}(\mathbb{R}^d)$、任意紧集 $K \subset \mathbb{R}^d$ 以及任意 $\epsilon > 0$，都存在一个 $g \in \mathcal{M}(\sigma)$ 满足
$$
\max_{\mathbf{x} \in K} |D^{\mathbf{k}} f(\mathbf{x}) - D^{\mathbf{k}} g(\mathbf{x})| < \epsilon
$$
对于所有 $\mathbf{k} \in \mathbb{Z}^d_+$，其中对于某个 $i$，有 $\mathbf{k} \le \mathbf{m}^i$。

上述定理表明，拥有足够神经元个数的前馈神经网络可以一致逼近任何函数及其偏导数。设$\mathcal F$表示特定选择的神经网络架构可以表示的所有函数类。但是解$u$可能不在$\mathcal F$中，我们定义$u_{\mathcal F} = \arg \min_{f\in \mathcal F}\|f-u\|$为$\mathcal F$中间中对解$u$的最佳估计。由于只在训练集$\mathcal T$上训练，于是定义$u_{\mathcal T} = \arg \min_{f\in \mathcal F}\mathcal L(f;\mathcal L)$作为全局损失最小的网络。但是最小化误差通常在计算上难以实现，于是根据优化器得到的解定义为$\tilde u_{\mathcal T}$。这里可以定义全部误差为
$$
\mathcal{E} := \|\tilde{u}_{\mathcal{T}} - u\| \le \underbrace{\|\tilde{u}_{\mathcal{T}} - u_{\mathcal{T}}\|} _{\mathcal{E}_{\text{opt}}} + \underbrace{\|u_{\mathcal{T}} - u_{\mathcal{F}}\|} _{\mathcal{E}_{\text{gen}}} + \underbrace{\|u_{\mathcal{F}} - u\|} _{\mathcal{E}_{\text{app}}}.
$$
其中**近似误差** $\mathcal{E}_{\text{app}}$ 衡量 $u_{\mathcal{F}}$ 能在多大程度上近似 $u$。**泛化误差** $\mathcal{E}_{\text{gen}}$ 由 $\mathcal{T}$ 中**残差点**的数量/位置以及函数族 $\mathcal{F}$ 的**容量**决定。尺寸更大的神经网络具有更小的近似误差，但可能导致更高的泛化误差，这被称为**偏差-方差权衡**（bias-variance tradeoff）。当泛化误差占主导地位时，就会发生**过拟合**（Overfitting）。此外，**优化误差** $\mathcal{E}_{\text{opt}}$ 源于损失函数的复杂性和优化设置，例如**学习率**和**迭代次数**。


### 1.3. PINNs和FEM的比较
PINNs作为目前最主流的深度学习求解pde技术，而有限元则是工程中应用最为广泛的方法，二者各有优劣，比如PINNs可以解决FEM维数诅咒问题，但是需要很大的训练资源，理论不完备。PINNs还提供了对函数及其偏导数的非线性逼近，但有限元往往是线性逼近。做个对比如下所示

| 对比项                          | PINN                                                                               | FEM        |
| :--------------------------- | :--------------------------------------------------------------------------------- | :--------- |
| **基函数 (Basis function)**     | 神经网络 (非线性)                                                                         | 分段多项式 (线性) |
| **参数 (Parameters)**          | 权重和偏差                                                                              | 点值         |
| **训练点 (Training points)**    | 散点 (无网格)                                                                           | 网格点        |
| **偏微分方程嵌入 (PDE embedding)**  | 损失函数                                                                               | 代数系统       |
| **参数求解器 (Parameter solver)** | 基于梯度的优化器                                                                           | 线性求解器      |
| **误差 (Errors)**              | $\mathcal{E}_{\text{app}}$、$\mathcal{E}_{\text{gen}}$ 和 $\mathcal{E}_{\text{opt}}$ | 近似/求积误差    |
| **误差界限 (Error bounds)**      | 尚不可用                                                                               | 部分可用       |

* 在 **FEM** 中，用待确定的点值分段多项式来近似解 $u$，而在PINNs中，我们构建一个由权重和偏差参数化的神经网络作为替代模型。
* **FEM** 通常需要**网格生成**，而PINNs是完全**无网格**的，并且可以使用网格或随机点。
* **FEM** 使用刚度矩阵将 PDE 转换为**代数系统**，而 PINNs将 PDE 和边界条件**嵌入到损失函数**中。
* 在最后一步，**FEM** 中的代数系统通过**线性求解器**直接求解，而PINNs中的网络是通过**基于梯度的优化器**学习的。

### 1.4. 基于残差的自适应细化方法（Residual-Based adaptive refinement）
在前面介绍中，残差点$\mathcal T$通常为域中的随机分布。但这种方法对于某些特定方程，比如某个地方梯度很大的PDE(example: Burgers equation)，表现会很差。这里介绍一种基于残差的自适应细化方法来改善训练过程中的残差分布。$\text{RAR}$的思路是，我们将在 $\text{PDE}$ **残差**
$$\left\| f\left(\mathbf{x} ; \frac{\partial \hat{u}}{\partial x_{1}}, \cdots, \frac{\partial \hat{u}}{\partial x_{d}}, \frac{\partial^{2} \hat{u}}{\partial x_{1} \partial x_{1}}, \cdots, \frac{\partial^{2} \hat{u}}{\partial x_{1} \partial x_{d}}, \cdots ; \boldsymbol{\lambda}\right)\right\|$$
**较大**的位置**添加更多残差点**，并重复添加点，直到**平均残差**
$$
\varepsilon_{r}=\frac{1}{V} \int_{\Omega}\left\|f\left(\mathbf{x} ; \frac{\partial \hat{u}}{\partial x_{1}}, \cdots, \frac{\partial \hat{u}}{\partial x_{d}}, \frac{\partial^{2} \hat{u}}{\partial x_{1} \partial x_{1}}, \cdots, \frac{\partial^{2} \hat{u}}{\partial x_{1} \partial x_{d}}, \cdots ; \boldsymbol{\lambda}\right)\right\| \mathbf{d} \mathbf{x}
\tag{1.3}
$$
**小于阈值** $\mathcal{E}_{0}$，其中 $V$ 是 $\Omega$ 的体积。

## 2. PINNs实现
本章节介绍用PINNs求解pde的全过程，主要依赖的库是[DeepXDE](https://github.com/lululxvi/deepxde)，也是就lulu团队创立的。

### 2.1.DeepXDE使用方法
首先介绍如何使用DeepXDE库，并且合理利用其中的函数来减少代码量。在DeepXDE 中求解微分方程无非是使用内置模块指定问题，包括计算域（几何和时间）、PDE、边界/初始条件、约束、训练数据、神经网络架构和训练超参数。该工作流程如下所示

| 步骤       | 描述                                                                                                                                                                 |
| :------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **步骤 1** | 使用 **`geometry`** 模块指定计算域。                                                                                                                                         |
| **步骤 2** | 使用 $\text{TensorFlow}$ 的语法指定 **$\text{PDE}$**（偏微分方程）。                                                                                                              |
| **步骤 3** | 指定边界条件和初始条件。                                                                                                                                                       |
| **步骤 4** | 组合几何、$\text{PDE}$ 和边界/初始条件，分别使用 **`data.PDE`**（针对时间无关问题）或 **`data.TimePDE`**（针对时间相关问题）。要指定训练数据，我们可以设置**具体的点位置**，或只设置**点的数量**，然后 $\text{DeepXDE}$ 将在网格上或随机地采样所需的点数。 |
| **步骤 5** | 使用 **`maps`** 模块构建一个神经网络。                                                                                                                                          |
| **步骤 6** | 通过组合步骤 4 中的 $\text{PDE}$ 问题和步骤 5 中的神经网络来定义一个 **`Model`**。                                                                                                          |
| **步骤 7** | 调用 **`Model.compile`** 来设置优化超参数，例如 **优化器** 和 **学习率**。公式 (1.2) 中的权重可以通过 **`loss_weights`** 来设置。                                                                     |
| **步骤 8** | 调用 **`Model.train`**，通过随机初始化或使用参数 **`model_restore_path`** 从预训练模型开始训练网络。使用 **`callbacks`** 可以非常灵活地监控和修改训练行为。                                                       |
#### DeepXDE中的各种内置函数和类

在 $\text{DeepXDE}$ 中，内置的**基本几何体**包括 **`interval`**、**`triangle`**、**`rectangle`**、**`polygon`**、**`disk`**、**`cuboid`** 和 **`sphere`**。可以利用这几个基本几何体，通过三种**布尔运算**：**`union`** ($\mid$，并集)、**`difference`** ($-$，差集) 和 **`intersection`** ($\&$，交集) 来构造出其他几何体。这种技术称为**构造实体几何** ($\text{CSG}$，Constructive Solid Geometry)。$\text{CSG}$ 支持 **2D** 和 **3D** 几何体。

$\text{DeepXDE}$ 支持四种标准的**边界条件**，包括 $\text{Dirichlet}$ (**`DirichletBC`**)、$\text{Neumann}$ (**`NeumannBC`**)、$\text{Robin}$ (**`RobinBC`**) 和 $\text{periodic}$ (**`PeriodicBC`**)，同时也可以使用 **`OperatorBC`** 定义更一般的边界条件。**初始条件**可以使用 **`IC`** 来定义。

$\text{DeepXDE}$ 中有两种可用的神经网络类型：**前馈神经网络** (**`maps.FNN`**) 和 **残差神经网络** (**`maps.ResNet`**)。

选择不同的训练**超参数**也很方便，例如损失函数类型 ($\text{loss types}$)、指标 ($\text{metrics}$)、优化器 ($\text{optimizers}$)、学习率调度 ($\text{learning rate schedules}$)、初始化 ($\text{initializations}$) 和正则化 ($\text{regularizations}$)。

除了求解微分方程外，$\text{DeepXDE}$ 还可以用于从**多保真度数据**中**近似函数**，以及**学习非线性算子**。


### 2.2. 自定义网络
除了上述类和函数，我们还可以自定义PINNs来求解不同的问题。首先是几何域上的自定义，除了上述的几个区域以及CSG技术，可以自由组合这些图形，创建自己的几何域的类代码如下
```python
class MyGeometry(Geometry):
    def inside(self, x):
        """Check if x is inside the geometry."""

    def on_boundary(self, x):
        """Check if x is on the geometry boundary."""

    def boundary_normal(self, x):
        """Compute the unit normal at x for Neumann or Robin boundary conditions."""

    def periodic_point(self, x, component):
        """Compute the periodic image of x for periodic boundary condition."""

    def uniform_points(self, n, boundary=True):
        """Compute the equispaced point locations in the geometry."""

    def random_points(self, n, random="pseudo"):
        """Compute the random point locations in the geometry."""

    def uniform_boundary_points(self, n):
        """Compute the equispaced point locations on the boundary."""

    def random_boundary_points(self, n, random="pseudo"):
        """Compute the random point locations on the boundary."""
```

在DeepXDE中，其支持两种网络，**前馈神经网络** (**`maps.FNN`**) 和 **残差神经网络** (**`maps.ResNet`**)。构建自己的神经网络如下
```python
class MyNet(Map):
	@property
	def inputs(self):
		"""Return the net inputs."""
	
	@property
	def outputs(self):
		"""Return the net outputs."""
	
	@property
	def targets(self):
		"""Return the targets of the net outputs."""
	
	def build(self):
		"""Construct the network."""
```

除了上述定义之外，监控神经网络的训练过程，然后实时进行修改，例如改变学习率。在 DeepXDE 中，这可以通过添加回调函数来实现，这里列出一些 DeepXDE 中已经实现的常用函数：
- **ModelCheckpoint**，它在特定的 epoch 之后或当找到一个更好的模型时**保存模型**。
- **OperatorPredictor**，它计算应用于输出的**算子（operator）的值**。
* **FirstDerivative**，它计算输出相对于输入的**一阶导数**。这是 **OperatorPredictor** 的一个特例，其中算子是一阶导数。
* **MovieDumper**，它在训练过程中**转储函数的变化（movie）**，例如其傅里叶变换频谱的变化。

自定义Callback类如下所示
```python
class MyCallback(Callback):
    def on_epoch_begin(self):
        """Called at the beginning of every epoch."""

    def on_epoch_end(self):
        """Called at the end of every epoch."""
```
# DeepXDE学习笔记

## 什么是DeepXDE？

DeepXDE是一个专门用于求解偏微分方程(PDE)的深度学习库，主要基于Physics-Informed Neural Networks (PINNs)的方法。

## 核心概念

### 1. Physics-Informed Neural Networks (PINNs)
- 将物理定律（PDE）作为约束条件直接嵌入到神经网络的损失函数中
- 不仅依赖数据，还利用物理方程的先验知识
- 可以处理正向问题和逆向问题

### 2. 主要组件
- **Geometry**: 定义求解域的几何形状
- **PDE**: 定义偏微分方程
- **Initial/Boundary Conditions**: 初始条件和边界条件
- **Neural Network**: 神经网络架构
- **Optimizer**: 优化器

### 3. 支持的后端
- TensorFlow
- PyTorch
- JAX
- PaddlePaddle

## 学习要点

1. 理解PDE的数学形式
2. 掌握边界条件和初始条件的设置
3. 学会设计合适的网络架构
4. 理解损失函数的构成

## 待补充内容

- [ ] 具体的数学推导
- [ ] 代码实例分析
- [ ] 常见问题和解决方案