# DeepXDE Tutorial Series | DeepXDE教程系列

> **目标**: 从零基础到精通，系统学习使用DeepXDE求解偏微分方程

[![DeepXDE](https://img.shields.io/badge/DeepXDE-v1.12+-blue.svg)](https://github.com/lululxvi/deepxde)
[![Python](https://img.shields.io/badge/Python-3.7+-green.svg)](https://www.python.org/)
[![Tutorial](https://img.shields.io/badge/Tutorial-Progressive-orange.svg)]()

---

## 📚 教程概览

### 🎯 学习路径

```mermaid
graph LR
    A[Tutorial 1<br/>1D Poisson] --> B[Tutorial 2<br/>2D Heat Equation]
    B --> C[Tutorial 3<br/>PDE Systems]
    C --> D[Tutorial 4<br/>Complex Geometry]
    D --> E[Tutorial 5<br/>Inverse Problems]
```

### 📖 教程列表

| 序号 | 教程名称 | 文件 | 难度 | 学习时间 | 核心概念 |
|------|----------|------|------|----------|----------|
| 1️⃣ | **1D泊松方程** | [`poisson_1d.ipynb`](poisson_1d.ipynb) | ⭐⭐ | 30分钟 | PDE基础, 边界条件, PINN |
| 2️⃣ | **2D热传导方程** | [`heat_2d.ipynb`](heat_2d.ipynb) | ⭐⭐⭐ | 45分钟 | 时空PDE, 初始条件, 多维可视化 |
| 3️⃣ | **PDE方程组** | [`system_pde.ipynb`](system_pde.ipynb) | ⭐⭐⭐⭐ | 60分钟 | 耦合系统, 多变量PDE, 反应扩散 |
| 4️⃣ | **NS方程基础** | [`ns_equations_basic.ipynb`](ns_equations_basic.ipynb) | ⭐⭐⭐⭐ | 75分钟 | Navier-Stokes, 流体力学, 腔体流动 |
| 5️⃣ | **NS方程进阶** | [`ns_equations_advanced.ipynb`](ns_equations_advanced.ipynb) | ⭐⭐⭐⭐⭐ | 90分钟 | 复杂边界, 高雷诺数, 管道流动 |
<!-- | 6️⃣ | **复杂几何域** | `complex_geometry.ipynb` | ⭐⭐⭐⭐ | 45分钟 | 不规则域, CSG几何 |
| 7️⃣ | **逆问题求解** | `inverse_problems.ipynb` | ⭐⭐⭐⭐⭐ | 90分钟 | 参数反演, 数据同化 | -->

---

## 🚀 快速开始

### 环境配置

```bash
# 1. 克隆项目
git clone https://github.com/your-repo/deepxde-learn.git
cd deepxde-learn/learn/tutorial

# 2. 安装依赖
pip install deepxde matplotlib numpy scipy

# 3. 启动Jupyter
jupyter notebook
```

### 🔧 系统要求

| 组件 | 最低版本 | 推荐版本 | 说明 |
|------|----------|----------|------|
| **Python** | 3.7+ | 3.9+ | 主要运行环境 |
| **DeepXDE** | 1.10+ | 1.12+ | 核心PDE求解库 |
| **TensorFlow** | 2.4+ | 2.8+ | 默认后端 |
| **NumPy** | 1.18+ | 1.21+ | 数值计算 |
| **Matplotlib** | 3.3+ | 3.5+ | 可视化工具 |

---

## 📋 学习计划建议

### 🌟 初学者路径 (2-3周)
```
Week 1: Tutorial 1-2 (1D Poisson + 2D Heat) 
        → 理解PINN基础概念和时空问题
        
Week 2: Tutorial 3-4 (PDE系统 + NS基础)
        → 掌握多变量耦合和流体力学基础
        
Week 3: 复习 + NS进阶学习
        → 深入复杂流动问题
```

### 🔥 进阶路径 (4-6周)
```
Week 1-2: Tutorial 1-3 (基础铺垫)
Week 3-4: Tutorial 4-5 (NS方程系列)  
Week 5-6: Tutorial 6-7 + 项目实战
```

### ⚡ 快速上手 (1周)
```
Day 1-2: Tutorial 1 (重点理解)
Day 3-4: Tutorial 2 (快速浏览)
Day 5-7: 选择应用场景实战
```

---

## 🧠 核心概念图谱

### 数学基础
- **偏微分方程 (PDE)**: 物理规律的数学表述
- **边界条件**: 定义域边界上的约束
- **初始条件**: 时变问题的起始状态
- **解析解**: 精确的数学解（用于验证）

### 物理直觉
- **扩散过程**: 热传导、浓度扩散
- **波动现象**: 振动、声波传播
- **守恒定律**: 能量、质量、动量守恒
- **多尺度问题**: 时间/空间多尺度耦合

### 技术实现
- **神经网络**: 函数近似器
- **自动微分**: 计算偏导数
- **损失函数**: PDE残差 + 边界/初始条件
- **优化算法**: Adam, L-BFGS等

---

## 🎯 每个教程的学习目标

### Tutorial 1: 1D Poisson Equation
```python
# 你将学会：
✅ PDE问题的基本设置和求解流程
✅ 边界条件的定义和应用
✅ 神经网络结构的选择策略
✅ 训练过程监控和结果验证
✅ 解的物理意义解释
```

### Tutorial 2: 2D Heat Equation
```python
# 你将掌握：
✅ 多维时空问题的处理方法
✅ 初始条件与边界条件的结合
✅ 时变PDE的可视化技巧
✅ 复杂度增加带来的挑战和对策
✅ 物理扩散过程的数值模拟
```

### Tutorial 3: PDE Systems
```python
# 你将探索：
✅ 多变量耦合PDE系统求解
✅ 反应扩散方程组的数值模拟
✅ 多输出神经网络的构建和训练
✅ 变量间耦合效应的分析
✅ 复杂系统的可视化技巧
```

### Tutorial 4: NS Equations Basic
```python
# 你将掌握：
✅ Navier-Stokes方程的物理意义和数学表述
✅ 不可压缩流体的基本假设和简化
✅ 腔体驱动流动的经典CFD问题
✅ 流体速度、压力、涡量场的分析
✅ 雷诺数对流动特性的影响
```

### Tutorial 5: NS Equations Advanced
```python
# 你将精通：
✅ 复杂边界条件的数学建模和数值处理
✅ 高雷诺数流动的强非线性特征
✅ 瞬态流动的发展过程和稳态特性
✅ 入流、出流和壁面边界的精确处理
✅ 管道流动的理论解与数值解对比
✅ 分阶段训练和自适应采样策略
```

### Tutorial 6: Complex Geometry (计划中)

---

## 🌊 Navier-Stokes方程专题系列

### 📚 系列概述

Navier-Stokes方程是流体力学的核心，描述了粘性流体的运动规律。本系列教程从基础到进阶，系统讲解如何使用PINN求解NS方程。

```mermaid
graph LR
    A[NS基础<br/>腔体流动<br/>Re=100] --> B[NS进阶<br/>管道流动<br/>Re=400]
    B --> C[复杂几何<br/>圆柱绕流<br/>Re=200]
    C --> D[湍流建模<br/>RANS方程<br/>Re>1000]
    D --> E[多物理场<br/>传热流动<br/>耦合求解]
```

### 🎯 学习递进路径

| 教程 | 物理现象 | 数学复杂度 | 数值挑战 | 应用价值 |
|------|----------|------------|----------|----------|
| **NS基础** | 腔体驱动流 | ⭐⭐⭐ | ⭐⭐⭐ | CFD入门经典 |
| **NS进阶** | 管道流动 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 工程流动基础 |
| **复杂几何** | 圆柱绕流 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 钝体绕流 |
| **湍流建模** | 高Re湍流 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 工程湍流 |

### 🔬 核心知识体系

**数学基础**
- 不可压缩NS方程组的推导和物理意义
- 边界条件类型：Dirichlet, Neumann, Robin
- 无量纲化和相似性分析
- 雷诺数的物理意义和流动分类

**数值方法**
- PINN的数学原理和实现细节
- 损失函数设计和权重平衡策略
- 分阶段训练和自适应采样
- 多尺度网络架构和物理约束

**流体力学**
- 层流、过渡流和湍流的特征
- 边界层理论和分离流动
- 涡量动力学和流动结构
- 压力驱动流和剪切驱动流

### 💡 实用技巧汇总

**建模技巧**
```python
# 1. 合理的无量纲化
Re = U * L / nu
x_star = x / L
u_star = u / U
p_star = p / (rho * U^2)

# 2. 边界条件的数值处理
bc_wall = dde.DirichletBC(geom, lambda x: 0, boundary_wall)
bc_inlet = dde.DirichletBC(geom, inlet_profile, boundary_inlet)

# 3. 损失函数权重调整
loss_weights = [1.0, 1.0, 10.0]  # [u_eq, v_eq, continuity]
```

**训练策略**
```python
# 分阶段训练
# Stage 1: 低学习率预训练
model.compile("adam", lr=1e-4)
model.train(10000)

# Stage 2: 标准训练
model.compile("adam", lr=1e-3) 
model.train(20000)

# Stage 3: 精细优化
model.compile("L-BFGS")
model.train()
```

**验证方法**
```python
# 物理约束检验
def verify_physics(model, test_points):
    # 连续性方程验证
    divergence = compute_divergence(model, test_points)
    
    # 边界条件验证
    bc_error = check_boundary_conditions(model, test_points)
    
    # 理论解对比
    theory_error = compare_with_theory(model, test_points)
    
    return divergence, bc_error, theory_error
```

### 🚀 进阶拓展方向

1. **计算效率优化**
   - GPU并行计算
   - 混合精度训练
   - 模型压缩技术

2. **物理约束增强**
   - 守恒律自动满足
   - 对称性嵌入
   - 多尺度建模

3. **工程应用扩展**
   - 优化设计问题
   - 实时流动预测
   - 数字孪生技术

---

## 🛠️ 工具和技巧

### 调试技巧
- **损失监控**: 观察PDE残差、边界条件、初始条件的损失组成
- **梯度检查**: 验证自动微分的正确性
- **采样可视化**: 检查采样点分布是否合理
- **解的物理性**: 验证结果是否符合物理直觉

### 性能优化
- **网络结构**: 调整深度/宽度平衡精度与效率
- **采样策略**: 在关键区域增加采样密度
- **学习率调度**: 使用自适应学习率
- **硬件加速**: 利用GPU加速训练过程

### 可视化技巧
- **1D问题**: 线图、误差分布图
- **2D问题**: 等高线图、3D曲面、动画
- **3D问题**: 切片可视化、体渲染
- **训练过程**: 损失曲线、梯度流

---

## 🤝 贡献指南

### 如何贡献
1. **提交bug报告**: 发现问题请在Issues中详细描述
2. **改进文档**: 优化教程内容和代码注释
3. **添加示例**: 贡献新的物理问题案例
4. **性能优化**: 提升训练效率和精度

### 代码规范
```python
# 文档字符串格式
def pde_function(x, y):
    """
    PDE定义函数
    
    Args:
        x: 输入坐标 (N, dim)
        y: 网络输出 (N, 1)
        
    Returns:
        PDE残差 (N, 1)
    """
    pass
```

---

## 📞 获取帮助

### 常见问题
- **Q: 训练不收敛怎么办？**
  - A: 检查PDE定义、增加采样点、调整网络结构、降低学习率

- **Q: 内存不足怎么处理？**
  - A: 减少采样点数量、使用批处理、降低网络复杂度

- **Q: 如何选择网络结构？**
  - A: 从简单开始，逐渐增加复杂度，平衡精度与效率

### 社区资源
- 📖 [DeepXDE官方文档](https://deepxde.readthedocs.io/)
- 💬 [GitHub Discussions](https://github.com/lululxvi/deepxde/discussions)
- 📝 [论文列表](https://github.com/lululxvi/deepxde/blob/master/REFERENCES.md)
- 🎥 [视频教程](https://www.youtube.com/playlist?list=PLV7LvtPPT4d5F)

---

## 📄 许可证

本教程系列遵循 [MIT License](../../LICENSE)，欢迎自由使用和分享。

---

## 🙏 致谢

感谢以下项目和团队：
- [DeepXDE](https://github.com/lululxvi/deepxde) - 核心PDE求解框架
- [TensorFlow](https://tensorflow.org/) - 深度学习后端
- [SciML](https://sciml.ai/) - 科学计算生态系统

---

<div align="center">

**Happy Learning! 开始你的PDE求解之旅吧！** 🚀

[![Star](https://img.shields.io/github/stars/lululxvi/deepxde?style=social)](https://github.com/lululxvi/deepxde)
[![Follow](https://img.shields.io/github/followers/yourusername?style=social)]()

</div>