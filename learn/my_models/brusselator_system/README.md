# Brusselator反应扩散方程组求解器

这是一个基于DeepXDE的Brusselator反应扩散方程组求解器，使用物理信息神经网络(PINNs)求解耦合偏微分方程组。

## 📚 方程组介绍

Brusselator是一个经典的反应扩散系统，描述了两种化学物质u和v的时空演化：

```
∂u/∂t = D_u∇²u + A - (B+1)u + u²v
∂v/∂t = D_v∇²v + Bu - u²v
```

其中：
- `u, v`: 两种化学物质的浓度
- `D_u, D_v`: 扩散系数
- `A, B`: 反应参数

## 📁 文件结构

```
├── brusselator_system.py          # 主求解器类
├── run_brusselator_examples.py    # 使用示例和演示
└── README.md                       # 说明文档
```

## 🚀 快速开始

### 1. 环境要求

确保已安装以下Python包：

```bash
pip install deepxde numpy matplotlib
```

### 2. 基础使用

```python
from brusselator_system import BrusselatorSolver

# 创建求解器
solver = BrusselatorSolver(A=1.0, B=3.0, D_u=0.2, D_v=0.1)

# 设置问题
solver.setup_geometry_and_conditions()

# 创建神经网络模型
solver.create_model()

# 训练模型
solver.train()

# 可视化结果
solver.visualize_evolution()
solver.analyze_center_evolution()
```

### 3. 运行示例

```bash
python run_brusselator_examples.py
```

程序会提供多个示例选项：
1. **基础使用示例** - 标准参数求解
2. **参数研究示例** - 不同B值的影响
3. **自定义初始条件示例** - 高斯和环形初始分布
4. **分析工具示例** - 深入分析系统行为

## 🔧 主要功能

### BrusselatorSolver类

- **初始化参数配置**: `__init__(A, B, D_u, D_v, domain_size, time_end)`
- **设置几何和边界条件**: `setup_geometry_and_conditions()`
- **创建神经网络模型**: `create_model(num_domain, num_boundary, num_initial, layer_sizes)`
- **模型训练**: `train(adam_iterations, adam_lr, use_lbfgs)`
- **预测**: `predict(x_points)`
- **可视化**:
  - `visualize_initial_conditions()` - 初始条件可视化
  - `visualize_evolution()` - 时空演化可视化
  - `analyze_center_evolution()` - 中心点演化和相空间分析
- **模型保存/加载**: `save_model()`, `load_model()`

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `A` | 1.0 | 反应参数A |
| `B` | 3.0 | 反应参数B |
| `D_u` | 0.2 | u的扩散系数 |
| `D_v` | 0.1 | v的扩散系数 |
| `domain_size` | 1.0 | 空间域大小 [0, domain_size]² |
| `time_end` | 2.0 | 时间域终点 [0, time_end] |

## 📊 使用技巧

### 1. 网络设计要点
- **多输出结构**: 输出维度=2 (对应u和v)
- **增加网络容量**: 建议使用4层以上，每层64个神经元
- **激活函数**: `tanh`对反应扩散方程效果最好

### 2. 训练策略
- **两阶段训练**: Adam预训练 + L-BFGS精细调优
- **降低学习率**: 耦合系统建议使用0.0005-0.001的学习率
- **增加训练点**: 方程组比单方程复杂，需要更多采样点

### 3. 参数调优
- **A, B参数**: 控制反应动力学，不同值产生不同的时空模式
- **扩散系数**: D_u和D_v的比值影响空间模式的形成
- **边界条件**: 当前实现使用零边界条件，可修改为周期性边界条件

## 🎯 应用示例

### 示例1: 标准Brusselator
```python
solver = BrusselatorSolver(A=1.0, B=3.0, D_u=0.2, D_v=0.1)
# 产生稳定的螺旋波或同心圆模式
```

### 示例2: 图灵模式
```python
solver = BrusselatorSolver(A=1.0, B=4.5, D_u=0.1, D_v=0.2)
# D_v > D_u时可能产生图灵不稳定性
```

### 示例3: 快速振荡
```python
solver = BrusselatorSolver(A=1.5, B=5.0, D_u=0.3, D_v=0.15)
# 高B值产生快速时间振荡
```

## 🔬 高级功能

### 自定义初始条件
```python
class MyCustomSolver(BrusselatorSolver):
    def initial_condition_u(self, x):
        # 自定义u的初始分布
        return self.A + custom_function(x)
    
    def initial_condition_v(self, x):
        # 自定义v的初始分布
        return self.B/self.A + custom_function(x)
```

### 多点分析
```python
# 分析多个空间点的时间演化
points = [[0.25, 0.25], [0.75, 0.75], [0.5, 0.5]]
time_series = []
for px, py in points:
    test_points = np.array([[px, py, t] for t in time_range])
    evolution = solver.predict(test_points)
    time_series.append(evolution)
```

## ⚠️ 注意事项

1. **计算资源**: 2D时空PDE计算较为耗时，建议在GPU上运行
2. **数值稳定性**: 非线性反应项可能导致训练不稳定，需要调整学习率
3. **边界处理**: 当前实现使用简化的零边界条件，实际应用可能需要周期性边界
4. **参数敏感性**: Brusselator系统对参数敏感，小的参数变化可能导致完全不同的行为

## 📖 参考资料

- [DeepXDE官方文档](https://deepxde.readthedocs.io/)
- [PINNs原始论文](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- [Brusselator反应扩散系统](https://en.wikipedia.org/wiki/Brusselator)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个求解器！

## 📄 许可证

MIT License