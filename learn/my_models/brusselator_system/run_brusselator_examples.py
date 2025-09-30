#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brusselator反应扩散方程组求解器使用示例

这个脚本演示了如何使用BrusselatorSolver求解耦合反应扩散方程组
"""

from brusselator_system import BrusselatorSolver
import matplotlib.pyplot as plt

def example_basic_usage():
    """基础使用示例"""
    print("=" * 60)
    print("基础使用示例：标准参数求解")
    print("=" * 60)
    
    # 创建求解器（使用默认参数）
    solver = BrusselatorSolver(A=1.0, B=3.0, D_u=0.2, D_v=0.1)
    
    # 设置问题
    solver.setup_geometry_and_conditions()
    
    # 可视化初始条件
    solver.visualize_initial_conditions()
    
    # 创建模型（使用较少的采样点以加快训练）
    solver.create_model(num_domain=2000, num_boundary=150, num_initial=150)
    
    # 训练模型（较少迭代以节省时间）
    solver.train(adam_iterations=5000, adam_lr=0.001, use_lbfgs=True)
    
    # 可视化结果
    solver.visualize_evolution(times=[0.0, 0.5, 1.0, 1.5, 2.0])
    solver.analyze_center_evolution()
    
    return solver

def example_parameter_study():
    """参数研究示例"""
    print("=" * 60)
    print("参数研究示例：不同B值的影响")
    print("=" * 60)
    
    B_values = [1.5, 3.0, 4.5]  # 不同的B参数值
    
    plt.figure(figsize=(15, 5))
    
    for i, B in enumerate(B_values):
        print(f"\n--- 求解 B = {B} 的情况 ---")
        
        # 创建求解器
        solver = BrusselatorSolver(A=1.0, B=B, D_u=0.2, D_v=0.1, time_end=1.0)
        solver.setup_geometry_and_conditions()
        
        # 创建和训练模型（快速训练）
        solver.create_model(num_domain=1500, num_boundary=100, num_initial=100)
        solver.train(adam_iterations=3000, adam_lr=0.001, use_lbfgs=False)
        
        # 分析中心点演化
        time_points = np.linspace(0, 1.0, 50)
        center_points = np.array([[0.5, 0.5, t] for t in time_points])
        center_evolution = solver.predict(center_points)
        
        # 绘制时间演化
        plt.subplot(1, 3, i+1)
        plt.plot(time_points, center_evolution[:, 0], 'b-', linewidth=2, label='u')
        plt.plot(time_points, center_evolution[:, 1], 'r-', linewidth=2, label='v')
        plt.xlabel('时间 t')
        plt.ylabel('浓度')
        plt.title(f'B = {B}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def example_custom_conditions():
    """自定义初始条件示例"""
    print("=" * 60)
    print("自定义初始条件示例")
    print("=" * 60)
    
    class CustomBrusselatorSolver(BrusselatorSolver):
        """自定义初始条件的求解器"""
        
        def initial_condition_u(self, x):
            """自定义u的初始条件：高斯分布"""
            center_x, center_y = 0.5, 0.5
            sigma = 0.1
            gauss = np.exp(-((x[:, 0:1] - center_x)**2 + (x[:, 1:2] - center_y)**2) / (2 * sigma**2))
            return self.A + 0.5 * gauss
        
        def initial_condition_v(self, x):
            """自定义v的初始条件：环形分布"""
            center_x, center_y = 0.5, 0.5
            r = np.sqrt((x[:, 0:1] - center_x)**2 + (x[:, 1:2] - center_y)**2)
            ring = np.exp(-((r - 0.3)**2) / (2 * 0.05**2))
            return self.B/self.A + 0.3 * ring
    
    # 创建自定义求解器
    solver = CustomBrusselatorSolver(A=1.0, B=3.0, D_u=0.15, D_v=0.08)
    solver.setup_geometry_and_conditions()
    
    # 可视化自定义初始条件
    solver.visualize_initial_conditions()
    
    # 训练和可视化
    solver.create_model(num_domain=2500, num_boundary=180, num_initial=180)
    solver.train(adam_iterations=6000, adam_lr=0.0008, use_lbfgs=True)
    
    solver.visualize_evolution(times=[0.0, 0.3, 0.6, 1.0, 1.5])
    
    return solver

def example_analysis_tools():
    """分析工具示例"""
    print("=" * 60)
    print("分析工具示例：深入分析系统行为")
    print("=" * 60)
    
    # 创建并训练求解器
    solver = BrusselatorSolver(A=1.2, B=2.8, D_u=0.25, D_v=0.12)
    solver.setup_geometry_and_conditions()
    solver.create_model(num_domain=2000, num_boundary=150, num_initial=150)
    solver.train(adam_iterations=4000, adam_lr=0.001, use_lbfgs=True)
    
    # 1. 多点时间演化分析
    print("\n📊 多点时间演化分析...")
    points_of_interest = [
        [0.25, 0.25],  # 左下角
        [0.75, 0.25],  # 右下角
        [0.50, 0.50],  # 中心
        [0.25, 0.75],  # 左上角
        [0.75, 0.75],  # 右上角
    ]
    
    time_points = np.linspace(0, 2.0, 100)
    
    plt.figure(figsize=(12, 8))
    
    for i, (px, py) in enumerate(points_of_interest):
        test_points = np.array([[px, py, t] for t in time_points])
        evolution = solver.predict(test_points)
        
        plt.subplot(2, 3, i+1)
        plt.plot(time_points, evolution[:, 0], 'b-', label='u', linewidth=2)
        plt.plot(time_points, evolution[:, 1], 'r-', label='v', linewidth=2)
        plt.xlabel('时间 t')
        plt.ylabel('浓度')
        plt.title(f'点({px}, {py})')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 添加相空间分析
    plt.subplot(2, 3, 6)
    for i, (px, py) in enumerate(points_of_interest):
        test_points = np.array([[px, py, t] for t in time_points])
        evolution = solver.predict(test_points)
        plt.plot(evolution[:, 0], evolution[:, 1], linewidth=2, 
                label=f'({px}, {py})', alpha=0.7)
    
    plt.xlabel('u浓度')
    plt.ylabel('v浓度')
    plt.title('相空间轨迹')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 2. 空间模式分析
    print("\n🔍 空间模式分析...")
    times = [0.5, 1.0, 1.5, 2.0]
    
    for t in times:
        # 计算空间平均值和方差
        x_grid = np.linspace(0, 1, 30)
        y_grid = np.linspace(0, 1, 30)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        points_grid = np.stack([X_grid.flatten(), Y_grid.flatten(), 
                               np.full_like(X_grid.flatten(), t)], axis=1)
        
        prediction = solver.predict(points_grid)
        u_vals = prediction[:, 0]
        v_vals = prediction[:, 1]
        
        print(f"t={t}: u_均值={np.mean(u_vals):.3f}±{np.std(u_vals):.3f}, "
              f"v_均值={np.mean(v_vals):.3f}±{np.std(v_vals):.3f}")

if __name__ == "__main__":
    import numpy as np
    
    # 设置matplotlib支持中文
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("🧪 Brusselator反应扩散方程组求解器示例")
    print("请选择要运行的示例：")
    print("1. 基础使用示例")
    print("2. 参数研究示例")
    print("3. 自定义初始条件示例")
    print("4. 分析工具示例")
    print("5. 运行所有示例")
    
    choice = input("\n请输入选择 (1-5): ").strip()
    
    if choice == "1":
        solver = example_basic_usage()
    elif choice == "2":
        example_parameter_study()
    elif choice == "3":
        solver = example_custom_conditions()
    elif choice == "4":
        example_analysis_tools()
    elif choice == "5":
        print("\n🚀 运行所有示例...")
        solver1 = example_basic_usage()
        example_parameter_study()
        solver2 = example_custom_conditions()
        example_analysis_tools()
        print("\n✅ 所有示例运行完成！")
    else:
        print("❌ 无效选择，运行基础示例...")
        solver = example_basic_usage()
    
    print("\n🎉 示例运行完成！")