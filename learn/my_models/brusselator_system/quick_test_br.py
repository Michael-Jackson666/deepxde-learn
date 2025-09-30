#!/usr/bin/env python3
"""
Quick test for Brusselator system with English interface
简单快速测试布鲁塞尔自催化反应系统
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from brusselator_system import BrusselatorSolver

def quick_test():
    """快速测试 - 只用Adam优化器，不用L-BFGS"""
    print("🧪 Quick Brusselator System Test")
    print("Parameters: A=1.0, B=3.0, D_u=0.2, D_v=0.1")
    print("Domain: [0,1]² × [0,1] (shorter time)")
    
    # 创建求解器 - 更短的时间域
    solver = BrusselatorSolver(
        A=1.0, B=3.0,
        D_u=0.2, D_v=0.1,
        domain_size=1.0,
        time_end=1.0  # 时间域更短
    )
    
    # 设置几何和条件
    solver.setup_geometry_and_conditions()
    
    # 创建模型
    solver.create_model(
        num_domain=1000,    # 减少点数
        num_boundary=100,
        num_initial=100
    )
    
    # 快速训练 - 只用Adam
    print("🚀 Starting quick training (Adam only)...")
    solver.train(
        adam_iterations=2000,  # 减少迭代次数
        adam_lr=0.001,
        use_lbfgs=False  # 不使用L-BFGS
    )
    
    # 简单预测测试
    print("🔮 Testing prediction...")
    test_result = solver.predict([[0.5, 0.5, 0.5]])
    print(f"Prediction at (0.5, 0.5, 0.5): {test_result}")
    print(f"Shape: {test_result.shape}")
    
    # 可视化初始条件
    print("📊 Visualizing initial conditions...")
    solver.visualize_initial_conditions()
    
    print("✅ Quick test completed!")
    return solver

if __name__ == "__main__":
    solver = quick_test()