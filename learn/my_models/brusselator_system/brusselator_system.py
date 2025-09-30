#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brusselator反应扩散方程组求解器
使用DeepXDE求解耦合反应扩散方程组

方程组：
∂u/∂t = D_u∇²u + A - (B+1)u + u²v
∂v/∂t = D_v∇²v + Bu - u²v

作者：DeepXDE Tutorial
日期：2025年9月30日
"""

import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import time


class BrusselatorSolver:
    """Brusselator反应扩散方程组求解器"""
    
    def __init__(self, A=1.0, B=3.0, D_u=0.2, D_v=0.1, domain_size=1.0, time_end=2.0):
        """
        初始化求解器参数
        
        Args:
            A (float): 反应参数A
            B (float): 反应参数B  
            D_u (float): u的扩散系数
            D_v (float): v的扩散系数
            domain_size (float): 空间域大小 [0, domain_size]²
            time_end (float): 时间域终点 [0, time_end]
        """
        self.A = A
        self.B = B
        self.D_u = D_u
        self.D_v = D_v
        self.domain_size = domain_size
        self.time_end = time_end
        
        # 设置随机种子
        np.random.seed(42)
        dde.config.set_random_seed(42)
        
        print(f"🧪 Brusselator反应扩散方程组求解器")
        print(f"参数设置: A={A}, B={B}, D_u={D_u}, D_v={D_v}")
        print(f"求解域: [0,{domain_size}]² × [0,{time_end}]")
    
    def pde_system(self, x, y):
        """
        定义耦合反应扩散方程组
        
        Args:
            x: 输入坐标 [x, y, t] (N, 3)
            y: 神经网络输出 [u, v] (N, 2)
            
        Returns:
            PDE残差 [residual_u, residual_v] (N, 2)
        """
        # 提取u和v（y的两个输出分量）
        u = y[:, 0:1]  # 第一个输出分量
        v = y[:, 1:2]  # 第二个输出分量
        
        # 计算u的各种导数
        du_t = dde.grad.jacobian(y, x, i=0, j=2)   # ∂u/∂t
        du_xx = dde.grad.hessian(y, x, i=0, j=0)   # ∂²u/∂x²
        du_yy = dde.grad.hessian(y, x, i=0, j=1)   # ∂²u/∂y²
        
        # 计算v的各种导数
        dv_t = dde.grad.jacobian(y, x, i=1, j=2)   # ∂v/∂t
        dv_xx = dde.grad.hessian(y, x, i=1, j=0)   # ∂²v/∂x²
        dv_yy = dde.grad.hessian(y, x, i=1, j=1)   # ∂²v/∂y²
        
        # u方程: ∂u/∂t = D_u∇²u + A - (B+1)u + u²v
        residual_u = du_t - self.D_u * (du_xx + du_yy) - self.A + (self.B + 1) * u - u * u * v
        
        # v方程: ∂v/∂t = D_v∇²v + Bu - u²v  
        residual_v = dv_t - self.D_v * (dv_xx + dv_yy) - self.B * u + u * u * v
        
        return [residual_u, residual_v]
    
    def initial_condition_u(self, x):
        """u的初始条件: u(x,y,0) = A + 0.1*sin(2πx)cos(2πy)"""
        return self.A + 0.1 * np.sin(2 * np.pi * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2])
    
    def initial_condition_v(self, x):
        """v的初始条件: v(x,y,0) = B/A + 0.1*cos(2πx)sin(2πy)"""
        return self.B/self.A + 0.1 * np.cos(2 * np.pi * x[:, 0:1]) * np.sin(2 * np.pi * x[:, 1:2])
    
    def setup_geometry_and_conditions(self):
        """设置几何域和边界/初始条件"""
        # 定义时空域
        spatial_domain = dde.geometry.Rectangle([0, 0], [self.domain_size, self.domain_size])
        time_domain = dde.geometry.TimeDomain(0, self.time_end)
        self.geomtime = dde.geometry.GeometryXTime(spatial_domain, time_domain)
        
        # 边界条件函数
        def boundary_condition(x, on_boundary):
            return on_boundary
        
        def zero_boundary_u(x):
            return np.zeros((len(x), 1))
        
        def zero_boundary_v(x):
            return np.zeros((len(x), 1))
        
        # 创建边界条件和初始条件
        bc_u = dde.icbc.DirichletBC(self.geomtime, zero_boundary_u, boundary_condition, component=0)
        bc_v = dde.icbc.DirichletBC(self.geomtime, zero_boundary_v, boundary_condition, component=1)
        ic_u = dde.icbc.IC(self.geomtime, self.initial_condition_u, lambda _, on_initial: on_initial, component=0)
        ic_v = dde.icbc.IC(self.geomtime, self.initial_condition_v, lambda _, on_initial: on_initial, component=1)
        
        self.bcs_and_ics = [bc_u, bc_v, ic_u, ic_v]
        
        print("✅ 几何域和边界/初始条件设置完成")
    
    def create_model(self, num_domain=3000, num_boundary=200, num_initial=200, 
                    layer_sizes=[3, 64, 64, 64, 64, 2], activation="tanh"):
        """
        创建神经网络模型
        
        Args:
            num_domain (int): 域内采样点数
            num_boundary (int): 边界采样点数
            num_initial (int): 初始条件采样点数
            layer_sizes (list): 网络层大小
            activation (str): 激活函数
        """
        # 创建训练数据
        self.data = dde.data.TimePDE(
            self.geomtime,
            self.pde_system,
            self.bcs_and_ics,
            num_domain=num_domain,
            num_boundary=num_boundary,
            num_initial=num_initial,
            num_test=1000
        )
        
        # 构建神经网络
        self.net = dde.nn.FNN(layer_sizes, activation, "Glorot uniform")
        
        # 创建模型
        self.model = dde.Model(self.data, self.net)
        
        print("🧠 神经网络模型创建完成")
        print(f"网络结构: {layer_sizes}")
        print(f"域内点数: {num_domain}, 边界点数: {num_boundary}, 初始点数: {num_initial}")
    
    def train(self, adam_iterations=8000, adam_lr=0.0005, use_lbfgs=True):
        """
        训练模型
        
        Args:
            adam_iterations (int): Adam优化器迭代次数
            adam_lr (float): Adam学习率
            use_lbfgs (bool): 是否使用L-BFGS精细调优
        """
        print("🚀 开始训练耦合反应扩散方程组...")
        
        # 第一阶段：Adam训练
        self.model.compile(optimizer="adam", lr=adam_lr, metrics=["l2 relative error"])
        
        start_time = time.time()
        self.losshistory, self.train_state = self.model.train(iterations=adam_iterations)
        train_time = time.time() - start_time
        
        print(f"📊 Adam训练完成！ 用时: {train_time:.1f}秒")
        print(f"最终训练损失: {self.train_state.loss_train:.6f}")
        print(f"最终测试损失: {self.train_state.loss_test:.6f}")
        
        # 第二阶段：L-BFGS精细调优
        if use_lbfgs:
            print("\n🔧 开始L-BFGS精细调优...")
            self.model.compile("L-BFGS")
            self.losshistory, self.train_state = self.model.train()
            
            print("🎉 L-BFGS训练完成！")
            print(f"最终训练损失: {self.train_state.loss_train:.6f}")
            print(f"最终测试损失: {self.train_state.loss_test:.6f}")
    
    def predict(self, x_points):
        """
        预测给定点的u和v值
        
        Args:
            x_points: 输入点坐标 (N, 3)
            
        Returns:
            predictions: [u, v] 预测值 (N, 2)
        """
        return self.model.predict(x_points)
    
    def visualize_initial_conditions(self):
        """可视化初始条件"""
        x_test = np.linspace(0, self.domain_size, 50)
        y_test = np.linspace(0, self.domain_size, 50)
        X_test, Y_test = np.meshgrid(x_test, y_test)
        points_init = np.stack([X_test.flatten(), Y_test.flatten(), np.zeros_like(X_test.flatten())], axis=1)
        
        u_init = self.initial_condition_u(points_init).reshape(X_test.shape)
        v_init = self.initial_condition_v(points_init).reshape(X_test.shape)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        im1 = axes[0].contourf(X_test, Y_test, u_init, levels=20, cmap='viridis')
        axes[0].set_title('u初始条件 (t=0)')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].contourf(X_test, Y_test, v_init, levels=20, cmap='plasma')
        axes[1].set_title('v初始条件 (t=0)')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.show()
    
    def visualize_evolution(self, times=None, resolution=40):
        """
        可视化时空演化
        
        Args:
            times (list): 可视化的时间点
            resolution (int): 空间分辨率
        """
        if times is None:
            times = [0.0, 0.5, 1.0, 1.5, 2.0]
        
        x_vis = np.linspace(0, self.domain_size, resolution)
        y_vis = np.linspace(0, self.domain_size, resolution)
        X_vis, Y_vis = np.meshgrid(x_vis, y_vis)
        
        fig, axes = plt.subplots(2, len(times), figsize=(4*len(times), 8))
        
        for i, t in enumerate(times):
            T_vis = np.full_like(X_vis, t)
            points_vis = np.stack([X_vis.flatten(), Y_vis.flatten(), T_vis.flatten()], axis=1)
            
            prediction = self.predict(points_vis)
            u_pred = prediction[:, 0].reshape(X_vis.shape)
            v_pred = prediction[:, 1].reshape(X_vis.shape)
            
            # 绘制u的分布
            im1 = axes[0, i].contourf(X_vis, Y_vis, u_pred, levels=20, cmap='viridis')
            axes[0, i].set_title(f'u浓度分布 (t={t})')
            axes[0, i].set_xlabel('x')
            if i == 0:
                axes[0, i].set_ylabel('y')
            plt.colorbar(im1, ax=axes[0, i])
            
            # 绘制v的分布
            im2 = axes[1, i].contourf(X_vis, Y_vis, v_pred, levels=20, cmap='plasma')
            axes[1, i].set_title(f'v浓度分布 (t={t})')
            axes[1, i].set_xlabel('x')
            if i == 0:
                axes[1, i].set_ylabel('y')
            plt.colorbar(im2, ax=axes[1, i])
        
        plt.tight_layout()
        plt.show()
    
    def analyze_center_evolution(self):
        """分析中心点的时间演化和相空间轨迹"""
        # 中心点时间演化
        time_points = np.linspace(0, self.time_end, 100)
        center_points = np.array([[self.domain_size/2, self.domain_size/2, t] for t in time_points])
        center_evolution = self.predict(center_points)
        u_center = center_evolution[:, 0]
        v_center = center_evolution[:, 1]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. 训练历史
        axes[0].semilogy(self.losshistory.steps, self.losshistory.loss_train, 'b-', label='训练损失')
        axes[0].semilogy(self.losshistory.steps, self.losshistory.loss_test, 'r--', label='测试损失')
        axes[0].set_xlabel('训练步数')
        axes[0].set_ylabel('损失')
        axes[0].set_title('训练历史')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 中心点时间演化
        axes[1].plot(time_points, u_center, 'b-', linewidth=2, label='u(中心点)')
        axes[1].plot(time_points, v_center, 'r-', linewidth=2, label='v(中心点)')
        axes[1].set_xlabel('时间 t')
        axes[1].set_ylabel('浓度')
        axes[1].set_title('中心点浓度时间演化')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 相空间轨迹
        axes[2].plot(u_center, v_center, 'g-', linewidth=2, alpha=0.7)
        axes[2].scatter(u_center[0], v_center[0], c='red', s=100, marker='o', label='起点')
        axes[2].scatter(u_center[-1], v_center[-1], c='blue', s=100, marker='s', label='终点')
        axes[2].set_xlabel('u浓度')
        axes[2].set_ylabel('v浓度')
        axes[2].set_title('相空间轨迹 (u-v)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("📈 分析结果:")
        print(f"最终u浓度: {u_center[-1]:.3f}")
        print(f"最终v浓度: {v_center[-1]:.3f}")
        print(f"系统是否趋于稳态: {'是' if abs(u_center[-1] - u_center[-10]) < 0.01 else '否'}")
    
    def save_model(self, filename="brusselator_model"):
        """保存训练好的模型"""
        self.model.save(filename)
        print(f"💾 模型已保存为 {filename}")
    
    def load_model(self, filename="brusselator_model"):
        """加载预训练模型"""
        self.model.restore(filename)
        print(f"📂 模型已从 {filename} 加载")


def main():
    """主函数 - 演示如何使用BrusselatorSolver"""
    # 创建求解器实例
    solver = BrusselatorSolver(A=1.0, B=3.0, D_u=0.2, D_v=0.1)
    
    # 设置几何域和条件
    solver.setup_geometry_and_conditions()
    
    # 可视化初始条件
    solver.visualize_initial_conditions()
    
    # 创建模型
    solver.create_model(num_domain=3000, num_boundary=200, num_initial=200)
    
    # 训练模型
    solver.train(adam_iterations=8000, adam_lr=0.0005, use_lbfgs=True)
    
    # 可视化结果
    solver.visualize_evolution()
    solver.analyze_center_evolution()
    
    # 保存模型
    solver.save_model("brusselator_model")
    
    print("\n🎉 Brusselator反应扩散方程组求解完成！")


if __name__ == "__main__":
    # 设置matplotlib支持中文
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 运行主程序
    main()