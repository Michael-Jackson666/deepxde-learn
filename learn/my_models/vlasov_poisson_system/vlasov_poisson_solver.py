#!/usr/bin/env python3
"""
更正确的 1D1V Vlasov–Poisson (VP) 系统 PINN 求解器

思路：采用“离散速度节点”的网络输出设计，将速度维用 Nv 个固定节点离散，
网络仅以 (x, t) 为输入，输出 [f(x, v_0, t), ..., f(x, v_{Nv-1}, t), phi(x,t)] 共 Nv+1 个通道。

优点：
- 便于在残差中实现速度积分 ρ(x,t) ≈ \sum w_i f(x,v_i,t) 和速度导数 ∂f/∂v 的有限差分离散；
- 易于在 DeepXDE 的残差函数中封装完整的方程、初始条件与周期边界条件。

方程 (无量纲常用形式)：
- Vlasov:    ∂f/∂t + v ∂f/∂x + E ∂f/∂v = 0,  其中 E(x,t) = -∂phi/∂x
- Poisson:   ∂²phi/∂x² = 1 - ∫ f dv ≈ 1 - \sum_i w_i f_i

残差包含：
- 方程残差：对每个速度节点的 Vlasov 残差，以及 Poisson 残差
- 初始条件：t=0 时 f(x,v_i,0)（Maxwellian+扰动），phi(x,0)（与扰动一致或取 0）
- 边界条件：x 方向周期边界（对所有 f_i 与 phi 应用 PeriodicBC）

注意：
- 速度边界（v_min, v_max）处的 ∂f/∂v 使用一侧差分；若需要更强约束，可额外给 f_i 添加先验衰减 IC/BC。
"""

import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde


class VP1D1VDiscreteV:
    """1D1V Vlasov–Poisson PINN（离散速度）"""

    def __init__(
        self,
        x_domain=(-np.pi, np.pi),
        t_domain=(0.0, 10.0),
        v_domain=(-6.0, 6.0),
        num_velocity_nodes=24,
        amplitude=0.01,
        k_mode=1.0,
        neutral_density=1.0,
        use_gl_quadrature=True,
        seed=42,
    ):
        self.x_min, self.x_max = x_domain
        self.t_min, self.t_max = t_domain
        self.v_min, self.v_max = v_domain
        self.Nv = int(num_velocity_nodes)
        self.amplitude = float(amplitude)
        self.k = float(k_mode)
        self.n0 = float(neutral_density)
        self.use_gl_quadrature = bool(use_gl_quadrature)

        np.random.seed(seed)
        dde.config.set_random_seed(seed)

        # 速度节点与权重（用于速度积分与 ∂/∂v 有限差分）
        if self.use_gl_quadrature:
            # Gauss–Legendre 节点/权重（[-1,1] 映射到 [v_min, v_max]）
            xi, wi = np.polynomial.legendre.leggauss(self.Nv)
            self.v_nodes = 0.5 * (self.v_max - self.v_min) * xi + 0.5 * (self.v_max + self.v_min)
            self.v_weights = 0.5 * (self.v_max - self.v_min) * wi
        else:
            self.v_nodes = np.linspace(self.v_min, self.v_max, self.Nv)
            dv = (self.v_max - self.v_min) / (self.Nv - 1)
            self.v_weights = np.full(self.Nv, dv)

        # 预先计算每个节点的“局部 dv”用于一侧/中心差分
        self.dv_nodes = np.empty(self.Nv)
        self.dv_nodes[0] = self.v_nodes[1] - self.v_nodes[0]
        self.dv_nodes[-1] = self.v_nodes[-1] - self.v_nodes[-2]
        self.dv_nodes[1:-1] = (self.v_nodes[2:] - self.v_nodes[:-2]) / 2.0

        # 构造几何与时间域
        self.geom = dde.geometry.Interval(self.x_min, self.x_max)
        self.timedomain = dde.geometry.TimeDomain(self.t_min, self.t_max)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, self.timedomain)

        # 占位：DeepXDE 网络、数据与模型
        self.net = None
        self.data = None
        self.model = None

    # ----------------------------- 物理先验 ------------------------------
    def maxwellian(self, v, vt=1.0):
        return (1.0 / np.sqrt(2.0 * np.pi) / vt) * np.exp(-0.5 * (v / vt) ** 2)

    def f0(self, x, v):
        # Landau damping: Maxwellian + 小扰动 cos(kx)
        base = self.maxwellian(v, vt=1.0)
        return base * (1.0 + self.amplitude * np.cos(self.k * x))

    def phi0(self, x):
        # 初始电势，可取 0 或与密度扰动相容的解析近似
        return np.zeros_like(x)

    # ------------------------------ PDE 残差 ------------------------------
    def pde_system(self, xin, yout):
        """
        残差输出形状为 (N, Nv+1)：
        - 前 Nv 列：每个速度节点的 Vlasov 残差
        - 最后一列：Poisson 残差
        """
        # x,t 拆分
        # inputs: (N, 2) with columns [x, t]
        x = xin[:, 0:1]
        t = xin[:, 1:2]

        # 输出 y: (N, Nv+1) -> f_i, phi
        phi = yout[:, self.Nv : self.Nv + 1]

        # 计算空间/时间导数
        dphi_dx = dde.grad.jacobian(yout, xin, i=self.Nv, j=0)  # ∂phi/∂x
        d2phi_dx2 = dde.grad.hessian(yout, xin, component=self.Nv, i=0, j=0)  # ∂²phi/∂x²
        E = -dphi_dx  # 电场 E = -∂phi/∂x

        # 收集 f_i 以及其 ∂/∂x, ∂/∂t（用于 Vlasov），并构造 ∂f/∂v 的有限差分
        # yout[:, i:i+1] 是第 i 个速度节点上的 f_i(x,t)
        f_list = []
        dfdx_list = []
        dfdt_list = []
        for i in range(self.Nv):
            fi = yout[:, i : i + 1]
            f_list.append(fi)
            dfdx_list.append(dde.grad.jacobian(yout, xin, i=i, j=0))  # ∂f_i/∂x
            dfdt_list.append(dde.grad.jacobian(yout, xin, i=i, j=1))  # ∂f_i/∂t

        # 速度导数 ∂f/∂v 的离散近似：中心差分（边界用一侧差分）
        dfdv_list = [None] * self.Nv
        # 边界 i=0: forward difference
        dfdv_list[0] = (f_list[1] - f_list[0]) / (self.v_nodes[1] - self.v_nodes[0])
        # 中间点: central difference
        for i in range(1, self.Nv - 1):
            dfdv_list[i] = (f_list[i + 1] - f_list[i - 1]) / (
                self.v_nodes[i + 1] - self.v_nodes[i - 1]
            )
        # 边界 i=Nv-1: backward difference
        dfdv_list[-1] = (f_list[-1] - f_list[-2]) / (
            self.v_nodes[-1] - self.v_nodes[-2]
        )

        # Vlasov 残差：ri = ∂f_i/∂t + v_i ∂f_i/∂x + E ∂f/∂v|_i
        # E(x,t) 与每个 i 共享，形状对齐即可
        vlasov_residuals = []
        for i in range(self.Nv):
            vi = self.v_nodes[i]
            ri = dfdt_list[i] + vi * dfdx_list[i] + E * dfdv_list[i]
            vlasov_residuals.append(ri)

        # Poisson 残差：r_phi = ∂²phi/∂x² - (1 - ∑ w_i f_i)
        # 将速度积分 ∑ w_i f_i 组合出来
        integ = None
        for wi, fi in zip(self.v_weights, f_list):
            term = wi * fi
            integ = term if integ is None else integ + term
        poisson_residual = d2phi_dx2 - (1.0 - integ)

        # 返回残差列表：DeepXDE 更推荐返回 [r1, r2, ...] 的 list
        return vlasov_residuals + [poisson_residual]

    # --------------------------- 初始与边界条件 ---------------------------
    def _ic_f_component(self, vi):
        # 返回用于 dde.icbc.IC 的 callable；DeepXDE 传入 (x,t)
        def ic(x):
            # x: (N, 2) columns [x, t]
            xx = x[:, 0:1]
            return self.f0(xx, vi)
        return ic

    def _ic_phi(self):
        def ic(x):
            xx = x[:, 0:1]
            return self.phi0(xx)
        return ic

    def build_data(
        self,
        num_domain=4000,
        num_boundary=200,
        num_initial=800,
        train_distribution="uniform",
    ):
        # 周期边界：只在 x 方向（GeometryXTime），需要提供右边界谓词
        def boundary_r(x, on_boundary):
            # x: [x, t]，仅当位于 x = x_max 的边界处返回 True
            return on_boundary and dde.utils.isclose(x[0], self.x_max)

        bcs = []
        # 对 Nv 个 f_i 设置周期边界
        for i in range(self.Nv):
            bcs.append(dde.icbc.PeriodicBC(self.geomtime, i, boundary_r))
        # 对 phi 设置周期边界
        bcs.append(dde.icbc.PeriodicBC(self.geomtime, self.Nv, boundary_r))

        # 初始条件：t=0
        ics = []
        for i in range(self.Nv):
            ics.append(
                dde.icbc.IC(
                    self.geomtime,
                    self._ic_f_component(self.v_nodes[i]),
                    lambda _, on_initial: on_initial,
                    component=i,
                )
            )
        ics.append(
            dde.icbc.IC(
                self.geomtime,
                self._ic_phi(),
                lambda _, on_initial: on_initial,
                component=self.Nv,
            )
        )

        self.data = dde.data.TimePDE(
            self.geomtime,
            self.pde_system,
            bcs + ics,
            num_domain=num_domain,
            num_boundary=num_boundary,
            num_initial=num_initial,
            num_test=1000,
            train_distribution=train_distribution,
        )

    def build_network(self, hidden_sizes=(128, 128, 128, 128), activation="tanh"):
        # 输入 2 维 (x,t)，输出 Nv+1 维（Nv 个 f_i + 1 个 phi）
        layer_sizes = [2] + list(hidden_sizes) + [self.Nv + 1]
        self.net = dde.nn.FNN(layer_sizes, activation, "Glorot uniform")

    def build_model(self):
        assert self.data is not None and self.net is not None
        self.model = dde.Model(self.data, self.net)

    # --------------------------------- 训练 ---------------------------------
    def train(self, adam_iters=15000, adam_lr=1e-3, use_lbfgs=False):
        # 训练信息美化回调
        class PrettyLogger(dde.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.best = np.inf
            @staticmethod
            def _to_scalar_loss(loss_val):
                import numpy as _np
                # DeepXDE 可能返回：标量、ndarray、由分项损失组成的 list/tuple
                if isinstance(loss_val, (list, tuple)):
                    total = 0.0
                    for item in loss_val:
                        total += PrettyLogger._to_scalar_loss(item)
                    return float(total)
                if isinstance(loss_val, _np.ndarray):
                    if loss_val.ndim == 0:
                        return float(loss_val)
                    return float(_np.sum(loss_val))
                try:
                    return float(loss_val)
                except Exception:
                    return float(_np.array(loss_val).sum())
            def on_train_begin(self):
                print("\n================ 训练开始 ================")
                print(f"优化器: Adam  学习率: {adam_lr}")
                print(f"Adam迭代: {adam_iters}  是否LBFGS: {use_lbfgs}")
                print("========================================\n")
            def on_epoch_end(self):
                step = self.model.train_state.step
                loss_list = self.model.train_state.loss_train
                loss = PrettyLogger._to_scalar_loss(loss_list)
                if loss < self.best:
                    self.best = loss
                if step % max(1, adam_iters // 50) == 0 or step == 1:
                    bar_len = 30
                    prog = min(1.0, step / max(1, adam_iters))
                    filled = int(bar_len * prog)
                    bar = "█" * filled + "·" * (bar_len - filled)
                    print(f"Step {step:6d} | Loss {loss:9.3e} | Best {self.best:9.3e} | {bar} {prog*100:5.1f}%")
            def on_train_end(self):
                print("\n=============== Adam 完成 ===============\n")

        self.model.compile(optimizer="adam", lr=adam_lr)
        losshistory, train_state = self.model.train(iterations=adam_iters, callbacks=[PrettyLogger()])
        if use_lbfgs:
            self.model.compile("L-BFGS")
            print("开始 L-BFGS 微调...")
            losshistory, train_state = self.model.train()
            print("L-BFGS 完成。")
        return losshistory, train_state

    # --------------------------------- 预测 ---------------------------------
    def predict_f_phi(self, x, t):
        """返回 f(x, v_i, t) (Nv,) 与 phi(x,t) 标量。"""
        pts = np.array([[x, t]], dtype=float)
        out = self.model.predict(pts).reshape(-1)
        f_vals = out[: self.Nv]
        phi = out[self.Nv]
        return f_vals, phi


def quick_run():
    # 基本配置（Landau damping）
    solver = VP1D1VDiscreteV(
        x_domain=(-np.pi, np.pi),
        t_domain=(0.0, 2.0),
        v_domain=(-6.0, 6.0),
        num_velocity_nodes=24,
        amplitude=0.05,
        k_mode=1.0,
        neutral_density=1.0,
        use_gl_quadrature=True,
    )

    solver.build_data(
        num_domain=6000,
        num_boundary=400,
        num_initial=800,
        train_distribution="uniform",
    )
    solver.build_network(hidden_sizes=(128, 128, 128, 128), activation="tanh")
    solver.build_model()

    print(
        f"构建完成：Nv={solver.Nv}, x∈[{solver.x_min},{solver.x_max}], t∈[{solver.t_min},{solver.t_max}], v∈[{solver.v_min},{solver.v_max}]"
    )

    # 允许通过环境变量或此处参数自定义迭代次数
    adam_iters = 800
    losshistory, train_state = solver.train(adam_iters=adam_iters, adam_lr=8e-4, use_lbfgs=False)

    # 简单检查：提取一个点的 f 与 phi
    f_vals, phi_val = solver.predict_f_phi(x=0.0, t=0.0)
    print(f"样例预测：phi(0,0)≈{phi_val:.4e}，f(x=0,t=0) 的均值≈{np.mean(f_vals):.4e}")

    # 可视化：
    # 1) t 固定时的 phi(x,t)
    x_plot = np.linspace(solver.x_min, solver.x_max, 200)
    t_show = [0.0, (solver.t_min + solver.t_max) * 0.5, solver.t_max]
    plt.figure(figsize=(8, 4))
    for ts in t_show:
        pts = np.column_stack([x_plot, np.full_like(x_plot, ts)])
        out = solver.model.predict(pts)
        phi = out[:, solver.Nv]
        plt.plot(x_plot, phi, label=f"t={ts:.2f}")
    plt.title("电势 phi(x,t)")
    plt.xlabel("x")
    plt.ylabel("phi")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) 给定时间切片下的 f(x,v,t) 热力图
    nx, nv = 128, solver.Nv
    x_grid = np.linspace(solver.x_min, solver.x_max, nx)
    t_slice = min(solver.t_max, 0.8 * solver.t_max)
    F = np.zeros((nv, nx))
    for j, x0 in enumerate(x_grid):
        pts = np.array([[x0, t_slice]])
        out = solver.model.predict(pts).reshape(-1)
        F[:, j] = out[: solver.Nv]
    plt.figure(figsize=(8, 4))
    extent = [solver.x_min, solver.x_max, solver.v_min, solver.v_max]
    plt.imshow(F, aspect="auto", origin="lower", extent=extent, cmap="viridis")
    plt.colorbar(label="f(x,v,t)")
    plt.xlabel("x")
    plt.ylabel("v (离散节点)")
    plt.title(f"f(x,v,t) @ t={t_slice:.2f}")
    plt.tight_layout()
    plt.show()

    # 3) 训练损失曲线
    if hasattr(losshistory, "loss_train"):
        plt.figure(figsize=(6, 4))
        lt = losshistory.loss_train
        if isinstance(lt, (list, tuple)) and len(lt) and isinstance(lt[0], (list, tuple)):
            lt = [sum(x) for x in lt]
        plt.semilogy(lt, label="train")
        if getattr(losshistory, "loss_test", None) is not None:
            plt.semilogy(losshistory.loss_test, label="test")
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.title("Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="1D1V Vlasov–Poisson PINN (discrete-v)")
    parser.add_argument("--adam_iters", type=int, default=8000, help="Adam 迭代次数")
    parser.add_argument("--adam_lr", type=float, default=8e-4, help="Adam 学习率")
    parser.add_argument("--lbfgs", action="store_true", help="是否启用 L-BFGS")
    parser.add_argument("--Nv", type=int, default=24, help="速度离散节点数")
    parser.add_argument("--amp", type=float, default=0.05, help="扰动幅度")
    parser.add_argument("--k", type=float, default=1.0, help="模数 k")
    args = parser.parse_args()

    solver = VP1D1VDiscreteV(
        x_domain=(-np.pi, np.pi),
        t_domain=(0.0, 2.0),
        v_domain=(-6.0, 6.0),
        num_velocity_nodes=args.Nv,
        amplitude=args.amp,
        k_mode=args.k,
        neutral_density=1.0,
        use_gl_quadrature=True,
    )
    solver.build_data(num_domain=6000, num_boundary=400, num_initial=800, train_distribution="uniform")
    solver.build_network(hidden_sizes=(128, 128, 128, 128), activation="tanh")
    solver.build_model()
    print(
        f"构建完成：Nv={solver.Nv}, x∈[{solver.x_min},{solver.x_max}], t∈[{solver.t_min},{solver.t_max}], v∈[{solver.v_min},{solver.v_max}]"
    )
    losshistory, train_state = solver.train(adam_iters=args.adam_iters, adam_lr=args.adam_lr, use_lbfgs=args.lbfgs)

    # 训练后做同样的可视化
    x_plot = np.linspace(solver.x_min, solver.x_max, 200)
    t_show = [0.0, (solver.t_min + solver.t_max) * 0.5, solver.t_max]
    plt.figure(figsize=(8, 4))
    for ts in t_show:
        pts = np.column_stack([x_plot, np.full_like(x_plot, ts)])
        out = solver.model.predict(pts)
        phi = out[:, solver.Nv]
        plt.plot(x_plot, phi, label=f"t={ts:.2f}")
    plt.title("电势 phi(x,t)")
    plt.xlabel("x")
    plt.ylabel("phi")
    plt.legend()
    plt.tight_layout()
    plt.show()

    nx, nv = 128, solver.Nv
    x_grid = np.linspace(solver.x_min, solver.x_max, nx)
    t_slice = min(solver.t_max, 0.8 * solver.t_max)
    F = np.zeros((nv, nx))
    for j, x0 in enumerate(x_grid):
        pts = np.array([[x0, t_slice]])
        out = solver.model.predict(pts).reshape(-1)
        F[:, j] = out[: solver.Nv]
    plt.figure(figsize=(8, 4))
    extent = [solver.x_min, solver.x_max, solver.v_min, solver.v_max]
    plt.imshow(F, aspect="auto", origin="lower", extent=extent, cmap="viridis")
    plt.colorbar(label="f(x,v,t)")
    plt.xlabel("x")
    plt.ylabel("v (离散节点)")
    plt.title(f"f(x,v,t) @ t={t_slice:.2f}")
    plt.tight_layout()
    plt.show()

    if hasattr(losshistory, "loss_train"):
        plt.figure(figsize=(6, 4))
        lt = losshistory.loss_train
        if isinstance(lt, (list, tuple)) and len(lt) and isinstance(lt[0], (list, tuple)):
            lt = [sum(x) for x in lt]
        plt.semilogy(lt, label="train")
        if getattr(losshistory, "loss_test", None) is not None:
            plt.semilogy(losshistory.loss_test, label="test")
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.title("Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 支持命令行自定义；无参数时运行 quick_run
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        quick_run()


