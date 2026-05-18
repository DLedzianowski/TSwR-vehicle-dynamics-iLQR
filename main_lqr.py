import torch
from torch.autograd.functional import jacobian
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import CubicSpline

from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from reference_paths import generate_reference_path_oval, generate_reference_path_jajo, generate_reference_path_track


def compute_kappa(ref_path):
    """
    Oblicza krzywizny ścieżki z wektora punktów ref_path.x_ref, ref_path.y_ref.
    """
    x = np.array(ref_path.x_ref)
    y = np.array(ref_path.y_ref)
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    # Krzywizna geometryczna
    kappa = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return kappa

def get_kappa_seq(ref_path, s0, N, ds):
    kappa_seq = []
    s = s0

    for _ in range(N):
        kappa_seq.append(ref_path.get_kappa(s))
        s += ds

    return torch.tensor(kappa_seq, dtype=torch.float32)

def compute_v_ref(kappa, vehicle):
    a_lat_max = 0.5 * vehicle.a_lat_max
    # kappa_abs = np.abs(kappa) + 1e-3
    kappa_abs = max(abs(kappa), 1e-4)

    v = np.sqrt(a_lat_max / kappa_abs)

    return np.clip(v, 5.0, vehicle.v_max)

def get_x_ref_seq(ref_path, vehicle, s0, N, dt):
    x_ref_seq = torch.zeros((N, 8), dtype=torch.float32)
    s = s0

    for i in range(N):
        kappa = ref_path.get_kappa(s)

        v_ref = compute_v_ref(kappa, vehicle)

        T_ref = (vehicle.Cr0 + vehicle.Cr2 * v_ref**2) / vehicle.Cm
        delta_ref = np.arctan((vehicle.lf + vehicle.lr) * kappa)
        r_ref = v_ref * kappa

        x_ref_seq[i] = torch.tensor([
            s,
            0.0,
            0.0,
            v_ref,
            0.0,
            r_ref,
            delta_ref,
            T_ref
        ], dtype=torch.float32)

        s += v_ref * dt

    return x_ref_seq

class VehicleModel:
    def __init__(self, params):
        # Parametry pojazdu
        self.v_max = params.longitudinal.v_max
        self.v_min = params.longitudinal.v_min
        self.m = params.m            # masa pojazdu
        self.Iz = params.I_z          # moment bezwładności yaw
        self.lf = params.a          # odległość środka ciężkości do przodu
        self.lr = params.b          # odległość środka ciężkości do tyłu
        self.Cm = 8000.0         # stała silnika (longitudinal)
        self.Cr0 = 100.0         # współczynnik oporów toczenia
        self.Cr2 = 0.5        # współczynnik oporów aerodynamicznego
        self.p_tv = 1000.0       # torque vectoring
        # Parametry Pacejki (lateral force)
        tire = params.tire
        self.Bf = tire.p_ky1  # B parameter przednich opon
        self.Cf = tire.p_cy1  # C parameter przednich opon
        self.Df = tire.p_dy1  # D parameter przednich opon
        self.Br = tire.p_ky1  # B parameter tylnych opon
        self.Cr = tire.p_cy1  # C parameter tylnych opon
        self.Dr = tire.p_dy1  # D parameter tylnych opon
        self.g = 9.81                # przyspieszenie grawitacyjne

        F_NF = self.lr / (self.lf + self.lr) * self.m * self.g
        F_NR = self.lf / (self.lf + self.lr) * self.m * self.g
        FyF_max = F_NF * self.Df
        FyR_max = F_NR * self.Dr
        self.a_lat_max = (FyF_max + FyR_max) / self.m
        
        self.vx_dot = []
        self.vy_dot = []

    def dynamics(self, x, u, kappa):
        """
        Oblicza pochodne stanu dla pojazdu w zależności od prędkości.
        x = [s, n, mi, vx, vy, r, delta, T], u = [delta_dot, T_dot]
        kappa - krzywizna trajektorii w punkcie s
        """
        s, n, mi, vx, vy, r, delta, T = x
        delta_dot, T_dot = u
        κ = kappa.clone() if torch.is_tensor(kappa) else torch.tensor(kappa)

        # Longitudinal force       
        F_M = self.Cm * T
        F_roll = self.Cr0 * torch.sign(vx)
        F_drag = self.Cr2 * vx * torch.abs(vx)
        if torch.abs(vx) < 0.1 and F_M <= self.Cr0:
            F_roll = 0.0
        F_x = F_M - F_roll - F_drag

        # Próg przejścia między kinematyką a dynamiką
        v_switch = 5.0

        if torch.abs(vx) < v_switch:
            # ===== KINEMATYCZNY MODEL =====
            F_yF = torch.tensor(0.0)
            F_yR = torch.tensor(0.0)
            alphaF = torch.tensor(0.0)
            alphaR = torch.tensor(0.0)

            # yaw z geometrii
            r_kin = vx / (self.lf + self.lr) * torch.tan(delta)
            s_dot = (vx * torch.cos(mi)) / (1 - n * κ)
            n_dot = vx * torch.sin(mi)
            mi_dot = r_kin - κ * s_dot
            vx_dot = F_x / self.m
            vy_dot = torch.tensor(0.0)
            r_dot = r_kin - r

        else:
            # ===== DYNAMICZNY MODEL =====
            # Zapobiegamy dzieleniu przez zero w kątach poślizgu
            vx_safe = vx if torch.abs(vx) > 1e-3 else torch.sign(vx)*1e-3
            alphaF = torch.atan((vy + self.lf * r) / vx_safe) - delta
            alphaR = torch.atan((vy - self.lr * r) / vx_safe)

            F_NF = self.lr / (self.lf + self.lr) * self.m * self.g
            F_NR = self.lf / (self.lf + self.lr) * self.m * self.g
            F_yF = F_NF * self.Df * torch.sin(self.Cf * torch.atan(self.Bf * alphaF))
            F_yR = F_NR * self.Dr * torch.sin(self.Cr * torch.atan(self.Br * alphaR))

            r_target = torch.tan(delta) * vx / (self.lf + self.lr)
            M_tv = self.p_tv * (r_target - r)

            s_dot = (vx * torch.cos(mi) - vy * torch.sin(mi)) / (1 - n * κ)
            n_dot = vx * torch.sin(mi) + vy * torch.cos(mi)
            mi_dot = r - κ * s_dot
            vx_dot = (F_x - F_yF * torch.sin(delta) + self.m * vy * r) / self.m
            vy_dot = (F_yR + F_yF * torch.cos(delta) - self.m * vx * r) / self.m
            r_dot = (F_yF * self.lf * torch.cos(delta) - F_yR * self.lr + M_tv) / self.Iz

        self.vx_dot.append(vx_dot.item())
        self.vy_dot.append(vy_dot.item())
        return torch.stack([s_dot, n_dot, mi_dot, vx_dot, vy_dot, r_dot, delta_dot, T_dot])

    def step(self, x, u, kappa, dt):
        """Przejście stanu o jeden krok czasu dt (Euler)."""
        dx = self.dynamics(x, u, kappa)
        x_next = x + dt * dx
        return x_next

    def linearize(self, x, u, kappa, dt):
        def f_x(x_):
            return self.dynamics(x_, u, kappa)

        def f_u(u_):
            return self.dynamics(x, u_, kappa)

        A_cont = jacobian(f_x, x)
        B_cont = jacobian(f_u, u)

        A = torch.eye(x.shape[0]) + dt * A_cont
        B = dt * B_cont

        return A, B


class ILQRController:
    def __init__(self, model, Q, R, Qf, N, dt, kappa_seq, x_ref_seq, nx = 8, nu = 2):
        self.model = model
        # macierz wag stanu
        if isinstance(Q, torch.Tensor):
            self.Q = Q.clone()
        else:
            self.Q = torch.tensor(Q, dtype=torch.float32)    

        # macierz wag sterowania
        if isinstance(R, torch.Tensor):
            self.R = R.clone()
        else:
            self.R = torch.tensor(R, dtype=torch.float32)    

        # macierz wag końcowego stanu
        if isinstance(Qf, torch.Tensor):
            self.Qf = Qf.clone()
        else:
            self.Qf = torch.tensor(Qf, dtype=torch.float32)  

        self.N = N
        self.dt = dt
        # Sekwencja krzywizn i stanów referencji
        self.kappa_seq = (
            kappa_seq.clone()
            if isinstance(kappa_seq, torch.Tensor)
            else torch.tensor(kappa_seq, dtype=torch.float32)
        )
        if isinstance(x_ref_seq, torch.Tensor):
            self.x_ref_seq = x_ref_seq.clone()
        else:
            self.x_ref_seq = torch.tensor(x_ref_seq, dtype=torch.float32)    
        self.nx = nx
        self.nu = nu
        # self.Rd = 10 * torch.eye(self.nu)   
        self.Rd = torch.diag(torch.tensor([
            10.0,      # steering smoothness
            10.0     # throttle smoothness
        ], dtype=torch.float32))

    def rollout(self, x0, u_seq):
        """Symuluje trajektorię przy danych sterowaniach (sekwencja długości N)."""
        x_seq = torch.zeros((self.N + 1, self.nx), dtype=torch.float32)
        if isinstance(x0, torch.Tensor):
            x = x0.clone()
        else:
            x = torch.tensor(x0, dtype=torch.float32)    

        x_seq[0] = x
        for k in range(self.N):
            x = self.model.step(x, u_seq[k], self.kappa_seq[k], self.dt)
            x_seq[k + 1] = x

        return x_seq

    def linearize_trajectory(self, x_seq, u_seq):
        """Liniowo przybliża całą trajektorię i zwraca listy macierzy A_k, B_k."""
        A_seq = []
        B_seq = []
        for k in range(self.N):
            A, B = self.model.linearize(x_seq[k], u_seq[k], self.kappa_seq[k], self.dt)
            A_seq.append(A)
            B_seq.append(B)
        return A_seq, B_seq

    def stage_cost(self, x, u, x_ref, u_prev):
        dx = x - x_ref
        du = u - u_prev
        return (dx @ self.Q @ dx) + (u @ self.R @ u) + (du @ self.Rd @ du) + self.track_penalty(x[1],x[3])    
    
    def terminal_cost(self, x):
        dx = x - self.x_ref_seq[-1]
        return dx @ self.Qf @ dx
    
    def compute_total_cost(self, x_seq, u_seq):
        """Sumaryczny koszt od 0 do N (wraz z terminalnym na N)."""
        cost = torch.tensor(0.0, dtype=torch.float32)

        for k in range(self.N):
            xk = x_seq[k]
            uk = u_seq[k]

            u_prev = u_seq[k-1] if k > 0 else torch.zeros_like(uk)

            cost += self.stage_cost(xk, uk, self.x_ref_seq[k], u_prev)
        cost += self.terminal_cost(x_seq[self.N])

        return cost

    def track_penalty(self, n, vx):
        limit = 3.0
        excess = torch.relu(torch.abs(n) - limit)
        v_over = torch.clamp(vx - v_ref, min=0.0)
        return (1000.0 * excess**2) + (1000.0 * v_over**3)
        
    def backward_pass(self, A_seq, B_seq, x_seq, u_seq):
        """Backward pass obliczający zyski zwrotne K i korekcje k dla każdej próbki."""
        n = self.nx
        m = self.nu
        K = [torch.zeros((m, n)) for _ in range(self.N)]
        k_vec = [torch.zeros(m) for _ in range(self.N)]
        # Pochodne wartości w punkcie terminalnym
        Vx = 2 * self.Qf @ (x_seq[self.N] - self.x_ref_seq[-1])
        Vxx = 2 * self.Qf
        # Iteracja wstecz
        for k in reversed(range(self.N)):
            xk = x_seq[k]
            uk = u_seq[k]
            # Gradienty kosztu
            u_prev = u_seq[k-1] if k > 0 else torch.zeros_like(uk)
            Lx = 2 * self.Q @ (xk - self.x_ref_seq[k])
            Lu = 2 * self.R @ uk + 2 * self.Rd @ (uk - u_prev)
            Lxx = 2 * self.Q
            Luu = 2 * self.R + 2 * self.Rd
            # Oblicz Q-funkcje
            A = A_seq[k]
            B = B_seq[k]
            Qx = Lx + A.T @ Vx
            Qu = Lu + B.T @ Vx
            Qxx = Lxx + A.T @ Vxx @ A
            Quu = Luu + B.T @ Vxx @ B
            Qux = B.T @ Vxx @ A

            # Oblicz optymalne K,k (gainy sprzężenia, feedforward)
            reg = 1e-4 * torch.eye(self.nu)
            Quu_reg = Quu + reg
            inv_Quu = torch.inverse(Quu_reg)
            # inv_Quu = torch.inverse(Quu)
            k_opt = -inv_Quu @ Qu
            K_opt = -inv_Quu @ Qux
            k_vec[k] = k_opt
            K[k] = K_opt
            # Zaktualizuj Vx, Vxx 
            Vx = Qx + K_opt.T @ Quu @ k_opt  # te dwa wyrażenia dają Vx = Qx - K^T Qu (równoważne)
            Vxx = Qxx + K_opt.T @ Quu @ K_opt  # Vxx = Qxx - K^T Quu K
        return K, k_vec
    
    def forward_pass(self, x0, x_seq, u_seq, K, k_vec, alpha):
        if isinstance(x0, torch.Tensor):
            x = x0.clone()
        else:
            x = torch.tensor(x0, dtype=torch.float32)    

        x_new_seq = torch.zeros_like(x_seq)
        u_new_seq = torch.zeros_like(u_seq)

        x_new_seq[0] = x

        cost = 0.0

        for k in range(self.N):
            dx = x - x_seq[k]

            u = u_seq[k] + alpha * k_vec[k] + K[k] @ dx
            u = torch.clamp(u,torch.tensor([-1.0, -10.0], dtype=torch.float32),torch.tensor([1.0, 10.0], dtype=torch.float32))

            u_new_seq[k] = u

            x = self.model.step(x, u, self.kappa_seq[k], self.dt)
            x_new_seq[k + 1] = x
            u_prev = u_seq[k-1] if k > 0 else torch.zeros_like(u)
            cost += self.stage_cost(x, u, self.x_ref_seq[k], u_prev)

        cost += self.terminal_cost(x_new_seq[self.N])

        return x_new_seq, u_new_seq, cost
        
    def optimize(self, x0, u_init, max_iters=10):
        u_seq = u_init.clone()
        best_cost = float('inf')
        best_x_seq = None
        best_u_seq = None

        for it in range(max_iters):
            # 1. rollout trajektorii
            x_seq = self.rollout(x0, u_seq)
            # 2. linearizacja wokół bieżącej trajektorii
            A_seq, B_seq = self.linearize_trajectory(x_seq, u_seq)
            # 3. backward pass -> oblicz K,k
            K, k_vec = self.backward_pass(A_seq, B_seq, x_seq, u_seq)
            # 4. line search w forward pass
            found_improve = False

            for alpha in [1.0, 0.5, 0.25, 0.1]:
                x_new_seq, u_new_seq, new_cost = self.forward_pass(x0, x_seq, u_seq, K, k_vec,alpha)
                print(f"iter {it}, "f"alpha {alpha}, "f"u norm {torch.norm(u_new_seq)}, "f"cost = {new_cost}")
                
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_x_seq = x_new_seq
                    best_u_seq = u_new_seq
                    u_seq = u_new_seq.clone()
                    x_seq = x_new_seq.clone()
                    found_improve = True
                    break

            if not found_improve:
                # brak poprawy
                break

            u_seq = best_u_seq.clone()

        if best_x_seq is None or best_u_seq is None:
            return x_seq, u_seq

        return best_x_seq, best_u_seq

class ReferencePath:
    def __init__(self, x_ref, y_ref):
        self.x_ref = np.asarray(x_ref)
        self.y_ref = np.asarray(y_ref)

        # zamknięcie toru
        if np.hypot(self.x_ref[0] - self.x_ref[-1],
                    self.y_ref[0] - self.y_ref[-1]) > 1e-6:
            self.x_ref = np.append(self.x_ref, self.x_ref[0])
            self.y_ref = np.append(self.y_ref, self.y_ref[0])

        # długość łuku
        self.s_ref = self._compute_s()
        self.length = self.s_ref[-1]

        # spline pozycji
        self.x_spline = CubicSpline(self.s_ref, self.x_ref)
        self.y_spline = CubicSpline(self.s_ref, self.y_ref)

        # pochodne do heading i krzywizny
        dx = self.x_spline.derivative()(self.s_ref)
        dy = self.y_spline.derivative()(self.s_ref)

        ddx = self.x_spline.derivative(2)(self.s_ref)
        ddy = self.y_spline.derivative(2)(self.s_ref)

        # heading
        self.psi_ref = np.arctan2(dy, dx)
        self.psi_spline = CubicSpline(self.s_ref, self.psi_ref)

        # krzywizna
        kappa = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        self.kappa_spline = CubicSpline(self.s_ref, kappa)

    def _compute_s(self):
        ds = np.hypot(np.diff(self.x_ref), np.diff(self.y_ref))
        return np.concatenate(([0.0], np.cumsum(ds)))

    def wrap_s(self, s):
        return s % self.length

    def get_reference(self, s):
        s = self.wrap_s(s)
        return (
            float(self.x_spline(s)),
            float(self.y_spline(s)),
            float(self.psi_spline(s))
        )

    def get_kappa(self, s):
        s = self.wrap_s(s)
        return float(self.kappa_spline(s))

    def to_global(self, s, n, mi):
        xr, yr, psi_r = self.get_reference(s)

        X = xr - n * np.sin(psi_r)
        Y = yr + n * np.cos(psi_r)
        psi = psi_r + mi

        return X, Y, psi

def plot_ilqr(ref_path, traj, v_ref_log, vehicle):
    traj = np.array(traj)

    n_list = []
    X, Y = [], []
    vx_list, vy_list, delta_list = [], [], []

    for xi in traj:
        s, n, mi = xi[0:3]
        vx, vy = xi[3:5]
        delta = xi[6]

        Xi, Yi, _ = ref_path.to_global(s, n, mi)
        n_list.append(n)
        X.append(Xi)
        Y.append(Yi)
        vx_list.append(vx)
        vy_list.append(vy)
        delta_list.append(delta)

    # --- Trajektoria ---
    plt.figure(figsize=(6,6))
    plt.plot(ref_path.x_ref, ref_path.y_ref, '--', label='ref')
    plt.plot(X, Y, label='trajectory')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title("Trajectory")
    plt.xlim(-500, 1000)
    plt.ylim(-1750, 500)
    plt.savefig('data/trajectory.png', dpi=150)
    fig = plt.gcf()
    with open("data/trajectory.pkl", "wb") as f:
        pickle.dump(fig, f)
    plt.show(block=False)

    t = np.arange(len(traj))
    fig, axs = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
    min_len = min(len(v_ref_log), len(vx_list))
    axs[0].plot(t[:min_len], v_ref_log[:min_len], label="v_ref")
    axs[0].plot(t, vx_list, label="vx")
    axs[0].legend()
    axs[0].set_ylabel("velocity")
    axs[0].set_ylim(0, 50)
    axs[0].grid(True)

    axs[1].plot(t, vy_list)
    axs[1].set_ylabel("vy")
    axs[1].grid(True)

    axs[2].plot(t, delta_list)
    axs[2].set_ylabel("delta")
    axs[2].grid(True)

    axs[3].plot(t, n_list)
    axs[3].set_ylabel("error")
    axs[3].set_xlabel("step")
    axs[3].grid(True)

    plt.tight_layout()
    plt.savefig('data/data.png', dpi=150)
    with open("data/data.pkl", "wb") as f:
        pickle.dump(fig, f)

    plt.show(block=False)
    
    # --- Acceleration phase plot ---
    min_len_acc = min(len(vehicle.vx_dot), len(vehicle.vy_dot), len(vx_list))

    vx_dot_arr = np.array(vehicle.vx_dot[:min_len_acc])
    vy_dot_arr = np.array(vehicle.vy_dot[:min_len_acc])
    speed_arr = np.array(vx_list[:min_len_acc])

    plt.figure(figsize=(7, 6))

    sc = plt.scatter(
        vx_dot_arr,
        vy_dot_arr,
        c=speed_arr,
        cmap='YlGnBu',
        s=15
    )

    plt.xlabel("vx_dot [m/s²]")
    plt.ylabel("vy_dot [m/s²]")
    plt.title("Acceleration phase plot")
    plt.grid(True)

    cbar = plt.colorbar(sc)
    cbar.set_label("vx [m/s]")

    plt.tight_layout()
    plt.savefig("data/acc_phase.png", dpi=150)

    fig = plt.gcf()
    with open("data/acc_phase.pkl", "wb") as f:
        pickle.dump(fig, f)

    plt.show()
    

params = parameters_vehicle1()
vehicle = VehicleModel(params)
s = np.linspace(0, 2*np.pi, 1402)
# x_ref, y_ref = generate_reference_path_jajo(s)
# x_ref, y_ref = generate_reference_path_oval()
x_ref, y_ref = generate_reference_path_track(s)
ref_path = ReferencePath(x_ref, y_ref)

nx = 8
nu = 2
max_iters = 10
N = 17
v_ref = vehicle.v_max
dt = 0.05
Q = np.diag([
    0.0,     # s
    3000.0,  # n
    500.0,   # μ
    50.0,    # vx
    100.0,   # vy
    500.0,   # r
    100.0,   # delta
    0.0     # T
])
R = np.diag([1, 0.1])
Qf = Q * 5
x0 = np.array([0.0, 0.0, 0.0, 25.0, 0.0, 0.0, 0.0, 0.0])
x0 = torch.tensor(x0, dtype=torch.float32)
x = x0.clone()
u_init = [np.array([0.0, 0.0]) for _ in range(N)]

u_init = torch.zeros((N, 2), dtype=torch.float32)
Q = torch.tensor(Q, dtype=torch.float32)
R = torch.tensor(R, dtype=torch.float32)
Qf = torch.tensor(Qf, dtype=torch.float32)
u_init = torch.zeros((N, nu), dtype=torch.float32)

kappa_seq = compute_kappa(ref_path)[:N]
kappa_seq = torch.tensor(kappa_seq, dtype=torch.float32)
x_ref_seq = get_x_ref_seq(ref_path, vehicle, x[0].item(), N, dt)
ilqr = ILQRController(vehicle, Q, R, Qf, N, dt, kappa_seq, x_ref_seq, nx, nu)

v_ref_log = [x[3].item()]
trajectory = [x.numpy()]

t = 0
try:
    for t in range(4200):
        print(f"------Time step {t}------")
        try:
            kappa_seq = get_kappa_seq(ref_path, x[0].item(), N, v_ref * dt)
            x_ref_seq = get_x_ref_seq(ref_path, vehicle, x[0].item(), N, dt)

            ilqr.kappa_seq = kappa_seq
            ilqr.x_ref_seq = x_ref_seq

            x_opt, u_opt = ilqr.optimize(x, u_init, max_iters)
            u = u_opt[0]

            x = vehicle.step(x, u, kappa_seq[0], dt)
            v_ref_log.append(x_ref_seq[0, 3].item())
            trajectory.append(x.detach().numpy())
            u_init = torch.vstack([u_opt[1:], u_opt[-1:]])
            print("error: ", x[1])
        except Exception as e:
            print(f"Błąd w kroku {t}: {e}")
            break
    
except KeyboardInterrupt:
    pass

plot_ilqr(ref_path, trajectory, v_ref_log, vehicle)