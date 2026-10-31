import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Importaciones de pymoo
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# ==============================================================================
# 1. DATOS EXPERIMENTALES DE ENTRADA
# ==============================================================================
t_exp = np.linspace(0, 2400, 41)  # 40 minutos en segundos
I_exp = np.array([
    3.0, 2.5, 2.4, 2.38, 2.37, 2.36, 2.36, 2.36, 2.36, 2.28,
    2.28, 2.27, 2.37, 2.37, 2.4, 2.39, 2.45, 2.47, 2.48, 2.49,
    2.5, 2.56, 2.58, 2.57, 2.57, 2.57, 2.6, 2.6, 2.63, 2.6,
    2.6, 2.58, 2.57, 2.58, 2.6, 2.57, 2.57, 2.58, 2.57, 2.56,
    2.56
])
Tw_exp = np.array([
    20.0, 23.73, 26.89, 29.63, 32.04, 34.2, 36.15, 37.93, 39.57, 41.09,
    42.5, 43.82, 45.06, 46.23, 47.34, 48.39, 49.39, 50.34, 51.25, 52.13,
    52.96, 53.77, 54.54, 55.28, 56.0, 56.7, 57.37, 58.02, 58.65, 59.26,
    59.85, 60.43, 60.99, 61.54, 62.07, 62.59, 63.09, 63.59, 64.07, 64.54,
    65.0
])

T_amb = 20.0  # [°C]

# Parámetros estáticos Peltier (Dymytrov 2017)
alpha = 0.0530   # [V/K]
R0 = 2.1000      # [Ohm]
Rq = 1.9500      # [K/W]
N = 2            # 2 celdas en serie

# Inercia térmica del sistema (Agua + Acero)
m_w, cp_w = 0.330, 4186.0        # 330 g de agua
m_acero, cp_acero = 0.150, 500.0 # 150 g de acero
C_sistema = m_w * cp_w + m_acero * cp_acero

# Interpolación continua de la corriente
I_func = interp1d(t_exp, I_exp, bounds_error=False, fill_value="extrapolate")


# ==============================================================================
# 2. FUNCIONES DE LA SIMULACIÓN FÍSICA
# ==============================================================================
def resolver_superficies_heating(Tw, I, R_contacto, R_qR):
    """Resuelve Tc y Th en modo calentamiento (Sistemas algebraico corregido)."""
    T0 = 273.15

    # Términos corregidos en la diagonal principal de la matriz A
    A11 = (1.0 / (N * R_qR)) + alpha * I + (1.0 / Rq)
    A12 = -1.0 / Rq
    B1 = (T_amb / (N * R_qR)) - alpha * T0 * I + 0.5 * (I**2) * R0

    A21 = -1.0 / Rq
    A22 = (1.0 / (N * R_contacto)) - alpha * I + (1.0 / Rq)
    B2 = (Tw / (N * R_contacto)) + alpha * T0 * I + 0.5 * (I**2) * R0

    A = np.array([[A11, A12], [A21, A22]])
    B = np.array([B1, B2])

    try:
        Tc, Th = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return T_amb, Tw, 0.0

    Qh = (Th - Tw) / (N * R_contacto)
    Q_util = N * Qh
    return Tc, Th, Q_util

def eduo_sistema(t, y, R_contacto, R_qR, UA):
    Tw = y[0]
    I_t = float(I_func(t))
    _, _, Q_util = resolver_superficies_heating(Tw, I_t, R_contacto, R_qR)
    Q_perdidas = UA * (Tw - T_amb)
    dTw_dt = (Q_util - Q_perdidas) / C_sistema
    return [dTw_dt]

def simular_curva(params):
    R_contacto, R_qR, UA = params
    sol = solve_ivp(
        eduo_sistema, 
        [t_exp[0], t_exp[-1]], 
        [Tw_exp[0]], 
        args=(R_contacto, R_qR, UA),
        t_eval=t_exp,
        method='RK45'
    )
    return sol.y[0]


# ==============================================================================
# 3. DEFINICIÓN DEL PROBLEMA EN PYMOO
# ==============================================================================
class PeltierIdentificationProblem(ElementwiseProblem):

    def __init__(self):
        # 3 variables a optimizar: [R_contacto, R_qR, UA]
        super().__init__(
            n_var=3,
            n_obj=1,
            n_ieq_constr=0,
            xl=np.array([0.001, 0.010, 0.001]),  # Límites inferiores
            xu=np.array([0.500, 1.000, 2.000])   # Límites superiores
        )

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            Tw_sim = simular_curva(x)
            if len(Tw_sim) != len(Tw_exp) or np.any(np.isnan(Tw_sim)):
                out["F"] = 1e6
            else:
                # RMSE como función de aptitud
                rmse = np.sqrt(np.mean((Tw_exp - Tw_sim)**2))
                out["F"] = rmse
        except Exception:
            out["F"] = 1e6


# ==============================================================================
# 4. EJECUCIÓN DE LA OPTIMIZACIÓN
# ==============================================================================
problem = PeltierIdentificationProblem()

# Puedes usar el Algoritmo Genético (GA) o Differential Evolution (DE)
algorithm = GA(pop_size=30) 
termination = get_termination("n_gen", 50)

print("Optimizando parámetros con pymoo...")
res = minimize(
    problem,
    algorithm,
    termination,
    seed=42,
    save_history=True,
    verbose=True
)

# Extraer parámetros óptimos
R_contacto_opt, R_qR_opt, UA_opt = res.X

print("\n==================================================")
print("     RESULTADOS DE LA OPTIMIZACIÓN (PYMOO)        ")
print("==================================================")
print(f"R_contacto óptimo : {R_contacto_opt:.4f} K/W")
print(f"R_qR óptimo       : {R_qR_opt:.4f} K/W")
print(f"UA óptimo         : {UA_opt:.4f} W/K")
print(f"RMSE mínimo       : {res.F[0]:.4f} °C")
print("==================================================")


# ==============================================================================
# 5. ANÁLISIS DE VARIABLES RESPECTO A SUS LÍMITES
# ==============================================================================
hist = res.history
hist_F = []
hist_X = []

for algo in hist:
    opt = algo.opt[0]
    hist_F.append(opt.F[0])
    hist_X.append(opt.X)

hist_X = np.array(hist_X)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

# Gráfico 1: Ajuste de Temperatura
Tw_sim_opt = simular_curva(res.X)
axs[0].plot(t_exp / 60, Tw_exp, 'ro', label='Datos Experimentales ($T_w$)')
axs[0].plot(t_exp / 60, Tw_sim_opt, 'b-', label='Simulado con pymoo')
axs[0].set_xlabel('Tiempo [min]')
axs[0].set_ylabel('Temperatura [°C]')
axs[0].set_title('Ajuste de Temperatura del Agua')
axs[0].grid(True, alpha=0.3)
axs[0].legend()

# Gráfico 2: Evolución de Variables entre sus Límites por Generación
generaciones = np.arange(1, len(hist_X) + 1)
axs[1].plot(generaciones, hist_X[:, 0] / problem.xu[0], label='$R_{contacto}$ / $xu_0$')
axs[1].plot(generaciones, hist_X[:, 1] / problem.xu[1], label='$R_{qR}$ / $xu_1$')
axs[1].plot(generaciones, hist_X[:, 2] / problem.xu[2], label='$UA$ / $xu_2$')
axs[1].set_xlabel('Generación')
axs[1].set_ylabel('Valor Relativo respecto al Límite Superior')
axs[1].set_title('Evolución de Parámetros entre sus Límites')
axs[1].set_ylim([0, 1.05])
axs[1].grid(True, alpha=0.3)
axs[1].legend()

plt.tight_layout()
plt.show()