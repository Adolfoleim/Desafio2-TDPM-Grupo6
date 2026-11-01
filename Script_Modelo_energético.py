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
# 1. CONFIGURACIÓN Y DATOS EXPERIMENTALES DE ENTRADA
# ==============================================================================
# SELECCIÓN DE MODO: Cambiar entre "heating" o "cooling"
MODO_OPERACION = "heating"  # Opción: "heating" o "cooling"

# --- Datos Experimentales ---
t_exp = np.linspace(0, 2700, 46)  # 40 minutos en segundos (41 puntos)

if MODO_OPERACION == "heating":
    # Datos de calentamiento (Agua para maruchan)
    I_exp = np.array([
        2.7, 2.67, 2.66, 2.56, 2.53, 2.53, 2.54, 2.57, 2.57, 2.57, 2.56, 2.55,
        2.53, 2.48, 2.47, 2.47, 2.43, 2.43, 2.42, 2.43, 2.41, 2.3, 2.39, 2.36,
        2.36, 2.3, 2.29, 2.25, 2.19, 2.19, 2.16, 2.16, 2.13, 2.09, 2.09, 2.09,
        2.09, 2.09, 2.09, 2.09, 2.08, 2.06, 2.06, 2.05, 2.06, 2.05
    ])
    Tw_exp = np.array([
        21.7, 21.9, 22.6, 23.6, 24.7, 25.6, 26.4, 27.3, 28.1, 29.0, 29.6, 30.2,
        31.2, 32.0, 33.3, 33.5, 33.8, 34.6, 35.2, 36.2, 37.0, 37.8, 38.6, 39.5,
        40.2, 41.0, 43.2, 44.1, 44.9, 45.7, 46.6, 47.4, 48.3, 49.5, 49.9, 50.8,
        51.5, 52.2, 53.0, 53.8, 54.4, 55.1, 56.0, 56.6, 57.2, 57.8
    ])
    T_amb = 21.5  # [°C]
    
    # Inercia térmica modo calentamiento (Agua + Recipiente acero)
    m_w, cp_w = 0.330, 4186.0        # 250 g de agua
    m_acero, cp_acero = 0.207, 500.0 # 150 g de acero
    C_sistema = m_w * cp_w + m_acero * cp_acero

else:  # "cooling"
    # Tiempo en segundos (0 a 40 minutos = 0 a 2400 segundos, 41 puntos)
    t_exp = np.linspace(0, 2400, 41)

    # Corriente medida / corregida [A]
    I_exp = np.array([
        1.830000, 1.871667, 1.913333, 1.955000, 1.996667, 2.038333, 2.080000, 2.121667,
        2.163333, 2.205000, 2.246667, 2.288333, 2.330000, 2.371667, 2.413333, 2.455000,
        2.496667, 2.538333, 2.580000, 2.621667, 2.663333, 2.705000, 2.746667, 2.788333,
        2.830000, 2.871667, 2.913333, 2.955000, 2.996667, 3.038333, 3.080000, 3.121667,
        3.163333, 3.205000, 3.246667, 3.288333, 3.330000, 3.371667, 3.413333, 3.455000,
        3.496667
    ])

    # Temperatura del agua/baño en modo enfriamiento [°C]
    Tw_exp = np.array([
        22.8, 16.8, 16.7, 16.6, 16.6, 16.3, 16.1, 15.8, 15.5, 15.2,
        15.1, 14.8, 14.4, 14.2, 13.9, 13.7, 13.4, 13.2, 13.0, 12.8,
        12.6, 12.3, 12.1, 11.9, 11.6, 11.4, 11.2, 11.0, 10.8, 10.6,
        10.4, 10.2, 10.0, 10.3,  9.8,  9.3,  9.2,  9.2,  9.0,  8.8,
        8.7
    ])
    T_amb = 21.5  # [°C]
    
    # Inercia térmica modo enfriamiento (Bebida + Lata + Baño de agua + Recipiente)
    m_bebida, cp_bebida = 0.355, 3900.0  # Bebida en lata (355 ml)
    m_lata, cp_lata = 0.015, 900.0       # Lata de aluminio (15 g)
    m_bano, cp_agua = 0.300, 4186.0      # 300 g de agua de acople
    m_acero, cp_acero = 0.207, 500.0     # Recipiente de acero (150 g)
    C_sistema = (m_bebida * cp_bebida + m_lata * cp_lata + 
                 m_bano * cp_agua + m_acero * cp_acero)

# Parámetros estáticos Peltier (Dymytrov 2017)
alpha = 0.0530   # [V/K]
R0 = 2.1000      # [Ohm]
Rq = 1.9500      # [K/W]
N = 2            # 2 celdas en serie

# Interpolación continua de la corriente
I_func = interp1d(t_exp, I_exp, bounds_error=False, fill_value="extrapolate")


# ==============================================================================
# 2. FUNCIONES DE LA SIMULACIÓN FÍSICA
# ==============================================================================
def resolver_superficies(Tw, I, R_contacto, R_qR, modo="heating"):
    """Resuelve Tc y Th según el modo de operación ('heating' o 'cooling')."""
    T0 = 273.15

    if modo == "heating":
        # Cara caliente en contacto con el fluido (Tw), cara fría con ambiente (T_amb)
        A11 = (1.0 / (N * R_qR)) + alpha * I + (1.0 / Rq)
        A12 = -1.0 / Rq
        B1  = (T_amb / (N * R_qR)) - alpha * T0 * I + 0.5 * (I**2) * R0

        A21 = -1.0 / Rq
        A22 = (1.0 / (N * R_contacto)) - alpha * I + (1.0 / Rq)
        B2  = (Tw / (N * R_contacto)) + alpha * T0 * I + 0.5 * (I**2) * R0

        A = np.array([[A11, A12], [A21, A22]])
        B = np.array([B1, B2])

        try:
            Tc, Th = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            return T_amb, Tw, 0.0

        Qh = (Th - Tw) / (N * R_contacto)
        Q_flujo = N * Qh  # Calor entregado al fluido (>0)
        return Tc, Th, Q_flujo

    else:  # "cooling"
        # Cara fría en contacto con el fluido (Tw), cara caliente con ambiente (T_amb)
        A11 = (1.0 / (N * R_contacto)) + alpha * I + (1.0 / Rq)
        A12 = -1.0 / Rq
        B1  = (Tw / (N * R_contacto)) - alpha * T0 * I + 0.5 * (I**2) * R0

        A21 = -1.0 / Rq
        A22 = (1.0 / (N * R_qR)) - alpha * I + (1.0 / Rq)
        B2  = (T_amb / (N * R_qR)) + alpha * T0 * I + 0.5 * (I**2) * R0

        A = np.array([[A11, A12], [A21, A22]])
        B = np.array([B1, B2])

        try:
            Tc, Th = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            return T_amb, Tw, 0.0

        Qc = (Tw - Tc) / (N * R_contacto)
        Q_flujo = N * Qc  # Calor extraído del fluido (>0)
        return Tc, Th, Q_flujo

def eduo_sistema(t, y, R_contacto, R_qR, UA, modo="heating"):
    Tw = y[0]
    I_t = float(I_func(t))
    _, _, Q_flujo = resolver_superficies(Tw, I_t, R_contacto, R_qR, modo=modo)
    Q_perdidas = UA * (Tw - T_amb)

    if modo == "heating":
        dTw_dt = (Q_flujo - Q_perdidas) / C_sistema
    else:  # "cooling"
        dTw_dt = (-Q_flujo - Q_perdidas) / C_sistema

    return [dTw_dt]

def simular_curva(params, modo="heating"):
    R_contacto, R_qR, UA = params
    sol = solve_ivp(
        eduo_sistema, 
        [t_exp[0], t_exp[-1]], 
        [Tw_exp[0]], 
        args=(R_contacto, R_qR, UA, modo),
        t_eval=t_exp,
        method='RK45'
    )
    return sol.y[0]


# ==============================================================================
# 3. DEFINICIÓN DEL PROBLEMA EN PYMOO
# ==============================================================================
class PeltierIdentificationProblem(ElementwiseProblem):

    def __init__(self, modo="heating"):
        self.modo = modo
        super().__init__(
            n_var=3,
            n_obj=1,
            n_ieq_constr=0,
            xl=np.array([0.001, 0.010, 0.001]),  # Límites inferiores
            xu=np.array([10.00, 10.000, 5.000])   # Límites superiores
        )

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            Tw_sim = simular_curva(x, modo=self.modo)
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
problem = PeltierIdentificationProblem(modo=MODO_OPERACION)

# Algoritmo Genético
algorithm = GA(pop_size=30) 
termination = get_termination("n_gen", 50)

print(f"Optimizando parámetros con pymoo (Modo: {MODO_OPERACION.upper()})...")
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
print(f"   RESULTADOS DE LA OPTIMIZACIÓN - MODO {MODO_OPERACION.upper()}")
print("==================================================")
print(f"R_contacto óptimo : {R_contacto_opt:.4f} K/W")
print(f"R_qR óptimo       : {R_qR_opt:.4f} K/W")
print(f"UA óptimo         : {UA_opt:.4f} W/K")
print(f"RMSE mínimo       : {res.F[0]:.4f} °C")
print("==================================================")


# ==============================================================================
# 5. ANÁLISIS DE VARIABLES RESPECTO A SUS LÍMITES Y GRÁFICOS
# ==============================================================================
hist = res.history
hist_X = []

for algo in hist:
    opt = algo.opt[0]
    hist_X.append(opt.X)

hist_X = np.array(hist_X)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

# Gráfico 1: Ajuste de Temperatura
Tw_sim_opt = simular_curva(res.X, modo=MODO_OPERACION)
axs[0].plot(t_exp / 60, Tw_exp, 'ro', label='Datos Experimentales ($T_w$)')
axs[0].plot(t_exp / 60, Tw_sim_opt, 'b-', label=f'Simulado ({MODO_OPERACION})')
axs[0].set_xlabel('Tiempo [min]')
axs[0].set_ylabel('Temperatura [°C]')
axs[0].set_title(f'Ajuste de Temperatura del Agua (Modo: {MODO_OPERACION})')
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

# ==============================================================================
# 6. FUNCIÓN DE SIMULACIÓN PARA ANÁLISIS DE SENSIBILIDAD
# ==============================================================================
def simular_caso(
    T_amb_custom=None,
    Tw_0=None,
    m_w_custom=None,
    UA_custom=None,
    R_contacto_custom=None,
    R_qR_custom=None,
    I_func_custom=None,
    modo=MODO_OPERACION,
):
    """Simula la respuesta térmica modificando variables de entrada u operativas,

    manteniendo los parámetros óptimos identificados para el resto.
    """
    # Asignación de valores personalizados o valores por defecto
    Tamb_val = T_amb_custom if T_amb_custom is not None else T_amb
    Tw0_val = Tw_0 if Tw_0 is not None else Tw_exp[0]
    mw_val = m_w_custom if m_w_custom is not None else m_w
    UA_val = UA_custom if UA_custom is not None else UA_opt
    R_cont_val = (
        R_contacto_custom if R_contacto_custom is not None else R_contacto_opt
    )
    R_qR_val = R_qR_custom if R_qR_custom is not None else R_qR_opt
    Ifunc_val = I_func_custom if I_func_custom is not None else I_func

    # Recalculado de la inercia térmica según la masa
    if modo == "heating":
        C_sys = mw_val * cp_w + m_acero * cp_acero
    else:
        C_sys = (
            mw_val * cp_bebida
            + m_lata * cp_lata
            + m_bano * cp_agua
            + m_acero * cp_acero
        )

    # Sistema dinámico
    def eduo_sens(t, y):
        Tw = y[0]
        I_t = float(Ifunc_val(t))
        T0 = 273.15

        if modo == "heating":
            A11 = (1.0 / (N * R_qR_val)) + alpha * I_t + (1.0 / Rq)
            A12 = -1.0 / Rq
            B1 = (Tamb_val / (N * R_qR_val)) - alpha * T0 * I_t + 0.5 * (
                I_t**2
            ) * R0

            A21 = -1.0 / Rq
            A22 = (1.0 / (N * R_cont_val)) - alpha * I_t + (1.0 / Rq)
            B2 = (Tw / (N * R_cont_val)) + alpha * T0 * I_t + 0.5 * (
                I_t**2
            ) * R0

            A = np.array([[A11, A12], [A21, A22]])
            B = np.array([B1, B2])

            try:
                Tc, Th = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                return [0.0]

            Qh = (Th - Tw) / (N * R_cont_val)
            dTw_dt = (N * Qh - UA_val * (Tw - Tamb_val)) / C_sys
        else:
            A11 = (1.0 / (N * R_cont_val)) + alpha * I_t + (1.0 / Rq)
            A12 = -1.0 / Rq
            B1 = (Tw / (N * R_cont_val)) - alpha * T0 * I_t + 0.5 * (
                I_t**2
            ) * R0

            A21 = -1.0 / Rq
            A22 = (1.0 / (N * R_qR_val)) - alpha * I_t + (1.0 / Rq)
            B2 = (Tamb_val / (N * R_qR_val)) + alpha * T0 * I_t + 0.5 * (
                I_t**2
            ) * R0

            A = np.array([[A11, A12], [A21, A22]])
            B = np.array([B1, B2])

            try:
                Tc, Th = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                return [0.0]

            Qc = (Tw - Tc) / (N * R_cont_val)
            dTw_dt = (-N * Qc - UA_val * (Tw - Tamb_val)) / C_sys

        return [dTw_dt]

    sol = solve_ivp(
        eduo_sens,
        [t_exp[0], t_exp[-1]],
        [Tw0_val],
        t_eval=t_exp,
        method="RK45",
    )

    return sol.t / 60.0, sol.y[0]


# ==============================================================================
# 7. GENERACIÓN DE GRÁFICOS DE SENSIBILIDAD POR TRÍOS (FIGURAS AGRUPADAS)
# ==============================================================================

# ------------------------------------------------------------------------------
# FIGURA 1: Trío de Variables Operativas de Entrada (T_amb, m_w, I)
# ------------------------------------------------------------------------------
fig1, axs1 = plt.subplots(1, 3, figsize=(18, 5))

# Subgráfico 1.1: Temperatura Ambiente
T_amb_valores = [10.0, 21.5, 30.0, 40.0]  # [°C] (21.5 °C es el caso base)
for Tamb in T_amb_valores:
    t_min, Tw_sim = simular_caso(T_amb_custom=Tamb, Tw_0=Tamb)
    if np.isclose(Tamb, T_amb):
        axs1[0].plot(
            t_min,
            Tw_sim,
            "k--",
            linewidth=2,
            label=f"$T_{{amb}} = {Tamb:.1f}\\,^\\circ\\text{{C}}$ (Caso Base)",
        )
    else:
        axs1[0].plot(
            t_min, Tw_sim, label=f"$T_{{amb}} = {Tamb:.0f}\\,^\\circ\\text{{C}}$"
        )

axs1[0].set_title("Efecto de la Temperatura Ambiente ($T_{amb}$)")
axs1[0].set_xlabel("Tiempo [min]")
axs1[0].set_ylabel("Temperatura del Agua [$^\\circ$C]")
axs1[0].grid(True, alpha=0.3)
axs1[0].legend()

# Subgráfico 1.2: Masa de Agua
m_w_valores = [0.150, 0.250, 0.330, 0.500]  # [kg] (0.330 kg es el caso base)
for mw in m_w_valores:
    t_min, Tw_sim = simular_caso(m_w_custom=mw)
    if np.isclose(mw, m_w):
        axs1[1].plot(
            t_min,
            Tw_sim,
            "k--",
            linewidth=2,
            label=f"$m_w = {int(mw*1000)}\\,\\text{{g}}$ (Caso Base)",
        )
    else:
        axs1[1].plot(
            t_min, Tw_sim, label=f"$m_w = {int(mw*1000)}\\,\\text{{g}}$"
        )

axs1[1].set_title("Efecto de la Masa de Agua ($m_w$)")
axs1[1].set_xlabel("Tiempo [min]")
axs1[1].set_ylabel("Temperatura del Agua [$^\\circ$C]")
axs1[1].grid(True, alpha=0.3)
axs1[1].legend()

# Subgráfico 1.3: Perfil de Corriente
I_constante_low = lambda t: 2.0
I_constante_mid = lambda t: 2.5
I_constante_high = lambda t: 3.0

t_min, Tw_var = simular_caso(I_func_custom=I_func)
_, Tw_c20 = simular_caso(I_func_custom=I_constante_low)
_, Tw_c25 = simular_caso(I_func_custom=I_constante_mid)
_, Tw_c30 = simular_caso(I_func_custom=I_constante_high)

axs1[2].plot(
    t_min,
    Tw_var,
    "k--",
    linewidth=2,
    label="Corriente Experimental Variable (Caso Base)",
)
axs1[2].plot(t_min, Tw_c20, label="I constante = 2.0 A")
axs1[2].plot(t_min, Tw_c25, label="I constante = 2.5 A")
axs1[2].plot(t_min, Tw_c30, label="I constante = 3.0 A")

axs1[2].set_title("Efecto de la Corriente Eléctrica ($I$)")
axs1[2].set_xlabel("Tiempo [min]")
axs1[2].set_ylabel("Temperatura del Agua [$^\\circ$C]")
axs1[2].grid(True, alpha=0.3)
axs1[2].legend()

plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# FIGURA 2: Trío de Parámetros del Modelo Térmico (UA, R_contacto, R_qR)
# ------------------------------------------------------------------------------
fig2, axs2 = plt.subplots(1, 3, figsize=(18, 5))

# Subgráfico 2.1: Aislamiento Térmico (UA)
UA_valores = [UA_opt, 1.0, 3.0, 5.0]  # [W/K] (UA_opt es el valor óptimo)
for ua_val in UA_valores:
    t_min, Tw_sim = simular_caso(UA_custom=ua_val)
    if np.isclose(ua_val, UA_opt):
        axs2[0].plot(
            t_min,
            Tw_sim,
            "k--",
            linewidth=2,
            label=f"UA = {ua_val:.4f} W/K (Óptimo)",
        )
    else:
        axs2[0].plot(t_min, Tw_sim, label=f"UA = {ua_val:.1f} W/K")

axs2[0].set_title("Efecto del Aislamiento Térmico ($UA$)")
axs2[0].set_xlabel("Tiempo [min]")
axs2[0].set_ylabel("Temperatura del Agua [$^\\circ$C]")
axs2[0].grid(True, alpha=0.3)
axs2[0].legend()

# Subgráfico 2.2: Resistencia de Contacto Térmico (R_contacto)
factores = [0.5, 1.0, 1.5, 2.0]
for f in factores:
    r_cont_val = R_contacto_opt * f
    t_min, Tw_sim = simular_caso(R_contacto_custom=r_cont_val)
    if f == 1.0:
        axs2[1].plot(
            t_min,
            Tw_sim,
            "k--",
            linewidth=2,
            label=f"$R_{{cont}} = {r_cont_val:.3f}\\,\\text{{K/W}}$ (100% Óptimo)",
        )
    else:
        axs2[1].plot(
            t_min,
            Tw_sim,
            label=f"$R_{{cont}} = {r_cont_val:.3f}\\,\\text{{K/W}}$ ({f*100:.0f}%)",
        )

axs2[1].set_title("Efecto de la Res. de Contacto ($R_{contacto}$)")
axs2[1].set_xlabel("Tiempo [min]")
axs2[1].set_ylabel("Temperatura del Agua [$^\\circ$C]")
axs2[1].grid(True, alpha=0.3)
axs2[1].legend()

# Subgráfico 2.3: Resistencia Térmica Interna (R_qR)
for f in factores:
    r_qr_val = R_qR_opt * f
    t_min, Tw_sim = simular_caso(R_qR_custom=r_qr_val)
    if f == 1.0:
        axs2[2].plot(
            t_min,
            Tw_sim,
            "k--",
            linewidth=2,
            label=f"$R_{{qR}} = {r_qr_val:.3f}\\,\\text{{K/W}}$ (100% Óptimo)",
        )
    else:
        axs2[2].plot(
            t_min,
            Tw_sim,
            label=f"$R_{{qR}} = {r_qr_val:.3f}\\,\\text{{K/W}}$ ({f*100:.0f}%)",
        )

axs2[2].set_title("Efecto de la Res. Interna ($R_{qR}$)")
axs2[2].set_xlabel("Tiempo [min]")
axs2[2].set_ylabel("Temperatura del Agua [$^\\circ$C]")
axs2[2].grid(True, alpha=0.3)
axs2[2].legend()

plt.tight_layout()
plt.show()

# ==============================================================================
# 8. EXPORTACIÓN DE RESULTADOS Y FLUJO DE CALOR ÚTIL A EXCEL
# ==============================================================================
import pandas as pd


def exportar_flujo_calor_excel(
    nombre_archivo="Flujo_de_Calor_Util.xlsx", usar_datos_simulados=True
):
    """Calcula el flujo de calor útil (Q_flujo) en cada instante de tiempo

    y exporta la tabla completa a un archivo Excel (.xlsx).
    """
    # 1. Seleccionar perfil de temperatura (simulado con parámetros óptimos o experimental)
    if usar_datos_simulados:
        Tw_eval = simular_curva(
            [R_contacto_opt, R_qR_opt, UA_opt], modo=MODO_OPERACION
        )
    else:
        Tw_eval = Tw_exp

    Tc_list, Th_list, Q_flujo_list, Q_perdidas_list = [], [], [], []

    # 2. Calcular variables térmicas en cada punto del tiempo
    for t, I, Tw in zip(t_exp, I_exp, Tw_eval):
        Tc, Th, Q_flujo = resolver_superficies(
            Tw, I, R_contacto_opt, R_qR_opt, modo=MODO_OPERACION
        )
        Q_perdidas = UA_opt * (Tw - T_amb)

        Tc_list.append(Tc)
        Th_list.append(Th)
        Q_flujo_list.append(Q_flujo)
        Q_perdidas_list.append(Q_perdidas)

    Q_flujo_arr = np.array(Q_flujo_list)
    Q_perdidas_arr = np.array(Q_perdidas_list)

    # 3. Estructurar el DataFrame
    df_resultados = pd.DataFrame(
        {
            "Tiempo [min]": t_exp / 60.0,
            "Tiempo [s]": t_exp,
            "Corriente I [A]": I_exp,
            "T_agua [°C]": Tw_eval,
            "Tc Superficie Fría [°C]": Tc_list,
            "Th Superficie Caliente [°C]": Th_list,
            "Flujo de Calor Útil Q_flujo [W]": Q_flujo_arr,
            "Pérdidas Térmicas Q_perdidas [W]": Q_perdidas_arr,
            "Flujo Neto de Calor [W]": Q_flujo_arr - Q_perdidas_arr
            if MODO_OPERACION == "heating"
            else -Q_flujo_arr - Q_perdidas_arr,
        }
    )

    # 4. Guardar archivo Excel
    df_resultados.to_excel(nombre_archivo, index=False)
    print(f"Archivo Excel exportado exitosamente: {nombre_archivo}")
    return df_resultados


# Ejecución de la función
df_excel = exportar_flujo_calor_excel("Flujo_de_Calor_Util.xlsx")