import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ==============================================================================
# 1. PARÁMETROS OPTIMIZADOS EN MODO CALENTAMIENTO (INGRESAR AQUÍ TUS RESULTADOS)
# ==============================================================================
# Reemplaza estos valores por los obtenidos en la optimización del modo calentamiento
R_contacto_opt = 8.0375  # [K/W] Resistencia de contacto térmico
R_qR_opt = 1.9562  # [K/W] Resistencia térmica interna
UA_opt = 0.0010  # [W/K] Coeficiente de pérdidas térmicas


# ==============================================================================
# 2. DATOS EXPERIMENTALES EN MODO ENFRIAMIENTO (COOLING)
# ==============================================================================
# Tiempo experimental (0 a 40 minutos en segundos, 41 puntos)
t_exp_cooling = np.linspace(0, 2400, 41)

# Corriente medida / corregida [A]
I_exp_cooling = np.array([
    1.830000, 1.871667, 1.913333, 1.955000, 1.996667, 2.038333, 2.080000, 2.121667,
    2.163333, 2.205000, 2.246667, 2.288333, 2.330000, 2.371667, 2.413333, 2.455000,
    2.496667, 2.538333, 2.580000, 2.621667, 2.663333, 2.705000, 2.746667, 2.788333,
    2.830000, 2.871667, 2.913333, 2.955000, 2.996667, 3.038333, 3.080000, 3.121667,
    3.163333, 3.205000, 3.246667, 3.288333, 3.330000, 3.371667, 3.413333, 3.455000,
    3.496667
])

# Temperatura experimental del fluido/agua [°C]
Tw_exp_cooling = np.array([
    22.8, 16.8, 16.7, 16.6, 16.6, 16.3, 16.1, 15.8, 15.5, 15.2,
    15.1, 14.8, 14.4, 14.2, 13.9, 13.7, 13.4, 13.2, 13.0, 12.8,
    12.6, 12.3, 12.1, 11.9, 11.6, 11.4, 11.2, 11.0, 10.8, 10.6,
    10.4, 10.2, 10.0, 10.3,  9.8,  9.3,  9.2,  9.2,  9.0,  8.8,
    8.7
])

T_amb = 21.5  # Temperatura ambiente [°C]

# Propiedades térmicas del sistema en enfriamiento
m_bebida, cp_bebida = 0.355, 3900.0
m_lata, cp_lata = 0.015, 900.0
m_bano, cp_agua = 0.300, 4186.0
m_acero, cp_acero = 0.207, 500.0
C_sistema_cooling = (
    m_bebida * cp_bebida
    + m_lata * cp_lata
    + m_bano * cp_agua
    + m_acero * cp_acero
)

alpha = 0.0530
R0 = 2.1000
Rq = 1.9500
N = 2

I_func_cooling = interp1d(
    t_exp_cooling, I_exp_cooling, bounds_error=False, fill_value="extrapolate"
)


# ==============================================================================
# 3. RESOLUCIÓN DE SUPERFICIES Y ENFRIAMIENTO CORREGIDO
# ==============================================================================
def resolver_superficies_cooling(Tw, I, R_contacto, R_qR):
    """Calcula Tc (superficie fría en contacto con agua) y Th (caliente en contacto con disipador)."""
    T0 = 273.15

    # Ecuación de conservación en superficie fría (Tc)
    A11 = (1.0 / (N * R_contacto)) + alpha * I + (1.0 / Rq)
    A12 = -1.0 / Rq
    B1 = (Tw / (N * R_contacto)) - alpha * T0 * I + 0.5 * (I**2) * R0

    # Ecuación de conservación en superficie caliente (Th)
    A21 = -1.0 / Rq
    A22 = (1.0 / (N * R_qR)) - alpha * I + (1.0 / Rq)
    B2 = (T_amb / (N * R_qR)) + alpha * T0 * I + 0.5 * (I**2) * R0

    A = np.array([[A11, A12], [A21, A22]])
    B = np.array([B1, B2])

    try:
        Tc, Th = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return T_amb, Tw, 0.0

    # Calor extraído del agua hacia la cara fría
    Qc = (Tw - Tc) / (N * R_contacto)
    return Tc, Th, N * Qc


def eduo_cooling(t, y, R_contacto, R_qR, UA):
    Tw = y[0]
    I_t = float(I_func_cooling(t))
    Tc, Th, Q_extraido = resolver_superficies_cooling(
        Tw, I_t, R_contacto, R_qR
    )

    # Pérdidas al ambiente (positivas si el agua está más caliente que el ambiente)
    Q_perdidas = UA * (Tw - T_amb)

    # El calor extraído SIEMPRE reduce la temperatura del fluido
    dTw_dt = (-Q_extraido - Q_perdidas) / C_sistema_cooling
    return [dTw_dt]


# Simulación numérica
sol = solve_ivp(
    eduo_cooling,
    [t_exp_cooling[0], t_exp_cooling[-1]],
    [Tw_exp_cooling[0]],
    args=(R_contacto_opt, R_qR_opt, UA_opt),
    t_eval=t_exp_cooling,
    method="RK45",
)

Tw_sim_cooling = sol.y[0]

# ==============================================================================
# 4. GRÁFICO DE COMPARACIÓN
# ==============================================================================
plt.figure(figsize=(9, 5))
plt.plot(
    t_exp_cooling / 60.0,
    Tw_exp_cooling,
    "ro",
    label="Datos Experimentales ($T_w$ Enfriamiento)",
)
plt.plot(
    t_exp_cooling / 60.0,
    Tw_sim_cooling,
    "b-",
    linewidth=2,
    label="Simulación en Enfriamiento",
)
plt.title("Validación en Modo Enfriamiento")
plt.xlabel("Tiempo [min]")
plt.ylabel("Temperatura del Agua [°C]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()