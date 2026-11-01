import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ==============================================================================
# 1. PARÁMETROS FÍSICOS Y PARÁMETROS OPTIMIZADOS (AG)
# ==============================================================================
# Propiedades del módulo Peltier (TEC1-12706 - Dymytrov 2017)
ALPHA = 0.0530    # Coeficiente Seebeck efectivo [V/K]
R0 = 2.1000       # Resistencia eléctrica interna [Ohm]
Rq = 1.9500       # Resistencia térmica interna [K/W]
N = 2             # Celdas en serie

# Propiedades de materiales
CP_AGUA = 4186.0     # [J/kg*K] Agua
CP_BEBIDA = 3900.0   # [J/kg*K] Refresco/Jugo
CP_ACERO = 500.0     # [J/kg*K] Recipiente de acero inoxidable
CP_ALUMINIO = 900.0  # [J/kg*K] Lata de aluminio

M_ACERO = 0.150      # Masa del recipiente de acero [kg] (150 g)
M_LATA = 0.015       # Masa de la lata de aluminio [kg] (15 g)

# Parámetros óptimos identificados
R_CONTACTO_OPT = 0.0116  # [K/W]
R_QR_OPT = 0.0182        # [K/W]
UA_OPT = 0.7413          # [W/K]

# Perfil de corriente experimental base
t_exp = np.linspace(0, 2100, 36)  # [s]
I_exp = np.array([
    2.91, 2.53, 2.46, 2.43, 2.38, 2.37, 2.30, 2.26, 2.24, 2.23,
    2.19, 2.13, 2.13, 2.12, 2.11, 2.10, 2.10, 2.10, 2.14, 2.11,
    2.11, 2.13, 2.16, 2.19, 2.23, 2.26, 2.27, 2.31, 2.29, 2.30,
    2.30, 2.31, 2.31, 2.31, 2.31, 2.31
])
I_func_base = interp1d(t_exp, I_exp, bounds_error=False, fill_value="extrapolate")


# ==============================================================================
# 2. MOTOR FÍSICO DE SUPERFICIES Y BALANCE TÉRMICO
# ==============================================================================

# --- MODO 1: CALENTAR (Solo Agua + Recipiente) ---
def resolver_superficies_heating(Tw, I, T_amb, R_contacto, R_qR):
    T0 = 273.15
    A11 = (1.0 / (N * R_qR)) + ALPHA * I + (1.0 / Rq)
    A12 = -1.0 / Rq
    B1  = (T_amb / (N * R_qR)) - ALPHA * T0 * I + 0.5 * (I**2) * R0

    A21 = -1.0 / Rq
    A22 = (1.0 / (N * R_contacto)) - ALPHA * I + (1.0 / Rq)
    B2  = (Tw / (N * R_contacto)) + ALPHA * T0 * I + 0.5 * (I**2) * R0

    A = np.array([[A11, A12], [A21, A22]])
    B = np.array([B1, B2])

    try:
        Tc, Th = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return T_amb, Tw, 0.0

    Qh = (Th - Tw) / (N * R_contacto)
    return Tc, Th, N * Qh

def eduo_heating(t, y, I_func, T_amb, m_agua_calentar, R_contacto, R_qR, UA):
    Tw = y[0]
    I_t = float(I_func(t))
    _, _, Q_util = resolver_superficies_heating(Tw, I_t, T_amb, R_contacto, R_qR)
    Q_perdidas = UA * (Tw - T_amb)
    
    # Inercia térmica: Agua para maruchan + Recipiente
    C_sistema = m_agua_calentar * CP_AGUA + M_ACERO * CP_ACERO
    return [(Q_util - Q_perdidas) / C_sistema]


# --- MODO 2: ENFRIAR (Bebida + Lata + Agua de Acople + Recipiente) ---
def resolver_superficies_cooling(Tw, I, T_amb, R_contacto, R_qR):
    T0 = 273.15
    A11 = (1.0 / (N * R_contacto)) + ALPHA * I + (1.0 / Rq)
    A12 = -1.0 / Rq
    B1  = (Tw / (N * R_contacto)) - ALPHA * T0 * I + 0.5 * (I**2) * R0

    A21 = -1.0 / Rq
    A22 = (1.0 / (N * R_qR)) - ALPHA * I + (1.0 / Rq)
    B2  = (T_amb / (N * R_qR)) + ALPHA * T0 * I + 0.5 * (I**2) * R0

    A = np.array([[A11, A12], [A21, A22]])
    B = np.array([B1, B2])

    try:
        Tc, Th = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return T_amb, Tw, 0.0

    Qc = (Tw - Tc) / (N * R_contacto)
    return Tc, Th, N * Qc

def eduo_cooling(t, y, I_func, T_amb, m_bebida, m_agua_bano, R_contacto, R_qR, UA):
    Tw = y[0]
    I_t = float(I_func(t))
    _, _, Q_extraida = resolver_superficies_cooling(Tw, I_t, T_amb, R_contacto, R_qR)
    Q_ganancias_amb = UA * (Tw - T_amb)
    
    # Inercia térmica completa: Bebida + Lata + Agua de nexo + Recipiente
    C_sistema = (
        m_bebida * CP_BEBIDA + 
        M_LATA * CP_ALUMINIO + 
        m_agua_bano * CP_AGUA + 
        M_ACERO * CP_ACERO
    )
    return [(-Q_extraida - Q_ganancias_amb) / C_sistema]


# ==============================================================================
# 3. FUNCIONES DE SIMULACIÓN Y PREDICCIÓN
# ==============================================================================
def simular_modo_calentar(t_max_min=35, T_amb=20.0, Tw_0=20.0, m_agua=0.330, I_input=I_func_base):
    t_eval = np.linspace(0, t_max_min * 60, 200)
    I_func = I_input if callable(I_input) else (lambda t: float(I_input))
    
    sol = solve_ivp(
        eduo_heating, [0, t_max_min * 60], [Tw_0],
        args=(I_func, T_amb, m_agua, R_CONTACTO_OPT, R_QR_OPT, UA_OPT),
        t_eval=t_eval, method='RK45'
    )
    return t_eval / 60.0, sol.y[0]

def simular_modo_enfriar(t_max_min=45, T_amb=20.0, Tw_0=20.0, m_bebida=0.355, m_agua_bano=0.100, I_input=2.5):
    t_eval = np.linspace(0, t_max_min * 60, 200)
    I_func = I_input if callable(I_input) else (lambda t: float(I_input))
    
    sol = solve_ivp(
        eduo_cooling, [0, t_max_min * 60], [Tw_0],
        args=(I_func, T_amb, m_bebida, m_agua_bano, R_CONTACTO_OPT, R_QR_OPT, UA_OPT),
        t_eval=t_eval, method='RK45'
    )
    return t_eval / 60.0, sol.y[0]


# ==============================================================================
# 4. GRÁFICOS DE PREDICCIÓN OPERATIVA
# ==============================================================================
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# --- PANEL 1: PREDICCIÓN MODO CALENTAR (AGUA PARA MARUCHAN) ---
t_h, Tw_h1 = simular_modo_calentar(m_agua=0.250, I_input=2.5)  # 250 g
_, Tw_h2 = simular_modo_calentar(m_agua=0.330, I_input=2.5)    # 330 g

axs[0].plot(t_h, Tw_h1, 'r-', linewidth=2, label='Calentar 250 g de agua')
axs[0].plot(t_h, Tw_h2, 'r--', linewidth=2, label='Calentar 330 g de agua')
axs[0].axhline(y=65.0, color='k', linestyle=':', label='Temp. servido maruchan (65 °C)')
axs[0].set_title('Modo Calentar: Agua para Maruchan ($I = 2.5\\,\\text{A}$)')
axs[0].set_xlabel('Tiempo [min]')
axs[0].set_ylabel('Temperatura del Agua [$^\\circ$C]')
axs[0].grid(True, alpha=0.3)
axs[0].legend()

# --- PANEL 2: PREDICCIÓN MODO ENFRIAR (LATA 355ml + BAÑO DE AGUA) ---
t_c, Tw_c1 = simular_modo_enfriar(m_bebida=0.355, m_agua_bano=0.100, I_input=2.5)  # 100 g de baño
_, Tw_c2 = simular_modo_enfriar(m_bebida=0.355, m_agua_bano=0.200, I_input=2.5)    # 200 g de baño

axs[1].plot(t_c, Tw_c1, 'b-', linewidth=2, label='Lata 355 ml + 100 g agua de baño')
axs[1].plot(t_c, Tw_c2, 'b--', linewidth=2, label='Lata 355 ml + 200 g agua de baño')
axs[1].axhline(y=8.0, color='k', linestyle=':', label='Temp. bebida fría (8 °C)')
axs[1].set_title('Modo Enfriar: Lata + Baño de Agua ($I = 2.5\\,\\text{A}$)')
axs[1].set_xlabel('Tiempo [min]')
axs[1].set_ylabel('Temperatura de la Bebida [$^\\circ$C]')
axs[1].grid(True, alpha=0.3)
axs[1].legend()

plt.tight_layout()
plt.show()