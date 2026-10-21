"""
Caso VisionAR - Predicción de la demanda de gafas de realidad aumentada en España
Simulación de Sistemas Logísticos (UAX) - Material de apoyo a la presentación

Estructura del script
---------------------
  PARTE 0. Supuestos y fuentes (todos los parámetros en un único sitio, con fuente y fecha)
  PARTE 1. Tamaño de mercado: TAM / SAM / SOM, reparto B2B-B2C, unidades, facturación y margen
  PARTE 2. Modelos predictivos de demanda 2026-2028 (exponencial y logístico) y escenarios
  PARTE 3. Gráficos para la presentación

Ejecución:  python visionar_modelo.py
Salidas:    resultados/*.csv  (tablas)   y   graficos/*.png  (figuras a 300 ppp)

Convenciones
  - Todos los precios del enunciado incluyen IVA (21 %). Todo se calcula SIN IVA.
  - t = 0 corresponde a 2026 (lanzamiento). Horizonte del caso: 2026-2028.
  - Escenarios: pesimista / base / optimista.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

BASE_DIR = Path(__file__).resolve().parent
DIR_RES = BASE_DIR / "resultados"
DIR_GRAF = BASE_DIR / "graficos"
DIR_RES.mkdir(exist_ok=True)
DIR_GRAF.mkdir(exist_ok=True)

ANIOS = [2026, 2027, 2028]
ESCENARIOS = ["pesimista", "base", "optimista"]


# =============================================================================
# PARTE 0. SUPUESTOS Y FUENTES
# =============================================================================
# Cada parámetro lleva su fuente y fecha. Los marcados como "Supuesto" son
# hipótesis propias del equipo, razonadas en el comentario; son los primeros
# candidatos a revisar.

# --- Precio y coste (enunciado) ----------------------------------------------
IVA = 0.21
PVP_CON_IVA = 900.0                       # Enunciado, tabla "Datos iniciales"
COSTE_UNITARIO = 400.0                    # Enunciado (coste, sin IVA)
PRECIO_SIN_IVA = PVP_CON_IVA / (1 + IVA)  # 743,80 €
UNIDADES_POR_EMPRESA = 10                 # Enunciado: gasto medio 9.000 € = 10 uds

# --- Demografía B2C ------------------------------------------------------------
POBLACION_ADULTA_SMARTPHONE = 40.05e6     # Enunciado (ONTSI: 97,2 % de 41,2 M adultos)
PENETRACION_GAFAS_AR = 0.03               # Enunciado (Statista): usuarios potenciales de gafas AR
# INE, Estadística Continua de Población / Padrón a 1-1-2025 (nota de prensa dic-2025):
# el 40,9 % de la población reside en municipios de más de 100.000 habitantes.
CUOTA_POB_URBANA_100K = 0.409

# --- Demografía B2B -------------------------------------------------------------
EMPRESAS_TECH_MARKETING = 120_000         # Enunciado (Red.es): tecnológicas y marketing
EMPRESAS_SEGURIDAD = 12_362               # Enunciado (INE): seguridad privada
# INE, DIRCE a 1-1-2025 (nota de prensa 11-12-2025): 3.310.824 empresas activas;
# 85.467 con 10-19 asalariados y 76.005 con 20 o más.
DIRCE_TOTAL = 3_310_824
DIRCE_10_O_MAS = 85_467 + 76_005
CUOTA_EMPRESAS_10_O_MAS = DIRCE_10_O_MAS / DIRCE_TOTAL          # 4,88 %
# INE, DIRCE 2025: Cataluña 18,4 %, Andalucía 16,3 %, Madrid 15,9 % de las empresas.
CUOTA_EMPRESAS_CCAA_COBERTURA = 0.184 + 0.163 + 0.159           # 50,6 %

# --- Mercado de gafas inteligentes (benchmark top-down) ----------------------------
# Envíos mundiales (millones de uds):
#   2023 y 2024: Counterpoint (feb-2025): 2024 > 2 M uds, +210 % interanual.
#   2025: IDC (crecimiento 2025 del +211,2 %). Omdia da 8,7 M de "AI glasses" con otra definición.
#   2026: IDC 13,6 M; 2030: IDC 27,3 M (TACC 2026-2030 del 18,9 %). 2027-2029 interpolados con esa TACC.
ENVIOS_MUNDIALES = {
    2023: 2.0 / 3.10,
    2024: 2.0,
    2025: 2.0 * 3.112,
    2026: 13.6,
    2027: 13.6 * 1.189,
    2028: 13.6 * 1.189 ** 2,
    2029: 13.6 * 1.189 ** 3,
    2030: 27.3,
}
CUOTA_EUROPA_OCCIDENTAL = 0.30            # Counterpoint, H2-2025: Europa Occidental = 30 % de los envíos
# Eurostat, población a 1-1-2025: España 49,08 M. Europa Occidental (UE sin países
# del Este + Reino Unido, Suiza y Noruega) ≈ 434 M. Cálculo propio a partir de Eurostat.
CUOTA_ESPANA_EN_EUR_OCC = 49.08 / 434.0   # ≈ 11,3 %

# --- Financiación y presupuesto (SOM) ----------------------------------------------
# El Referente / prensa económica (oct-2025): en los 9 primeros meses de 2025 la
# Serie A fue la ronda más frecuente en España (86 operaciones) y el importe medio
# por operación fue de 10,6 M€. Supuesto: VisionAR cierra una Serie A de 10 M€.
RONDA_SERIE_A = 10.0e6
# Supuesto: presupuesto de marketing B2C anual (12 %, 16 % y 20 % de la ronda; a partir
# de 2027 se financia en parte con el margen bruto generado).
PRESUPUESTO_MKT_B2C = {2026: 1.2e6, 2027: 1.6e6, 2028: 2.0e6}
# Coste de adquisición de cliente (CAC) B2C, en €. Eightx (2026): el CAC "blended" de
# las marcas DTC europeas está entre 65 y 110 €, y en electrónica de consumo entre
# 70 y más de 220 €; el CAC solo de pago es unas 2,4 veces el blended.
# Una marca nueva sin tráfico orgánico paga el CAC de pago: 145 € (punto medio) x 2,4 ≈ 350 €.
CAC_B2C = {"pesimista": 500.0, "base": 350.0, "optimista": 220.0}

# --- Capacidad comercial B2B (SOM) ------------------------------------------------
# Supuesto: comerciales en plantilla a final de cada año (uno nuevo por año).
COMERCIALES = {2026: 3, 2027: 4, 2028: 5}
# Un comercial nuevo rinde la mitad el primer año (rampa de 3-6 meses y ciclo de
# venta de unos 90 días en operaciones de menos de 25.000 $; abmatic/Emulent 2025-26).
FACTOR_RAMPA_PRIMER_ANIO = 0.5
# Operaciones cerradas por comercial y año. Dos vías:
#  a) Embudo: 15 oportunidades/mes x 12 x 21 % de tasa de cierre media B2B (Kondo 2025) ≈ 38
#  b) Cuota: 800.000 $ de cuota mediana x 50 % de cumplimiento (43-58 %, Hyperbound 2025)
#     ≈ 370.000 € / 7.438 € por operación ≈ 50
OPERACIONES_POR_COMERCIAL = {"pesimista": 30, "base": 40, "optimista": 50}
# Coste anual cargado por comercial. RepVue (abr-2025): salario base mediano de un
# account manager en España de 51.802 € (51.802 x 1,31 de Seguridad Social ≈ 68.000 €).
COSTE_COMERCIAL_ANUAL = 70_000.0

# --- Parámetros de difusión (modelo logístico) -------------------------------------
# Vía 1. Metaanálisis del modelo de Bass (Sultan, Farley y Lehmann, 1990): p = 0,03 y q = 0,38.
BASS_P, BASS_Q = 0.03, 0.38
# Inicio de la categoría: Ray-Ban Meta de 2ª generación (sept-2023), que dispara el mercado.
ANIO_INICIO_CATEGORIA = 2023
# Supuesto: cuota de VisionAR en el SAM a largo plazo. Meta tiene el 73 % de la categoría
# (Counterpoint, H1-2025) y el resto se reparte entre muchos competidores; en el segmento
# con pantalla AR los dos líderes (Xreal y RayNeo) suman más del 60 %. Para un retador
# premium de nicho se supone un 10 %.
CUOTA_LARGO_PLAZO_SAM = 0.10

# --- Crecimiento (modelo exponencial) ----------------------------------------------
G_ENUNCIADO = 0.15        # Enunciado (Statista / PwC): crecimiento del mercado AR
G_IDC = 0.189             # IDC: TACC 2026-2030 de los envíos de gafas inteligentes


# =============================================================================
# PARTE 1. TAM / SAM / SOM
# =============================================================================

def calcular_tam_sam():
    """TAM y SAM por segmento, en unidades y en € sin IVA."""
    # TAM: todos los clientes potenciales compran el producto.
    tam_b2c_uds = POBLACION_ADULTA_SMARTPHONE * PENETRACION_GAFAS_AR
    tam_b2b_emp = EMPRESAS_TECH_MARKETING + EMPRESAS_SEGURIDAD
    tam_b2b_uds = tam_b2b_emp * UNIDADES_POR_EMPRESA

    # SAM B2C: VisionAR distribuye con demostración presencial (retail en grandes
    # ciudades + venta online). Con un producto de 900 € que hay que probar, el
    # comprador realista vive donde hay puntos de venta y cobertura 5G: municipios
    # de más de 100.000 habitantes.
    sam_b2c_uds = tam_b2c_uds * CUOTA_POB_URBANA_100K

    # SAM B2B: 1) solo empresas con 10 o más asalariados, porque el pedido tipo es de
    # 10 unidades; 2) solo las comunidades cubiertas por el equipo comercial directo
    # (Madrid, Cataluña y Andalucía; esta última por su peso en el turismo).
    sam_b2b_emp = tam_b2b_emp * CUOTA_EMPRESAS_10_O_MAS * CUOTA_EMPRESAS_CCAA_COBERTURA
    sam_b2b_uds = sam_b2b_emp * UNIDADES_POR_EMPRESA

    filas = [
        ("TAM", "B2C", tam_b2c_uds, np.nan),
        ("TAM", "B2B", tam_b2b_uds, tam_b2b_emp),
        ("SAM", "B2C", sam_b2c_uds, np.nan),
        ("SAM", "B2B", sam_b2b_uds, sam_b2b_emp),
    ]
    df = pd.DataFrame(filas, columns=["nivel", "segmento", "unidades", "empresas"])
    df["facturacion_eur"] = df["unidades"] * PRECIO_SIN_IVA
    return df


def calcular_som(escenario="base"):
    """SOM 2026-2028 bottom-up: presupuesto (B2C) + capacidad comercial (B2B)."""
    filas = []
    comerciales_previos = 0
    for anio in ANIOS:
        # B2C: unidades = presupuesto de marketing / CAC (una unidad por cliente)
        uds_b2c = PRESUPUESTO_MKT_B2C[anio] / CAC_B2C[escenario]

        # B2B: comerciales equivalentes a jornada completa (los nuevos rinden la mitad)
        nuevos = COMERCIALES[anio] - comerciales_previos
        fte = comerciales_previos + nuevos * FACTOR_RAMPA_PRIMER_ANIO
        operaciones = fte * OPERACIONES_POR_COMERCIAL[escenario]
        uds_b2b = operaciones * UNIDADES_POR_EMPRESA
        comerciales_previos = COMERCIALES[anio]

        coste_comercial_b2b = COMERCIALES[anio] * COSTE_COMERCIAL_ANUAL
        filas.append((anio, "B2C", uds_b2c, np.nan, PRESUPUESTO_MKT_B2C[anio]))
        filas.append((anio, "B2B", uds_b2b, operaciones, coste_comercial_b2b))

    df = pd.DataFrame(filas, columns=["anio", "segmento", "unidades", "empresas", "coste_adquisicion_eur"])
    df["escenario"] = escenario
    df["facturacion_eur"] = df["unidades"] * PRECIO_SIN_IVA
    df["margen_bruto_eur"] = df["unidades"] * (PRECIO_SIN_IVA - COSTE_UNITARIO)
    df["contribucion_eur"] = df["margen_bruto_eur"] - df["coste_adquisicion_eur"]
    return df


def contraste_top_down_b2c(som_base):
    """Cuota implícita de VisionAR B2C en el mercado español de gafas inteligentes."""
    filas = []
    for anio in ANIOS:
        mercado_es = ENVIOS_MUNDIALES[anio] * 1e6 * CUOTA_EUROPA_OCCIDENTAL * CUOTA_ESPANA_EN_EUR_OCC
        uds = som_base.query("anio == @anio and segmento == 'B2C'")["unidades"].iloc[0]
        filas.append((anio, mercado_es, uds, uds / mercado_es))
    return pd.DataFrame(filas, columns=["anio", "mercado_gafas_espana_uds", "visionar_b2c_uds", "cuota_implicita"])


def margen_bruto_unitario():
    return (PRECIO_SIN_IVA - COSTE_UNITARIO) / PRECIO_SIN_IVA


# =============================================================================
# PARTE 2. MODELOS PREDICTIVOS DE DEMANDA
# =============================================================================

def logistica(t, K, a, b):
    """Demanda acumulada de la curva S: K / (1 + e^(-a (t - b)))."""
    return K / (1 + np.exp(-a * (t - b)))


def demanda_anual_logistica(K, a, b, anios=ANIOS):
    """Demanda anual = diferencia de la acumulada entre años consecutivos."""
    t = np.array([anio - ANIOS[0] for anio in anios], dtype=float)
    return logistica(t, K, a, b) - logistica(t - 1, K, a, b)


def demanda_exponencial(D0, g, anios=ANIOS):
    t = np.array([anio - ANIOS[0] for anio in anios], dtype=float)
    return D0 * (1 + g) ** t


def estimar_parametros_difusion():
    """Velocidad (a) e inflexión (b) por dos vías independientes.

    Vía 1 (Bass): a ≈ p + q;  momento del pico t* = ln(q/p) / (p + q), contado
                  desde el inicio de la categoría (2023).
    Vía 2 (análogo): ajuste por mínimos cuadrados de una curva logística a los
                  envíos mundiales acumulados de gafas inteligentes (2023-2030).
    b se pasa a la escala del caso (t = 0 en 2026).
    """
    desfase = ANIOS[0] - ANIO_INICIO_CATEGORIA          # 3 años de categoría antes del lanzamiento

    a_bass = BASS_P + BASS_Q
    b_bass = np.log(BASS_Q / BASS_P) / (BASS_P + BASS_Q) - desfase

    anios = np.array(sorted(ENVIOS_MUNDIALES))
    acumulado = np.cumsum([ENVIOS_MUNDIALES[a] for a in anios])
    (K_an, a_an, b_an), _ = curve_fit(logistica, anios - ANIO_INICIO_CATEGORIA, acumulado,
                                      p0=[100, 0.5, 5], maxfev=20000)
    b_an = b_an - desfase

    tabla = pd.DataFrame({
        "via": ["Bass (Sultan et al., 1990)", "Análogo: envíos mundiales (Counterpoint/IDC)"],
        "a": [a_bass, a_an],
        "b_t0_2026": [b_bass, b_an],
    })
    return tabla, K_an


def estimar_parametros_modelos(tam_sam, soms, difusion):
    """Parámetros D0, g, K, a, b por escenario, con su origen trazable."""
    a_vias = difusion["a"].to_numpy()
    b_vias = difusion["b_t0_2026"].to_numpy()
    sam_uds = tam_sam.query("nivel == 'SAM'")["unidades"].sum()

    params = {}
    for esc in ESCENARIOS:
        som = soms[esc].groupby("anio")["unidades"].sum()

        # --- Exponencial: D0 = SOM bottom-up de 2026 -------------------------------
        D0 = som[2026]
        g_implicita = (som[2028] / som[2026]) ** 0.5 - 1   # TACC del plan comercial
        g = {"pesimista": G_ENUNCIADO, "base": G_IDC, "optimista": g_implicita}[esc]

        # --- Logístico: a y b -------------------------------------------------
        if esc == "pesimista":      # difusión más lenta y tardía (Bass)
            a, b = a_vias[0], b_vias[0]
        elif esc == "optimista":    # difusión del análogo real (más rápida)
            a, b = a_vias[1], b_vias[1]
        else:                       # media de las dos vías
            a, b = a_vias.mean(), b_vias.mean()

        # --- Logístico: K por dos vías -------------------------------------------
        #  Vía 1 (top-down): K = SAM (uds) x cuota a largo plazo
        #  Vía 2 (capacidad): K tal que la curva reproduzca el SOM a 3 años del plan comercial
        K_topdown = sam_uds * CUOTA_LARGO_PLAZO_SAM
        fraccion_3a = demanda_anual_logistica(1.0, a, b).sum()
        K_capacidad = som.sum() / fraccion_3a
        K = {"pesimista": min(K_topdown, K_capacidad),
             "base": (K_topdown + K_capacidad) / 2,
             "optimista": max(K_topdown, K_capacidad)}[esc]

        params[esc] = dict(D0=D0, g=g, g_implicita=g_implicita, K=K, K_topdown=K_topdown,
                           K_capacidad=K_capacidad, a=a, b=b)
    return params


def proyectar_demanda(params):
    filas = []
    for esc, p in params.items():
        exp_ = demanda_exponencial(p["D0"], p["g"])
        log_ = demanda_anual_logistica(p["K"], p["a"], p["b"])
        for i, anio in enumerate(ANIOS):
            filas.append((esc, anio, "Exponencial", exp_[i]))
            filas.append((esc, anio, "Logístico", log_[i]))
    df = pd.DataFrame(filas, columns=["escenario", "anio", "modelo", "unidades"])
    df["facturacion_eur"] = df["unidades"] * PRECIO_SIN_IVA
    df["margen_bruto_eur"] = df["unidades"] * (PRECIO_SIN_IVA - COSTE_UNITARIO)
    return df


def plan_produccion(demanda):
    """Previsión de consenso (media de los dos modelos base) y rango de todos los escenarios."""
    base = demanda.query("escenario == 'base'").groupby("anio")["unidades"].mean()
    rango = demanda.groupby("anio")["unidades"].agg(["min", "max"])
    df = pd.DataFrame({"consenso_base": base, "minimo": rango["min"], "maximo": rango["max"]})
    return df.round(0)


# =============================================================================
# PARTE 3. GRÁFICOS
# =============================================================================
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Paleta categórica validada (skill dataviz, modo claro). B2C = azul, B2B = naranja.
C_B2C = "#2a78d6"
C_B2B = "#eb6834"
C_LOG = "#1baf7a"          # modelo logístico
C_EXP = "#52514e"          # modelo exponencial (gris, línea discontinua)
TXT = "#0b0b0b"
TXT2 = "#52514e"
REJILLA = "#e4e3df"
SUPERFICIE = "#ffffff"

plt.rcParams.update({
    "font.family": "Calibri",
    "font.size": 12,
    "axes.edgecolor": REJILLA,
    "axes.labelcolor": TXT2,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.titlecolor": TXT,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": REJILLA,
    "grid.linewidth": 0.8,
    "xtick.color": TXT2,
    "ytick.color": TXT2,
    "legend.frameon": False,
    "figure.facecolor": SUPERFICIE,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def fmt_miles(x, _=None):
    return f"{x:,.0f}".replace(",", ".")


def fmt_meur(x):
    return f"{x / 1e6:,.1f} M€".replace(",", "X").replace(".", ",").replace("X", ".")


def guardar(fig, nombre):
    fig.savefig(DIR_GRAF / nombre)
    plt.close(fig)


def grafico_tam_sam_som(tam_sam, som_base):
    """Barras horizontales apiladas B2C/B2B en M€ para TAM, SAM y SOM (2026-28)."""
    som_seg = som_base.groupby("segmento")["facturacion_eur"].sum()
    niveles = ["TAM", "SAM", "SOM 2026-28"]
    b2c = [tam_sam.query("nivel == 'TAM' and segmento == 'B2C'")["facturacion_eur"].iloc[0],
           tam_sam.query("nivel == 'SAM' and segmento == 'B2C'")["facturacion_eur"].iloc[0],
           som_seg["B2C"]]
    b2b = [tam_sam.query("nivel == 'TAM' and segmento == 'B2B'")["facturacion_eur"].iloc[0],
           tam_sam.query("nivel == 'SAM' and segmento == 'B2B'")["facturacion_eur"].iloc[0],
           som_seg["B2B"]]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    y = np.arange(len(niveles))[::-1]
    ax.barh(y, b2c, height=0.55, color=C_B2C, label="B2C", edgecolor=SUPERFICIE, linewidth=2)
    ax.barh(y, b2b, left=b2c, height=0.55, color=C_B2B, label="B2B", edgecolor=SUPERFICIE, linewidth=2)
    for yi, c, b in zip(y, b2c, b2b):
        total = c + b
        ax.text(total + max(b2c[0] + b2b[0], 1) * 0.01, yi,
                f"{fmt_meur(total)}  (B2C {fmt_meur(c)} · B2B {fmt_meur(b)})",
                va="center", color=TXT, fontsize=11)
    ax.set_yticks(y, niveles)
    ax.tick_params(axis="y", length=0, labelsize=13, labelcolor=TXT)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:,.0f}".replace(",", ".")))
    ax.set_xlabel("Millones de € (sin IVA)")
    ax.set_xlim(0, (b2c[0] + b2b[0]) * 1.55)
    ax.grid(axis="y", visible=False)
    ax.set_title("Tamaño de mercado de VisionAR en España")
    ax.legend(loc="lower right", ncols=2)
    guardar(fig, "g1_tam_sam_som.png")


def grafico_som_segmentos(som_base):
    """Unidades por año y segmento (escenario base), con la facturación anotada."""
    piv = som_base.pivot(index="anio", columns="segmento", values="unidades")
    fact = som_base.groupby("anio")["facturacion_eur"].sum()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(ANIOS))
    ax.bar(x, piv["B2C"], width=0.55, color=C_B2C, label="B2C", edgecolor=SUPERFICIE, linewidth=2)
    ax.bar(x, piv["B2B"], width=0.55, bottom=piv["B2C"], color=C_B2B, label="B2B",
           edgecolor=SUPERFICIE, linewidth=2)
    for xi, anio in zip(x, ANIOS):
        total = piv.loc[anio].sum()
        ax.text(xi, total * 1.02, f"{fmt_miles(total)} uds\n{fmt_meur(fact[anio])}",
                ha="center", va="bottom", color=TXT, fontsize=11)
    ax.set_xticks(x, ANIOS)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_miles))
    ax.set_ylabel("Unidades")
    ax.set_ylim(0, piv.sum(axis=1).max() * 1.25)
    ax.grid(axis="x", visible=False)
    ax.set_title("SOM por segmento (escenario base)")
    ax.legend(loc="upper left", ncols=2)
    guardar(fig, "g2_som_segmentos.png")


def grafico_modelos(demanda, soms):
    """Demanda anual 2026-28: exponencial y logístico (base + banda de escenarios)."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for modelo, color, estilo in [("Exponencial", C_EXP, "--"), ("Logístico", C_LOG, "-")]:
        d = demanda.query("modelo == @modelo").pivot(index="anio", columns="escenario", values="unidades")
        ax.fill_between(ANIOS, d["pesimista"], d["optimista"], color=color, alpha=0.12, linewidth=0)
        ax.plot(ANIOS, d["base"], estilo, color=color, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SUPERFICIE, markeredgewidth=2, label=f"{modelo} (base)")
        ax.annotate(f"{fmt_miles(d['base'].iloc[-1])}", (ANIOS[-1], d["base"].iloc[-1]),
                    xytext=(10, 0), textcoords="offset points", va="center", color=TXT, fontsize=11)
    som = soms["base"].groupby("anio")["unidades"].sum()
    ax.scatter(ANIOS, som, marker="D", s=50, color=TXT, zorder=5, label="SOM bottom-up (base)")
    ax.set_xticks(ANIOS)
    ax.set_xlim(ANIOS[0] - 0.3, ANIOS[-1] + 0.5)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_miles))
    ax.set_ylabel("Unidades / año")
    ax.set_title("Demanda anual prevista de VisionAR Glasses")
    ax.text(0, -0.13, "Bandas sombreadas: rango pesimista-optimista de cada modelo", transform=ax.transAxes,
            color=TXT2, fontsize=10, va="top")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.95))
    guardar(fig, "g3_modelos_demanda.png")


def grafico_curva_s(params):
    """Demanda acumulada logística a largo plazo, con el horizonte 2026-28 sombreado."""
    t = np.linspace(-1, 10, 300)
    anios = ANIOS[0] + t
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axvspan(2025.5, 2028.5, color=REJILLA, alpha=0.5, linewidth=0)
    ax.text(2027, 0.02, "Horizonte del caso", ha="center", va="bottom", color=TXT2, fontsize=10,
            transform=ax.get_xaxis_transform())
    estilos = {"pesimista": ":", "base": "-", "optimista": "--"}
    for esc in ESCENARIOS:
        p = params[esc]
        ax.plot(anios, logistica(t, p["K"], p["a"], p["b"]), estilos[esc], color=C_LOG,
                linewidth=2 if esc == "base" else 1.5, label=f"{esc.capitalize()}")
    pb = params["base"]
    ax.axhline(pb["K"], color=TXT2, linewidth=1, linestyle="--")
    ax.text(anios[0], pb["K"] * 1.02, f"K base = {fmt_miles(pb['K'])} uds", color=TXT2, fontsize=10, va="bottom")
    ax.plot([ANIOS[0] + pb["b"]], [pb["K"] / 2], "o", color=C_LOG, markersize=8,
            markeredgecolor=SUPERFICIE, markeredgewidth=2)
    ax.annotate(f"b = {ANIOS[0] + pb['b']:.1f}".replace(".", ",") + "\n(mitad del techo)", (ANIOS[0] + pb["b"], pb["K"] / 2),
                xytext=(12, -28), textcoords="offset points", color=TXT, fontsize=10)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_miles))
    ax.set_ylabel("Unidades acumuladas")
    ax.set_ylim(0, max(p["K"] for p in params.values()) * 1.12)
    ax.set_title("Curva S de adopción de VisionAR (logístico)")
    ax.text(0, -0.13, "La curva logística nunca parte de cero: el valor de 2025 no son ventas. "
            "La demanda anual es la diferencia entre años consecutivos.", transform=ax.transAxes,
            color=TXT2, fontsize=10, va="top")
    ax.legend(loc="lower right", title="Escenario", title_fontsize=10)
    guardar(fig, "g4_curva_s.png")


def grafico_economia_unitaria(som_base):
    """Precio sin IVA desglosado por unidad: coste, adquisición y contribución, por segmento."""
    tot = som_base.groupby("segmento")[["unidades", "coste_adquisicion_eur"]].sum()
    adq = tot["coste_adquisicion_eur"] / tot["unidades"]
    segs = ["B2C", "B2B"]
    coste = [COSTE_UNITARIO] * 2
    adquisicion = [adq[s] for s in segs]
    contrib = [PRECIO_SIN_IVA - COSTE_UNITARIO - adq[s] for s in segs]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(segs))[::-1]
    ax.barh(y, coste, height=0.5, color="#c3c2b7", label="Coste del producto", edgecolor=SUPERFICIE, linewidth=2)
    ax.barh(y, adquisicion, left=coste, height=0.5, color=[C_B2C, C_B2B], alpha=0.45,
            edgecolor=SUPERFICIE, linewidth=2, label="Coste de adquisición")
    ax.barh(y, np.clip(contrib, 0, None), left=np.add(coste, adquisicion), height=0.5, color=[C_B2C, C_B2B],
            edgecolor=SUPERFICIE, linewidth=2, label="Contribución")
    for yi, s, c, a_, k in zip(y, segs, coste, adquisicion, contrib):
        ax.text(c / 2, yi, f"{c:.0f} €", ha="center", va="center", color=TXT, fontsize=10)
        ax.text(c + a_ / 2, yi, f"{a_:.0f} €", ha="center", va="center", color=TXT, fontsize=10)
        ax.text(c + a_ + max(k, 0) + 8, yi, f"Contribución {k:.0f} €/ud", va="center", color=TXT, fontsize=11)
    ax.axvline(PRECIO_SIN_IVA, color=TXT2, linewidth=1, linestyle="--")
    ax.text(PRECIO_SIN_IVA + 6, 0.5, f"Precio sin IVA {PRECIO_SIN_IVA:.0f} €", ha="left", va="center",
            color=TXT2, fontsize=10)
    ax.set_yticks(y, segs)
    ax.tick_params(axis="y", length=0, labelsize=13, labelcolor=TXT)
    ax.set_xlim(0, PRECIO_SIN_IVA * 1.35)
    ax.set_xlabel("€ por unidad (media 2026-28, escenario base)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Economía unitaria: B2B deja más margen por unidad")
    ax.legend(loc="lower right", fontsize=9, bbox_to_anchor=(1, -0.45), ncols=3)
    guardar(fig, "g5_economia_unitaria.png")


# =============================================================================
# EJECUCIÓN
# =============================================================================

def main():
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    pd.set_option("display.width", 140)

    # ---------------- PARTE 1 ----------------
    print("=" * 78)
    print("PARTE 1 · TAM / SAM / SOM")
    print("=" * 78)
    print(f"Precio sin IVA: {PRECIO_SIN_IVA:.2f} €  |  Margen bruto unitario: {margen_bruto_unitario():.1%}")

    tam_sam = calcular_tam_sam()
    soms = {esc: calcular_som(esc) for esc in ESCENARIOS}
    som_base = soms["base"]

    print("\nTAM y SAM:")
    print(tam_sam.to_string(index=False))
    for nivel in ["TAM", "SAM"]:
        sub = tam_sam.query("nivel == @nivel")
        print(f"  {nivel} total: {sub['unidades'].sum():,.0f} uds · {fmt_meur(sub['facturacion_eur'].sum())}")

    print("\nSOM 2026-2028 (escenario base):")
    print(som_base.to_string(index=False))
    seg = som_base.groupby("segmento")[["unidades", "facturacion_eur", "margen_bruto_eur", "contribucion_eur"]].sum()
    seg["peso_unidades"] = seg["unidades"] / seg["unidades"].sum()
    seg["peso_facturacion"] = seg["facturacion_eur"] / seg["facturacion_eur"].sum()
    print("\nPeso B2B / B2C sobre el SOM (base):")
    print(seg.to_string())

    resumen_som = pd.DataFrame({
        esc: {"unidades_3a": s["unidades"].sum(), "facturacion_3a": s["facturacion_eur"].sum(),
              "margen_bruto_3a": s["margen_bruto_eur"].sum(), "contribucion_3a": s["contribucion_eur"].sum()}
        for esc, s in soms.items()
    }).T
    sam_total = tam_sam.query("nivel == 'SAM'")
    resumen_som["pct_sam_unidades"] = resumen_som["unidades_3a"] / sam_total["unidades"].sum()
    print("\nSOM por escenario (acumulado 2026-28):")
    print(resumen_som.to_string())

    contraste = contraste_top_down_b2c(som_base)
    print("\nContraste top-down B2C (cuota implícita en el mercado español de gafas inteligentes):")
    print(contraste.to_string(index=False))

    # ---------------- PARTE 2 ----------------
    print("\n" + "=" * 78)
    print("PARTE 2 · MODELOS PREDICTIVOS")
    print("=" * 78)
    difusion, K_analogo = estimar_parametros_difusion()
    print("Parámetros de difusión por vía (b en escala t = 0 en 2026):")
    print(difusion.to_string(index=False))

    params = estimar_parametros_modelos(tam_sam, soms, difusion)
    tabla_params = pd.DataFrame(params).T
    print("\nParámetros por escenario:")
    print(tabla_params.to_string())

    demanda = proyectar_demanda(params)
    print("\nDemanda anual prevista (unidades):")
    print(demanda.pivot_table(index=["modelo", "escenario"], columns="anio", values="unidades").round(0).to_string())

    produccion = plan_produccion(demanda)
    print("\nPlan de producción orientativo (unidades):")
    print(produccion.to_string())

    # ---------------- Exportación ----------------
    tam_sam.to_csv(DIR_RES / "1_tam_sam.csv", index=False, sep=";", decimal=",")
    pd.concat(soms.values()).to_csv(DIR_RES / "2_som_escenarios.csv", index=False, sep=";", decimal=",")
    contraste.to_csv(DIR_RES / "3_contraste_top_down.csv", index=False, sep=";", decimal=",")
    difusion.to_csv(DIR_RES / "4_parametros_difusion.csv", index=False, sep=";", decimal=",")
    tabla_params.to_csv(DIR_RES / "5_parametros_modelos.csv", sep=";", decimal=",")
    demanda.to_csv(DIR_RES / "6_demanda_prevista.csv", index=False, sep=";", decimal=",")
    produccion.to_csv(DIR_RES / "7_plan_produccion.csv", sep=";", decimal=",")

    # ---------------- PARTE 3 ----------------
    grafico_tam_sam_som(tam_sam, som_base)
    grafico_som_segmentos(som_base)
    grafico_modelos(demanda, soms)
    grafico_curva_s(params)
    grafico_economia_unitaria(som_base)
    print(f"\nTablas en: {DIR_RES}\nGráficos en: {DIR_GRAF}")


if __name__ == "__main__":
    main()
