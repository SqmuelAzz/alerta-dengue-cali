"""Aplicación web "Alerta Dengue Cali" (Streamlit).

Cuatro secciones para la Secretaría de Salud Pública Distrital:
1. Situación: última semana, canal endémico y resumen de cada año.
2. Pronóstico: casos esperados en las próximas 4 semanas con semáforo; se puede
   actualizar con los casos de semanas recientes que aún no están en los datos.
3. Comunas: perfiles (K-means) e indicadores de las 22 comunas.
4. Riesgo de un caso: probabilidad de signos de alarma o dengue grave y de
   hospitalización con los datos de la primera consulta.

La interfaz se adapta al ancho de la pantalla: las tarjetas se reorganizan en una
sola columna en el celular y la barra lateral se oculta en pantallas pequeñas.

Ejecución: make app  (o streamlit run app/dashboard.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # raíz del proyecto

from app import servicio  # noqa: E402

# ---------------------------------------------------------------------------
# Identidad visual
# ---------------------------------------------------------------------------
COLORES_ZONA = {"Éxito": "#15803d", "Seguridad": "#ca8a04", "Alerta": "#ea580c",
                "Brote": "#dc2626"}
FONDOS_ZONA = {"Éxito": "#f0fdf4", "Seguridad": "#fefce8", "Alerta": "#fff7ed",
               "Brote": "#fef2f2"}
SIGNIFICADO_ZONA = {
    "Éxito": "Por debajo del cuartil 1: menos casos de los esperados para la semana.",
    "Seguridad": "Entre el cuartil 1 y la mediana: comportamiento habitual.",
    "Alerta": "Entre la mediana y el cuartil 3: más casos de lo habitual; vigilar.",
    "Brote": "Por encima del cuartil 3: situación epidémica; activar la respuesta.",
}
COLORES_GRUPO = {"A": "#dc2626", "B": "#2563eb", "C": "#7c3aed"}
COLORES_NIVEL = {"Bajo": "#15803d", "Medio": "#ea580c", "Alto": "#dc2626"}
PRIMARIO = "#0f766e"
TEXTO = "#0f172a"
SECUNDARIO = "#64748b"
COLOR_LLUVIA = "#0284c7"
COLOR_TEMPERATURA = "#e11d48"
NOMBRES_CLIMA = {"lluvia_mm": "Lluvia", "temp_anomalia": "Temperatura corregida",
                 "temp_media": "Temperatura registrada (sin corregir)"}
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]

ESTILOS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1280px;}
.encabezado {background: linear-gradient(120deg, #0f766e 0%, #155e75 100%); color: #fff;
  border-radius: 16px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;}
.encabezado h1 {color: #fff; font-size: 1.9rem; margin: 0; padding: 0; line-height: 1.2;}
.encabezado p {color: #ccfbf1; margin: .35rem 0 0 0; font-size: .95rem;}
.tarjetas {display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: .75rem; margin: .5rem 0 1rem 0;}
.tarjeta {background: #fff; border: 1px solid #e2e8f0; border-radius: 14px;
  padding: .85rem 1rem; box-shadow: 0 1px 2px rgba(15, 23, 42, .05);}
.tarjeta .etiqueta {color: #64748b; font-size: .78rem; text-transform: uppercase;
  letter-spacing: .04em; font-weight: 600;}
.tarjeta .valor {color: #0f172a; font-size: 1.65rem; font-weight: 700; line-height: 1.25;
  margin-top: .2rem;}
.tarjeta .detalle {color: #64748b; font-size: .85rem; margin-top: .15rem;}
.estado {border-radius: 14px; padding: 1rem 1.2rem; margin: .25rem 0 1rem 0;
  border-left: 6px solid;}
.estado .titulo {font-size: 1.15rem; font-weight: 700;}
.estado .texto {font-size: .92rem; margin-top: .2rem; color: #334155;}
.insignia {display: inline-block; color: #fff; padding: .12rem .65rem; border-radius: 999px;
  font-size: .8rem; font-weight: 700; letter-spacing: .02em;}
.barra {background: #e2e8f0; border-radius: 999px; height: 12px; overflow: hidden;
  margin: .5rem 0 .25rem 0;}
.barra > div {height: 100%; border-radius: 999px;}
.leyenda {display: flex; align-items: flex-start; gap: .5rem; margin: .35rem 0;
  font-size: .85rem;}
.leyenda .punto {width: 12px; height: 12px; border-radius: 50%; margin-top: .2rem;
  flex: 0 0 12px;}
.grupo {border-radius: 14px; padding: .9rem 1rem; border: 1px solid #e2e8f0;
  border-top: 5px solid;}
.grupo .titulo {font-weight: 700; font-size: 1rem;}
.grupo .texto {font-size: .88rem; color: #334155; margin-top: .25rem;}
.factores {display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: .75rem; margin: .25rem 0 1rem 0;}
.factor {display: grid; grid-template-columns: minmax(0, 1fr) 120px 58px; align-items: center;
  gap: .5rem; padding: .32rem 0; border-bottom: 1px solid #f1f5f9; font-size: .86rem;}
.factor:last-child {border-bottom: none;}
.factor .nombre {color: #0f172a; font-weight: 600;}
.factor .nombre span {color: #64748b; font-weight: 400;}
.factor .eje {position: relative; height: 10px; background: #f1f5f9; border-radius: 999px;}
.factor .eje::after {content: ""; position: absolute; left: 50%; top: -3px; bottom: -3px;
  width: 1px; background: #94a3b8;}
.factor .eje > div {position: absolute; top: 0; bottom: 0; border-radius: 999px;}
.factor .puntos {text-align: right; font-weight: 700; font-variant-numeric: tabular-nums;}
.pie {color: #94a3b8; font-size: .8rem; text-align: center; margin-top: 2rem;}
@media (max-width: 640px) {
  .block-container {padding-left: .8rem; padding-right: .8rem; padding-top: 1rem;}
  .encabezado {padding: 1rem; border-radius: 12px;}
  .encabezado h1 {font-size: 1.45rem;}
  .tarjetas {grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .5rem;}
  .tarjeta {padding: .7rem .75rem;}
  .tarjeta .valor {font-size: 1.3rem;}
  .factores {grid-template-columns: 1fr;}
  .factor {grid-template-columns: minmax(0, 1fr) 80px 52px;}
  button[data-baseweb="tab"] {padding-left: .4rem; padding-right: .4rem; font-size: .85rem;}
}
</style>
"""


# ---------------------------------------------------------------------------
# Formato y piezas de HTML
# ---------------------------------------------------------------------------

def entero(valor) -> str:
    """Número entero con punto de miles (formato colombiano)."""
    return f"{float(valor):,.0f}".replace(",", ".")


def decimal(valor, digitos: int = 1) -> str:
    """Número con coma decimal y punto de miles."""
    texto = f"{float(valor):,.{digitos}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


def fecha_corta(fecha) -> str:
    fecha = pd.Timestamp(fecha)
    return f"{fecha.day} {MESES[fecha.month - 1]} {fecha.year}"


def insignia(texto: str, color: str) -> str:
    return f"<span class='insignia' style='background:{color}'>{texto}</span>"


def tarjeta(etiqueta: str, valor: str, detalle: str = "", color: str | None = None) -> str:
    borde = f" style='border-top:4px solid {color}'" if color else ""
    return (f"<div class='tarjeta'{borde}><div class='etiqueta'>{etiqueta}</div>"
            f"<div class='valor'>{valor}</div><div class='detalle'>{detalle}</div></div>")


def mostrar_tarjetas(tarjetas: list[str]) -> None:
    st.markdown(f"<div class='tarjetas'>{''.join(tarjetas)}</div>", unsafe_allow_html=True)


def mostrar_estado(titulo: str, texto: str, zona: str | None) -> None:
    color = COLORES_ZONA.get(zona, SECUNDARIO)
    fondo = FONDOS_ZONA.get(zona, "#f8fafc")
    st.markdown(f"<div class='estado' style='border-color:{color};background:{fondo}'>"
                f"<div class='titulo' style='color:{color}'>{titulo}</div>"
                f"<div class='texto'>{texto}</div></div>", unsafe_allow_html=True)


def mostrar_tabla(tabla: pd.DataFrame) -> None:
    """Tabla a todo el ancho (compatible con versiones nuevas y anteriores de Streamlit)."""
    try:
        st.dataframe(tabla, hide_index=True, width="stretch")
    except Exception:  # versiones que solo aceptan ancho en píxeles
        st.dataframe(tabla, hide_index=True, use_container_width=True)


def boton_descarga(tabla: pd.DataFrame, nombre: str, etiqueta: str) -> None:
    st.download_button(etiqueta, tabla.to_csv(index=False).encode("utf-8-sig"),
                       file_name=nombre, mime="text/csv")


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def _estilo(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cbd5e1")
    ax.tick_params(colors="#475569", labelsize=9)
    ax.grid(axis="y", color="#e2e8f0", lw=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: entero(v)))


def grafico_canal(datos: pd.DataFrame, anio: int):
    """Casos de un año sobre las bandas del canal endémico."""
    datos = datos.sort_values("semana_epi")
    s = datos["semana_epi"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.fill_between(s, 0, datos["q1"], color=COLORES_ZONA["Éxito"], alpha=0.18, lw=0,
                    label="Éxito")
    ax.fill_between(s, datos["q1"], datos["mediana"], color=COLORES_ZONA["Seguridad"],
                    alpha=0.22, lw=0, label="Seguridad")
    ax.fill_between(s, datos["mediana"], datos["q3"], color=COLORES_ZONA["Alerta"], alpha=0.22,
                    lw=0, label="Alerta")
    ax.plot(s, datos["q3"], color=COLORES_ZONA["Brote"], lw=1.3, ls="--",
            label="Umbral de brote (Q3)")
    ax.plot(s, datos["casos"], color=TEXTO, lw=2.2, label=f"Casos {anio}")
    brote = datos[datos["supera_q3"].fillna(False).astype(bool)]
    ax.scatter(brote["semana_epi"], brote["casos"], color=COLORES_ZONA["Brote"], s=16,
               zorder=3, label="Semana en brote")
    ax.set_xlim(1, max(52, int(s.max())))
    ax.set_xlabel("Semana epidemiológica (inicio de síntomas)", color="#475569")
    ax.set_ylabel("Casos por semana", color="#475569")
    _estilo(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=6, fontsize=8,
              frameon=False)
    fig.tight_layout()
    return fig


def grafico_clima(tabla: pd.DataFrame, anio: int, cambios: list[dict]):
    """Casos, lluvia y temperatura de cada semana del año, con el cambio de sensores."""
    s = tabla["semana_epi"]
    fig, (ax_c, ax_l, ax_t) = plt.subplots(3, 1, figsize=(10, 6.4), sharex=True,
                                           gridspec_kw={"height_ratios": [1.2, 1, 1]})
    ax_c.plot(s, tabla["casos"], color=TEXTO, lw=2)
    ax_c.set_ylabel("Casos", color="#475569")
    ax_l.bar(s, tabla["lluvia_mm"], color=COLOR_LLUVIA, width=0.8)
    ax_l.set_ylabel("Lluvia (mm)", color="#475569")
    ax_t.plot(s, tabla["temp_media"], color=COLOR_TEMPERATURA, lw=1.8, marker="o", ms=2.5)
    ax_t.set_ylabel("Temperatura (°C)", color="#475569")
    for ax, columna in ((ax_l, "lluvia_mm"), (ax_t, "temp_media")):
        _estilo(ax)
        if tabla[columna].notna().sum() == 0:
            ax.text(0.5, 0.5, "El IDEAM no publicó lecturas de este año", ha="center",
                    va="center", transform=ax.transAxes, color=SECUNDARIO, fontsize=9)
            ax.set_yticks([])
    ax_t.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: decimal(v, 0)))
    _estilo(ax_c)
    if cambios:
        inicio = min(c["semana_epi"] for c in cambios)
        fin = max(c["semana_epi"] for c in cambios)
        for ax in (ax_c, ax_l, ax_t):
            ax.axvspan(inicio - 0.5, fin + 0.5, color="#94a3b8", alpha=0.16, lw=0)
        ax_t.text(inicio, 1.02, "Cambio de sensores del IDEAM", fontsize=8, va="bottom",
                  color=SECUNDARIO, transform=ax_t.get_xaxis_transform())
    ax_t.set_xlim(0.5, max(52, int(s.max())) + 0.5)
    ax_t.set_xlabel(f"Semana epidemiológica de {anio}", color="#475569")
    fig.align_ylabels()
    fig.tight_layout()
    return fig


def grafico_rezagos(rezagos: pd.DataFrame):
    """Correlación entre el clima de hace k semanas y el crecimiento de los casos."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    estilos = {"lluvia_mm": (COLOR_LLUVIA, "-"), "temp_anomalia": (COLOR_TEMPERATURA, "-"),
               "temp_media": ("#94a3b8", "--")}
    for variable, (color, linea) in estilos.items():
        datos = rezagos[rezagos["variable"] == variable].sort_values("rezago_semanas")
        if datos.empty:
            continue
        ax.plot(datos["rezago_semanas"], datos["rho"], color=color, ls=linea, lw=2,
                marker="o", ms=4, label=NOMBRES_CLIMA[variable])
        significativas = datos[datos["p_value"] < 0.05]
        ax.scatter(significativas["rezago_semanas"], significativas["rho"], s=60,
                   facecolors="none", edgecolors=color, lw=1.4)
    ax.axhline(0, color="#94a3b8", lw=1)
    ax.set_xlabel("Semanas de anticipación del clima", color="#475569")
    ax.set_ylabel("Correlación (Spearman)", color="#475569")
    _estilo(ax)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: decimal(v, 1)))
    ax.set_xticks(range(0, int(rezagos["rezago_semanas"].max()) + 1))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8,
              frameon=False)
    fig.tight_layout()
    return fig


def grafico_pronostico(historial: pd.DataFrame, tabla: pd.DataFrame):
    """Últimas semanas observadas, el pronóstico con su intervalo y el umbral de brote."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fechas_futuras = pd.to_datetime(tabla["fecha_inicio"])
    todas = pd.concat([pd.to_datetime(historial["fecha_inicio"]), fechas_futuras])
    umbral = pd.concat([historial["q3"], tabla["q3"]])
    ax.step(todas, umbral, where="mid", color=COLORES_ZONA["Brote"], lw=1.3, ls="--",
            label="Umbral de brote (Q3)")
    oficiales = historial[~historial["agregada"]]
    ax.plot(pd.to_datetime(oficiales["fecha_inicio"]), oficiales["casos_modelo"], color=TEXTO,
            lw=2.2, marker="o", ms=3.5, label="Casos observados")
    agregadas = historial[historial["agregada"]]
    if not agregadas.empty:
        ax.plot(pd.to_datetime(agregadas["fecha_inicio"]), agregadas["casos_modelo"],
                color="#0891b2", lw=0, marker="D", ms=6, label="Semanas agregadas")
    ultima = historial.iloc[-1]
    x = pd.concat([pd.Series([pd.Timestamp(ultima["fecha_inicio"])]), fechas_futuras])
    y = [ultima["casos_modelo"], *tabla["pronostico"]]
    bajo = [ultima["casos_modelo"], *tabla["limite_inferior"]]
    alto = [ultima["casos_modelo"], *tabla["limite_superior"]]
    ax.fill_between(x, bajo, alto, color=PRIMARIO, alpha=0.18, lw=0, label="Intervalo 80 %")
    ax.plot(x, y, color=PRIMARIO, lw=2.2, ls="--", marker="s", ms=5, label="Pronóstico")
    ax.axvline(pd.Timestamp(ultima["fecha_inicio"]), color="#94a3b8", lw=1, ls=":")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%y"))
    ax.set_ylabel("Casos por semana", color="#475569")
    ax.set_ylim(bottom=0)
    _estilo(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=8,
              frameon=False)
    fig.tight_layout()
    return fig


def grafico_comunas(tabla: pd.DataFrame, seleccionada: int | None = None):
    orden = tabla.sort_values("tasa_100k")
    fig, ax = plt.subplots(figsize=(8, 7))
    colores = [COLORES_GRUPO.get(g, "#94a3b8") for g in orden["grupo"]]
    barras = ax.barh(orden["territorio"], orden["tasa_100k"], color=colores, height=0.7)
    for barra, comuna in zip(barras, orden["comuna"], strict=True):
        if comuna == seleccionada:
            barra.set_edgecolor(TEXTO)
            barra.set_linewidth(2)
    ax.axvline(tabla["tasa_100k"].median(), color=SECUNDARIO, lw=1, ls=":")
    ax.text(tabla["tasa_100k"].median(), len(orden) - 0.3, " mediana", color=SECUNDARIO,
            fontsize=8, va="bottom")
    ax.set_xlabel(f"Casos por 100.000 habitantes ({tabla['periodo'].iloc[0]})", color="#475569")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cbd5e1")
    ax.tick_params(colors="#475569", labelsize=9)
    ax.grid(axis="x", color="#e2e8f0", lw=0.8)
    ax.set_axisbelow(True)
    manijas = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORES_GRUPO.values()]
    ax.legend(manijas, [f"Grupo {g}" for g in COLORES_GRUPO], loc="lower right", fontsize=8,
              frameon=False)
    fig.tight_layout()
    return fig


def mostrar_grafico(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Secciones
# ---------------------------------------------------------------------------

@st.cache_resource
def recursos() -> servicio.Recursos:
    return servicio.cargar_recursos()


def barra_lateral(r: servicio.Recursos) -> None:
    actual = servicio.ultima_semana(r)
    with st.sidebar:
        st.markdown("### 🦟 Alerta Dengue Cali")
        st.caption("Producto de datos para la vigilancia del dengue en Santiago de Cali.")
        st.markdown(f"**Datos hasta:** semana {actual['semana_epi']} de {actual['anio_epi']} "
                    f"(desde el {fecha_corta(actual['fecha_inicio'])})")
        st.markdown("**Cómo leer el semáforo**")
        leyenda = "".join(
            f"<div class='leyenda'><span class='punto' style='background:{COLORES_ZONA[z]}'>"
            f"</span><span><b>{z}.</b> {SIGNIFICADO_ZONA[z]}</span></div>"
            for z in COLORES_ZONA)
        st.markdown(leyenda, unsafe_allow_html=True)
        st.markdown("**Fuentes**")
        st.caption("SIVIGILA (Instituto Nacional de Salud), Secretaría de Salud Pública y "
                   "Departamento Administrativo de Planeación de Cali, DANE (población) e "
                   "IDEAM (clima). Periodo 2020-2025, semana epidemiológica de inicio de "
                   "síntomas.")
        st.markdown("**Autores**")
        st.caption("Samuel Acevedo Mambuscay y Luis Carlos Tez · Universidad Libre, "
                   "seccional Cali")


def seccion_situacion(r: servicio.Recursos) -> None:
    actual = servicio.ultima_semana(r)
    zona = actual["zona"]
    mostrar_estado(
        f"Semana {actual['semana_epi']} de {actual['anio_epi']}: zona de {zona.lower()}",
        f"{entero(actual['casos'])} casos con inicio de síntomas en la semana del "
        f"{fecha_corta(actual['fecha_inicio'])}. {SIGNIFICADO_ZONA.get(zona, '')}", zona)
    mostrar_tarjetas([
        tarjeta("Casos de la semana", entero(actual["casos"]),
                f"Semana {actual['semana_epi']} de {actual['anio_epi']}", COLORES_ZONA[zona]),
        tarjeta("Umbral de brote", entero(actual["canal_q3"]),
                "Cuartil 3 del canal para esa semana"),
        tarjeta("Valor habitual", entero(actual["canal_mediana"]),
                "Mediana del canal para esa semana"),
        tarjeta("Zona del canal", insignia(zona, COLORES_ZONA[zona]),
                "Éxito · Seguridad · Alerta · Brote"),
    ])

    st.markdown("#### Canal endémico por año")
    anios = sorted(int(a) for a in r.situacion["anio_epi"].unique())
    anio = st.radio("Año", anios[::-1], index=0, horizontal=True,
                    label_visibility="collapsed")
    resumen = servicio.resumen_anio(r, anio)
    datos = r.situacion[r.situacion["anio_epi"] == anio]
    mostrar_tarjetas([
        tarjeta(f"Casos {anio}", entero(resumen["casos"]),
                f"{resumen['semanas_con_datos']} semanas completas"),
        tarjeta("Incidencia", "—" if resumen["tasa_100k"] is None
                else decimal(resumen["tasa_100k"]), "Casos por 100.000 habitantes"),
        tarjeta("Semanas en brote", f"{resumen['semanas_brote']}",
                f"de {resumen['semanas_con_datos']} con datos completos",
                COLORES_ZONA["Brote"] if resumen["semanas_brote"] else None),
        tarjeta("Semana pico", "—" if resumen["semana_pico"] is None
                else f"S{resumen['semana_pico']}",
                "" if resumen["casos_pico"] is None else f"{entero(resumen['casos_pico'])} casos"),
    ])
    mostrar_grafico(grafico_canal(datos, anio))
    if anio in (2022, 2023):
        st.caption("Los archivos del Distrito para este año se publicaron con corte el 6 de "
                   "diciembre: las últimas semanas no tienen datos.")
    with st.expander("¿Cómo se construye el canal endémico?"):
        st.markdown(
            f"Para cada semana se toman los casos de los años de baja transmisión "
            f"({r.canal['anios_base'].iloc[0]}) en una ventana de ±2 semanas y se calculan "
            "el cuartil 1, la mediana y el cuartil 3 (método de cuartiles de Bortman). Una "
            "semana por encima del cuartil 3 se considera **en brote**.")
    columnas = ["anio_epi", "semana_epi", "fecha_inicio", "casos", "q1", "mediana", "q3",
                "zona"]
    boton_descarga(datos[columnas], f"situacion_semanal_{anio}.csv",
                   f"Descargar las semanas de {anio} (CSV)")
    seccion_clima(r, anio)


def seccion_clima(r: servicio.Recursos, anio: int) -> None:
    """Lluvia y temperatura del IDEAM junto a los casos del año elegido."""
    if not {"lluvia_mm", "temp_media"} & set(r.serie.columns):
        return                                    # pipeline anterior sin datos del IDEAM
    tabla, resumen = servicio.clima_anio(r, anio)
    st.markdown(f"#### 🌦️ Clima y casos en {anio} (IDEAM)")
    relacion = servicio.relacion_clima(r)
    lluvia, anomalia = relacion.get("lluvia_mm"), relacion.get("temp_anomalia")
    sin_relacion = all(v is None or abs(v["rho"]) < 0.3 for v in (lluvia, anomalia))
    mostrar_tarjetas([
        tarjeta("Lluvia semanal", "—" if resumen["lluvia_semanal_media_mm"] is None
                else f"{decimal(resumen['lluvia_semanal_media_mm'])} mm",
                f"Promedio en {resumen['semanas_con_lluvia']} de {resumen['semanas']} semanas "
                "con lecturas", COLOR_LLUVIA),
        tarjeta("Temperatura media", "—" if resumen["temperatura_media_c"] is None
                else f"{decimal(resumen['temperatura_media_c'])} °C",
                f"{resumen['semanas_con_temperatura']} de {resumen['semanas']} semanas con "
                "lecturas", COLOR_TEMPERATURA),
        tarjeta("Semanas sin datos del IDEAM", f"{resumen['semanas_sin_clima']}",
                "Sin lluvia ni temperatura publicadas"),
        tarjeta("¿Anticipa los casos?",
                insignia("No con estos datos" if sin_relacion else "Relación débil",
                         SECUNDARIO if sin_relacion else COLORES_ZONA["Alerta"]),
                "Correlación con el crecimiento de los casos"),
    ])
    mostrar_grafico(grafico_clima(tabla, anio, resumen["cambio_sistema"]))
    origen = ("Lluvia y temperatura: promedio de las estaciones urbanas del IDEAM (Universidad "
              "del Valle, Base Aérea y Siloé) con al menos la mitad de sus lecturas en la semana; "
              "los huecos son semanas sin lecturas publicadas.")
    anio_cambio = min((c["anio_epi"] for c in r.cambio_sistema), default=None)
    if resumen["cambio_sistema"]:
        detalle = ", ".join(f"{c['estacion']} (semana {c['semana_epi']})"
                            for c in resumen["cambio_sistema"])
        origen += (f" Franja gris: el IDEAM cambió el sistema de medición de temperatura en "
                   f"{detalle}. El sistema nuevo registra entre 1 y 4 °C menos que el anterior, "
                   "así que la diferencia con los años previos no es un cambio real del clima.")
    elif anio_cambio is not None and anio < anio_cambio:
        origen += (f" La temperatura de este año se midió con el sistema anterior del IDEAM, "
                   f"que registra entre 1 y 4 °C más que el usado desde {anio_cambio}: no se "
                   "debe comparar directamente con los años siguientes.")
    st.caption(origen)
    if r.clima_rezagos is not None and not r.clima_rezagos.empty:
        with st.expander("¿El clima de semanas anteriores anticipa los casos?"):
            st.markdown(
                "La lluvia crea criaderos del mosquito y el calor acelera su ciclo, así que se "
                "esperaría que el clima de hace algunas semanas anticipe el aumento de casos. "
                "Para comprobarlo se comparó el clima de cada semana con el crecimiento de "
                "los casos en las 4 semanas siguientes, con 0 a 12 semanas de anticipación "
                "(los círculos marcan correlaciones con p < 0,05).")
            mostrar_grafico(grafico_rezagos(r.clima_rezagos))
            puntos = []
            media = relacion.get("temp_media")
            if media:
                puntos.append(
                    f"- **Temperatura registrada:** parece anticipar los casos (ρ = "
                    f"{decimal(media['rho'], 2)} a {media['rezago_semanas']} semanas), pero es "
                    "un efecto del cambio de sensores de 2024, que coincidió con el final de "
                    "la epidemia.")
            if anomalia:
                puntos.append(
                    f"- **Temperatura corregida** (diferencia con la media de cada estación y "
                    f"sistema de medición): la relación desaparece (la más fuerte es ρ = "
                    f"{decimal(anomalia['rho'], 2)}, p = {decimal(anomalia['p_value'], 2)}).")
            if lluvia:
                puntos.append(
                    f"- **Lluvia:** solo una relación débil a {lluvia['rezago_semanas']} "
                    f"semanas (ρ = {decimal(lluvia['rho'], 2)}); con {len(r.clima_rezagos)} "
                    "comparaciones puede ser casualidad, y su signo es el contrario al "
                    "esperado.")
            puntos.append(
                "- Por eso el pronóstico no usa el clima: el modelo con temperatura quedó "
                "empatado con el modelo sin ella y se eligió el más simple. Faltan lecturas "
                "del IDEAM en muchas semanas (casi todo 2021 y 2023), y el clima de Cali "
                "varía poco de una semana a otra.")
            st.markdown("\n".join(puntos))
    boton_descarga(tabla, f"clima_y_casos_{anio}.csv",
                   f"Descargar casos y clima de {anio} (CSV)")


def seccion_pronostico(r: servicio.Recursos) -> None:
    st.markdown("Casos esperados en las próximas cuatro semanas y su zona en el canal "
                "endémico. Si ya tiene el conteo de semanas posteriores a la última de los "
                "datos, agréguelo para actualizar el pronóstico.")
    with st.form("casos_recientes"):
        texto = st.text_input(
            "Casos de las semanas recientes (opcional)", placeholder="Ejemplo: 45, 52",
            help="Una cifra por semana, en orden, separadas por comas. Se agregan después de "
                 "la última semana de los datos.")
        enviado = st.form_submit_button("Actualizar pronóstico", type="primary")
    casos = []
    if enviado and texto.strip():
        try:
            casos = [int(v) for v in texto.replace(";", ",").split(",") if v.strip()]
        except ValueError:
            st.error("Escriba solo números enteros separados por comas.")
            return
    try:
        historial, tabla = servicio.historial_y_pronostico(r, casos)
    except servicio.EntradaInvalida as error:
        st.error(str(error))
        return

    if tabla["alerta"].any():
        primera = tabla[tabla["alerta"]].iloc[0]
        mostrar_estado("Alerta de brote",
                       f"El pronóstico supera el umbral de brote desde la semana "
                       f"{primera['semana_epi']} de {primera['anio_epi']}. Se recomienda "
                       "activar las acciones de control vectorial y preparar la red de "
                       "atención.", "Brote")
    else:
        maxima = tabla["zona_pronostico"].map(
            {z: i for i, z in enumerate(COLORES_ZONA)}).max()
        zona = list(COLORES_ZONA)[int(maxima)]
        mostrar_estado("Sin alerta de brote",
                       "El pronóstico de las próximas cuatro semanas está por debajo del "
                       "umbral de brote del canal endémico.", zona)

    mostrar_tarjetas([
        tarjeta(f"Semana {f['semana_epi']} · {f['anio_epi']}",
                f"{entero(f['pronostico'])} casos",
                f"Entre {entero(f['limite_inferior'])} y {entero(f['limite_superior'])} · "
                f"{insignia(f['zona_pronostico'], COLORES_ZONA[f['zona_pronostico']])}",
                COLORES_ZONA[f["zona_pronostico"]])
        for _, f in tabla.iterrows()
    ])
    mostrar_grafico(grafico_pronostico(historial, tabla))
    st.caption("Línea continua: semanas observadas. Línea discontinua y franja: pronóstico y "
               "su intervalo del 80 %. La línea roja es el umbral de brote de cada semana.")

    with st.expander("¿Qué tan preciso es el pronóstico?"):
        prueba = r.metadatos["pronostico"]["prueba"]["por_horizonte"]
        desempeno = pd.DataFrame([
            {"Horizonte": f"{h} semana{'s' if int(h) > 1 else ''}",
             "Modelo": m["modelo"].replace("_", " "),
             "Error medio (casos)": round(m["mae"]),
             "Error del ingenuo": round(m["mae_ingenuo"]),
             "Mejora (%)": round(m["mejora_vs_ingenuo_pct"], 1),
             "Semanas de brote detectadas (%)": round(m["alerta_sensibilidad_pct"], 1)}
            for h, m in prueba.items()])
        mostrar_tabla(desempeno)
        coberturas = [m["cobertura_intervalo_pct"] for m in prueba.values()]
        st.caption("Evaluado en 2024-2025, semanas que el modelo no vio al entrenarse. El "
                   "ingenuo repite los casos de la última semana. El intervalo del 80 % cubrió "
                   f"entre el {decimal(min(coberturas), 0)} % y el "
                   f"{decimal(max(coberturas), 0)} % de las semanas reales: en epidemias "
                   "grandes la incertidumbre es mayor que la que muestra la franja.")
    columnas = ["horizonte", "anio_epi", "semana_epi", "fecha_inicio", "pronostico",
                "limite_inferior", "limite_superior", "q3", "zona_pronostico", "alerta"]
    redondeo = dict.fromkeys(["pronostico", "limite_inferior", "limite_superior", "q3"], 1)
    boton_descarga(tabla[columnas].round(redondeo), "pronostico_dengue_cali.csv",
                   "Descargar el pronóstico (CSV)")


def seccion_comunas(r: servicio.Recursos) -> None:
    tabla = servicio.tabla_comunas(r)
    meta = r.metadatos["comunas"]
    periodo = tabla["periodo"].iloc[0]
    st.markdown(f"Las 22 comunas se agrupan en {meta['k']} perfiles según su incidencia, "
                f"severidad, hospitalización, edad, estrato, afiliación y carga histórica de "
                f"arbovirosis (periodo {periodo}).")
    grupos = []
    for grupo in sorted(meta["grupos"]):
        miembros = meta["grupos"][grupo]
        sub = tabla[tabla["grupo"] == grupo]
        color = COLORES_GRUPO.get(grupo, SECUNDARIO)
        grupos.append(
            f"<div class='grupo' style='border-top-color:{color}'>"
            f"<div class='titulo' style='color:{color}'>Grupo {grupo}</div>"
            f"<div class='texto'>{meta['descripciones'][grupo]}.</div>"
            f"<div class='texto'><b>Comunas:</b> {', '.join(str(c) for c in miembros)}</div>"
            f"<div class='texto'><b>Incidencia media:</b> {entero(sub['tasa_100k'].mean())} "
            f"por 100.000</div></div>")
    st.markdown(f"<div class='tarjetas'>{''.join(grupos)}</div>", unsafe_allow_html=True)

    izquierda, derecha = st.columns([3, 2], gap="large")
    with derecha:
        numero = st.selectbox("Comuna", tabla["comuna"].tolist(),
                              format_func=lambda c: f"Comuna {c}")
        detalle = servicio.comuna(r, int(numero))
        mediana = tabla[["tasa_100k", "pct_severos", "pct_subsidiado", "edad_mediana"]].median()
        color = COLORES_GRUPO.get(detalle["grupo"], SECUNDARIO)
        st.markdown(f"{insignia('Grupo ' + detalle['grupo'], color)} {detalle['perfil']}",
                    unsafe_allow_html=True)

        def comparacion(valor, referencia):
            return "por encima" if valor > referencia else "por debajo"

        mostrar_tarjetas([
            tarjeta("Incidencia", entero(detalle["tasa_100k"]),
                    f"{comparacion(detalle['tasa_100k'], mediana['tasa_100k'])} de la mediana "
                    f"({entero(mediana['tasa_100k'])})", color),
            tarjeta("Casos severos", f"{decimal(detalle['pct_severos'])} %",
                    "Con signos de alarma o graves"),
            tarjeta("Hospitalizados", f"{decimal(detalle['pct_hospitalizados'])} %",
                    f"{entero(detalle['casos'])} casos en el periodo"),
            tarjeta("Estrato medio", decimal(detalle["estrato_medio"]),
                    f"{decimal(detalle['pct_subsidiado'])} % régimen subsidiado"),
        ])
    with izquierda:
        mostrar_grafico(grafico_comunas(tabla, int(numero)))

    st.markdown("#### Indicadores por comuna")
    columnas = {"comuna": "Comuna", "grupo": "Grupo", "casos": "Casos",
                "tasa_100k": "Incidencia por 100.000", "pct_severos": "% severos",
                "pct_hospitalizados": "% hospitalizados", "edad_mediana": "Edad mediana",
                "pct_menores_15": "% menores de 15", "estrato_medio": "Estrato medio",
                "pct_subsidiado": "% subsidiado"}
    vista = tabla[list(columnas)].rename(columns=columnas).round(1)
    mostrar_tabla(vista)
    boton_descarga(vista, "indicadores_comunas.csv", "Descargar los indicadores (CSV)")


def _resultado(titulo: str, resultado: dict, descripcion: str) -> str:
    color = COLORES_NIVEL[resultado["nivel_riesgo"]]
    porcentaje = resultado["probabilidad"] * 100
    promedio, veces = resultado.get("promedio"), resultado.get("veces_promedio")
    comparacion = ("" if promedio is None else
                   f"<b>Promedio de los casos: {decimal(promedio * 100, 0)} %</b> · este caso: "
                   f"{decimal(veces, 1)} veces el promedio.<br>")
    referencia = resultado.get("pct_positivos_en_nivel_prueba")
    contexto = ("" if referencia is None else
                f"En la evaluación del modelo, el {decimal(referencia, 0)} % de los casos de "
                f"nivel {resultado['nivel_riesgo'].lower()} {descripcion}.")
    marca = "" if promedio is None else (
        f"<div style='position:absolute;left:{promedio * 100:.0f}%;top:-4px;bottom:-4px;"
        f"width:2px;background:{TEXTO}' title='Promedio de los casos'></div>")
    return (f"<div class='tarjeta' style='border-top:4px solid {color}'>"
            f"<div class='etiqueta'>{titulo}</div>"
            f"<div class='valor'>{decimal(porcentaje, 0)} % "
            f"{insignia('Riesgo ' + resultado['nivel_riesgo'].lower(), color)}</div>"
            f"<div class='barra' style='position:relative;overflow:visible'>"
            f"<div style='width:{porcentaje:.0f}%;background:{color}'></div>{marca}</div>"
            f"<div class='detalle'>{comparacion}{contexto}</div></div>")


def _factores(titulo: str, factores: list[dict], minimo: float = 0.5) -> str:
    """Lista de las características que más suben o bajan la probabilidad del caso."""
    visibles = [f for f in factores if abs(f["puntos"]) >= minimo]
    escala = max([abs(f["puntos"]) for f in visibles] + [5.0])
    filas = []
    for f in visibles:
        color = COLORES_NIVEL["Alto"] if f["puntos"] > 0 else COLORES_NIVEL["Bajo"]
        ancho = abs(f["puntos"]) / escala * 50
        posicion = f"left:50%;width:{ancho:.0f}%" if f["puntos"] > 0 else \
            f"right:50%;width:{ancho:.0f}%"
        filas.append(f"<div class='factor'><div class='nombre'>{f['factor']} "
                     f"<span>· {f['valor']}</span></div>"
                     f"<div class='eje'><div style='{posicion};background:{color}'></div></div>"
                     f"<div class='puntos' style='color:{color}'>"
                     f"{'+' if f['puntos'] > 0 else '−'}{decimal(abs(f['puntos']), 1)}</div>"
                     f"</div>")
    if not filas:
        filas.append("<div class='detalle'>Ninguna característica cambia el resultado en más "
                     "de medio punto: el caso se parece al promedio.</div>")
    return (f"<div class='tarjeta'><div class='etiqueta'>{titulo}</div>"
            f"{''.join(filas)}</div>")


def _como_funciona(r: servicio.Recursos) -> str:
    """Datos y desempeño de los dos clasificadores, leídos de sus metadatos."""
    def resumen(meta: dict, nombre: str) -> str:
        if not meta:
            return ""
        prueba = meta.get("prueba", {})
        return (f"- **{nombre}:** {entero(meta.get('n_train', 0))} "
                f"{meta.get('datos', 'casos')}; AUC de {decimal(prueba.get('roc_auc', 0), 2)} "
                f"en {entero(prueba.get('n_test', 0))} casos de prueba.")
    lineas = [
        "Dos modelos Random Forest estiman cada probabilidad con la información disponible "
        "en la primera consulta:",
        "",
        resumen(r.metadatos.get("riesgo", {}), "Signos de alarma o dengue grave"),
        resumen(r.metadatos.get("hospitalizacion", {}), "Hospitalización"),
        "",
        "La clasificación clínica y los síntomas solo vienen en los casos que publica el "
        "Distrito (2022-2023); los microdatos del INS (2020, 2021, 2024 y 2025) traen la "
        "hospitalización pero no la clasificación. Por eso el modelo de hospitalización "
        "usa todos los años, con las variables que traen ambas fuentes (edad, sexo, "
        "estrato, afiliación, días desde el inicio de síntomas y condición de migrante), "
        "y el de signos de alarma usa solo 2022-2023.",
        "",
        "Los signos de alarma no son variables del modelo porque definen la clasificación. "
        "Los niveles Bajo, Medio y Alto dividen los casos en tercios según su probabilidad. "
        "La edad es el factor que más pesa: de los pacientes de 80 años o más, el 61 % tuvo "
        "signos de alarma y el 53 % fue hospitalizado. Hay pocos casos de más de 90 años, "
        "así que para ellos el modelo responde como para los de 80-90.",
    ]
    return "\n".join(lineas)


def seccion_riesgo(r: servicio.Recursos) -> None:
    st.markdown("Probabilidad de que un caso presente **signos de alarma o dengue grave** y "
                "de que sea **hospitalizado**, con la información disponible en la primera "
                "consulta.")
    with st.form("caso"):
        c1, c2 = st.columns(2)
        with c1:
            edad = st.number_input("Edad (años)", 0, 110, 30)
            sexo = st.selectbox("Sexo", servicio.SEXOS,
                                format_func={"F": "Femenino", "M": "Masculino"}.get)
            dias = st.number_input("Días desde el inicio de los síntomas", 0, 30, 3)
        with c2:
            estrato = st.selectbox("Estrato", [None, 1, 2, 3, 4, 5, 6],
                                   format_func=lambda v: "Sin dato" if v is None else str(v))
            comuna = st.selectbox("Comuna de residencia", [None, *range(1, 23)],
                                  format_func=lambda v: "Sin dato" if v is None
                                  else f"Comuna {v}")
            afiliacion = st.selectbox("Régimen de afiliación", [None, *servicio.AFILIACIONES],
                                      format_func=lambda v: "Sin dato" if v is None else v)
        st.markdown("**Síntomas y antecedentes**")
        st.caption("Son los síntomas generales del dengue: casi todos los pacientes los "
                   "tienen, por eso cambian poco el resultado. Los signos de alarma (dolor "
                   "abdominal intenso, vómito persistente, sangrado, somnolencia…) no se "
                   "piden: son lo que el modelo anticipa, y un paciente que ya los tiene es "
                   "un caso de alarma que se maneja según la guía clínica.")
        etiquetas = {"fiebre": "Fiebre", "cefalea": "Dolor de cabeza",
                     "dolor_retroocular": "Dolor detrás de los ojos", "mialgias": "Dolor muscular",
                     "artralgia": "Dolor articular", "erupcion": "Erupción en la piel",
                     "migrante": "Persona migrante",
                     "desplazamiento_15d": "Viajó en los últimos 15 días"}
        columnas = st.columns(2)
        marcados = {campo: columnas[i % 2].checkbox(texto, value=campo == "fiebre")
                    for i, (campo, texto) in enumerate(etiquetas.items())}
        enviado = st.form_submit_button("Calcular riesgo", type="primary")
    if not enviado:
        st.caption("Complete los datos del caso y presione «Calcular riesgo».")
        return
    caso = {"edad": int(edad), "sexo": sexo, "dias_sintomas_consulta": int(dias),
            "estrato": estrato, "comuna": comuna, "afiliacion": afiliacion, **marcados}
    try:
        resultado = servicio.evaluar_caso(r, caso)
    except servicio.EntradaInvalida as error:
        st.error(str(error))
        return
    tarjetas = [_resultado("Signos de alarma o dengue grave", resultado,
                           "tuvo signos de alarma o dengue grave")]
    if resultado.get("hospitalizacion"):
        tarjetas.append(_resultado("Hospitalización", resultado["hospitalizacion"],
                                   "fue hospitalizado"))
    mostrar_tarjetas(tarjetas)
    explicaciones = [_factores("Qué pesa en el riesgo de signos de alarma",
                               resultado.get("factores", []))]
    if resultado.get("hospitalizacion"):
        explicaciones.append(_factores("Qué pesa en el riesgo de hospitalización",
                                       resultado["hospitalizacion"].get("factores", [])))
    if any(res.get("factores") for res in (resultado, resultado.get("hospitalizacion") or {})):
        st.markdown("#### ¿Qué explica el resultado?")
        st.caption("Puntos porcentuales que cada característica suma (rojo) o resta (verde) "
                   "frente a los valores habituales de los casos notificados en Cali.")
        st.markdown(f"<div class='factores'>{''.join(explicaciones)}</div>",
                    unsafe_allow_html=True)
    st.info("Herramienta de apoyo para priorizar el seguimiento de casos. No reemplaza la "
            "valoración clínica ni los criterios de la guía de atención del dengue.")
    with st.expander("¿Cómo funciona el cálculo?"):
        st.markdown(_como_funciona(r))


def main() -> None:
    st.set_page_config(page_title="Alerta Dengue Cali", page_icon="🦟", layout="wide",
                       initial_sidebar_state="auto")
    st.markdown(ESTILOS, unsafe_allow_html=True)
    st.markdown("<div class='encabezado'><h1>🦟 Alerta Dengue Cali</h1>"
                "<p>Situación, pronóstico y riesgo del dengue en Santiago de Cali · "
                "Secretaría de Salud Pública Distrital</p></div>", unsafe_allow_html=True)
    try:
        r = recursos()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()
        return
    barra_lateral(r)
    situacion, pronostico, comunas, riesgo = st.tabs(
        ["📊 Situación", "📈 Pronóstico", "🏘️ Comunas", "🩺 Riesgo de un caso"])
    with situacion:
        seccion_situacion(r)
    with pronostico:
        seccion_pronostico(r)
    with comunas:
        seccion_comunas(r)
    with riesgo:
        seccion_riesgo(r)
    st.markdown("<div class='pie'>Alerta Dengue Cali · Universidad Libre, seccional Cali · "
                "Datos abiertos del SIVIGILA, la Alcaldía de Cali y el IDEAM</div>",
                unsafe_allow_html=True)


if __name__ == "__main__":       # Streamlit ejecuta el archivo como __main__
    main()
