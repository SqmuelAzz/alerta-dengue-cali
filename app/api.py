"""API REST de Alerta Dengue Cali (Flask).

Rutas:
    GET  /salud                  estado del servicio y versión de los modelos
    GET  /situacion              última semana frente al canal endémico
    GET  /pronostico             casos esperados en las próximas 4 semanas con semáforo
    POST /pronostico             igual, agregando antes los casos de semanas recientes
                                 {"casos_recientes": [120, 135]}
    POST /riesgo                 riesgo de signos de alarma o dengue grave de un caso
    GET  /comunas                indicadores y perfil (grupo K-means) de las 22 comunas
    GET  /comunas/<numero>       una comuna
    GET  /clima?anio=2024        casos, lluvia y temperatura semanales del IDEAM de un año
                                 y la relación del clima previo con el crecimiento de casos

Ejecución: make api  (o python -m app.api) y luego http://127.0.0.1:5000/salud
"""
from __future__ import annotations

import os

from flask import Flask, jsonify, request

from app import servicio


def create_app(recursos: servicio.Recursos | None = None) -> Flask:
    """Crea la aplicación; los recursos se cargan una sola vez al iniciar."""
    app = Flask(__name__)
    app.json.ensure_ascii = False
    app.json.sort_keys = False
    recursos = recursos or servicio.cargar_recursos()

    @app.errorhandler(servicio.EntradaInvalida)
    def entrada_invalida(error):
        return jsonify({"error": str(error)}), 400

    @app.errorhandler(404)
    def no_encontrado(_error):
        return jsonify({"error": "Ruta o recurso no encontrado"}), 404

    @app.get("/salud")
    def salud():
        meta = recursos.metadatos
        return jsonify({
            "estado": "ok",
            "modelos": {nombre: {"version": m.get("name"), "entrenado": m.get("saved_at")}
                        for nombre, m in meta.items()},
            "ultima_semana_datos": meta.get("pronostico", {}).get("ultima_semana_entrenamiento"),
        })

    @app.get("/situacion")
    def situacion():
        return jsonify(servicio.ultima_semana(recursos))

    @app.route("/pronostico", methods=["GET", "POST"])
    def pronostico():
        casos = None
        if request.method == "POST":
            cuerpo = request.get_json(silent=True)
            if not isinstance(cuerpo, dict):
                raise servicio.EntradaInvalida("Envíe un JSON como {\"casos_recientes\": [..]}")
            casos = cuerpo.get("casos_recientes")
        tabla = servicio.pronosticar(recursos, casos)
        columnas = ["horizonte", "anio_epi", "semana_epi", "fecha_inicio", "pronostico",
                    "limite_inferior", "limite_superior", "q3", "zona_pronostico", "alerta"]
        redondeo = dict.fromkeys(["pronostico", "limite_inferior", "limite_superior", "q3"], 1)
        return jsonify({"nivel_intervalo": recursos.pronostico["nivel_intervalo"],
                        "semanas": servicio.registros(tabla[columnas].round(redondeo))})

    @app.post("/riesgo")
    def riesgo():
        return jsonify(servicio.evaluar_caso(recursos, request.get_json(silent=True)))

    @app.get("/clima")
    def clima():
        anios = sorted(int(a) for a in recursos.serie["anio_epi"].unique())
        anio = request.args.get("anio", default=anios[-1], type=int)
        if anio not in anios:
            return jsonify({"error": f"Año sin datos: use uno entre {anios[0]} y {anios[-1]}"}), 404
        tabla, resumen = servicio.clima_anio(recursos, anio)
        return jsonify({**resumen, "relacion_con_casos": servicio.relacion_clima(recursos),
                        "detalle_semanal": servicio.registros(tabla)})

    @app.get("/comunas")
    def comunas():
        return jsonify(servicio.registros(servicio.tabla_comunas(recursos).round(2)))

    @app.get("/comunas/<int:numero>")
    def una_comuna(numero: int):
        try:
            return jsonify(servicio.comuna(recursos, numero))
        except KeyError:
            return jsonify({"error": f"La comuna {numero} no existe (1 a 22)"}), 404

    return app


if __name__ == "__main__":
    create_app().run(host=os.getenv("API_HOST", "127.0.0.1"),
                     port=int(os.getenv("API_PORT", "5000")))
