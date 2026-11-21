# Definición del problema de aprendizaje automático

**Proyecto:** Alerta Dengue Cali — Secretaría de Salud Pública Distrital de Santiago de Cali
**Autores:** Samuel Acevedo Mambuscay (2216646) y Luis Carlos Tez (2216702) — Universidad Libre, seccional Cali
**Cuaderno:** `notebooks/04_modelado.ipynb` · **Código:** `flows/05_ml_models.py`

## 1. Necesidad

La vigilancia del dengue en Cali es reactiva: el canal endémico se revisa con semanas ya
cerradas y las brigadas de control vectorial se programan cuando el aumento de casos ya es
evidente. El producto de datos busca anticipar: cuántos casos vienen en las próximas semanas, si
la ciudad va a entrar en brote, qué pacientes tienen más riesgo de agravarse y qué comunas
necesitan una intervención diferente.

## 2. Problemas y variables objetivo

| # | Pregunta | Tipo | Objetivo | Unidad |
|---|---|---|---|---|
| 1 | ¿Cuántos casos habrá en las semanas t+1 a t+4? | Regresión sobre una serie de tiempo | Casos de la semana t+h (se modela el cambio `log(1+casos[t+h]) − log(1+casos[t])`) | Semana epidemiológica de la ciudad |
| 2 | ¿La ciudad estará en brote? | Clasificación de cada semana | Pronóstico mayor que el cuartil 3 del canal endémico | Semana |
| 3 | ¿Este caso tendrá signos de alarma o será grave? | Clasificación binaria | `severo` = con signos de alarma o dengue grave | Caso notificado |
| 4 | ¿El paciente requerirá hospitalización? | Clasificación binaria | `hospitalizado` | Caso notificado |
| 5 | ¿Cómo se agrupan las comunas? | Agrupamiento (no supervisado) | — | Comuna urbana (22) |

## 3. Momento de la predicción y variables permitidas

La regla es usar solo lo que se conoce en el momento en que se toma la decisión.

- **Pronóstico:** se hace al cerrar la semana t. Variables: los casos de la semana t y de las 7 anteriores (como nivel y cambios), la posición en el año de la semana que se pronostica y, como candidata, la anomalía de temperatura de las últimas 4 semanas (IDEAM). No se usan semanas interpoladas ni como origen ni como objetivo, porque la interpolación usa datos posteriores.
- **Riesgo y hospitalización:** se predice en la **primera consulta**. Signos de alarma: edad, sexo, estrato, comuna, afiliación, migrante, desplazamiento en los últimos 15 días, días desde el inicio de síntomas y los síntomas generales (fiebre, cefalea, dolor retroocular, mialgias, artralgia, erupción), con los 13.966 casos del Distrito 2022-2023, los únicos que traen la clasificación clínica. Hospitalización: el dato está en todas las fuentes, así que se usan los 71.586 casos de 2020-2025 con las variables que traen el INS y el Distrito (edad, sexo, estrato, afiliación, días desde el inicio de síntomas y migrante).
- **Excluidas por fuga de datos:** los 15 signos de alarma y de gravedad (dolor abdominal, vómito, caída de plaquetas, choque, etc.) definen la clasificación: con ellos el AUC sube a 1,000 y, en entrenamiento, ninguno de los 6.700 casos sin signos de alarma es severo. También se excluyen la hospitalización, la conducta, el tipo de caso, la condición final y la clasificación, que ocurren después de la consulta.
- **Clima:** la temperatura se prueba como variable candidata. Se usa la anomalía por estación y sistema de medición, porque el IDEAM cambió de sistema en 2024 y la temperatura registrada bajó entre 1 y 4 °C sin que cambiara el clima.

## 4. Métricas y por qué

| Modelo | Métrica principal | Complementarias | Razón |
|---|---|---|---|
| Pronóstico | **MAE** en casos por semana | RMSE, MAPE, mejora frente al ingenuo, cobertura del intervalo del 80 % | El MAE se lee directo ("nos equivocamos en 46 casos por semana") y sirve para dimensionar camas e insumos. La comparación con el ingenuo dice si el modelo aporta algo. |
| Alerta de brote | **Sensibilidad** | Precisión, especificidad | Perder una semana de brote es más costoso que una falsa alarma. |
| Riesgo y hospitalización | **ROC-AUC** | Sensibilidad, especificidad, VPP, VPN, Brier, % de positivos por nivel, desempeño por subgrupo | El modelo se usa para **ordenar** la lista de seguimiento; el AUC mide justamente si ordena bien, sin depender de un umbral. |
| Comunas | **Coeficiente de silueta** | Inercia, tamaño mínimo de grupo | Mide qué tan separados quedan los grupos sin necesitar etiquetas. |

## 5. Costo de los errores

- **Pronóstico por debajo de lo real o alerta que no se activa (falso negativo):** las brigadas y las camas llegan tarde; es el error más grave porque una epidemia como la de 2024 crece en pocas semanas.
- **Falsa alarma:** se gastan insumos y horas de brigada; es un costo real pero recuperable.
- **Paciente severo clasificado como de riesgo bajo:** podría recibir menos seguimiento. Por eso el modelo **no reemplaza** la valoración clínica ni los criterios de la guía de atención del dengue: solo ayuda a priorizar llamadas o visitas de seguimiento.
- **Paciente sin riesgo clasificado como alto:** seguimiento innecesario; costo bajo.

## 6. Validación

- **Pronóstico:** partición temporal. Se entrena con 2020-2022, se eligen los modelos con 2023 y se prueba **una sola vez** con 2024-2025. La alerta se evalúa con un canal construido solo con años anteriores a la prueba (2021 y 2022). El modelo definitivo se reentrena con todas las semanas.
- **Clasificadores:** el 20 % de casos más recientes (notificados desde el 25 de octubre de 2023, n = 2.833) es la prueba. En el 80 % restante (n = 11.133) se comparan modelos con validación cruzada estratificada de 5 pliegues, se ajustan hiperparámetros con búsqueda en malla y se fija el umbral con predicciones fuera de muestra. La prueba se abre una sola vez; el modelo definitivo se reentrena con los 13.966 casos.
- **Agrupamiento:** se compara k = 2 a 6 con la silueta.
- Todo el preprocesamiento (imputación, escalado, codificación) está dentro de un `Pipeline` de scikit-learn ajustado solo con entrenamiento.

## 7. Criterio de éxito

| Modelo | Criterio fijado antes de la prueba | Resultado en la prueba | ¿Se cumple? |
|---|---|---|---|
| Pronóstico | MAE menor que el ingenuo en 2 a 4 semanas | 1,7 %, 4,7 % y 6,3 % menor (MAE 46, 60 y 72 casos) | Sí, por poco |
| Pronóstico | Intervalo del 80 % que cubra cerca del 80 % de las semanas | Cubre entre 61 % y 67 % | **No** (limitación) |
| Alerta | Sensibilidad ≥ 90 % | 96 % a 98 % | Sí |
| Riesgo | AUC mayor que 0,60 y mejor que la referencia (0,50) | 0,616 (IC 95 % 0,594-0,637) | Sí, con discriminación modesta |
| Hospitalización | AUC mayor que 0,60 | 0,647 (IC 95 % 0,627-0,669) | Sí |
| Comunas | Silueta mayor que 0,25 y grupos interpretables | 0,307 con k = 3 | Sí |

## 8. Selección de modelos

- **Pronóstico.** Candidatos: ingenuo, tendencia reciente amortiguada al 25 %, 50 % y 75 %, la misma tendencia con la anomalía de temperatura, Ridge, Random Forest, gradient boosting y un ensamble Random Forest + ingenuo. En la validación 2023 ningún modelo de aprendizaje automático supera al ingenuo (Random Forest es 27 % a 51 % peor; el ensamble, 11 % a 21 % peor): con una sola epidemia en el entrenamiento, aprenden que los casos tienden a bajar. Se elige, por horizonte, el de menor MAE; si otros quedan a menos del 1 % (empate), el más simple. Resultado: ingenuo a 1 semana, tendencia al 25 % a 2 y 3 semanas, tendencia al 50 % a 4 semanas. Las versiones con temperatura empatan con la tendencia sin clima (su coeficiente es prácticamente cero) y no se eligen; en la prueba habrían cambiado el error en menos de 1 caso por semana.
- **Riesgo.** Candidatos: referencia (prevalencia), regresión logística, Random Forest y gradient boosting. AUC en validación cruzada: 0,500, 0,591, 0,620 y 0,623. Random Forest y gradient boosting empatan dentro de una desviación estándar; se elige Random Forest (el propuesto en el proyecto y más fácil de explicar). La búsqueda en malla (18 combinaciones) eligió 400 árboles, profundidad 10, hojas de al menos 20 casos y 20 % de las variables por nodo (AUC 0,623).
- **Hospitalización.** Mismo diseño e hiperparámetros; AUC en validación cruzada 0,656 (Random Forest) frente a 0,608 (logística).
- **Comunas.** K-means con variables estandarizadas; k = 3 (silueta 0,307; k = 4 da 0,297 y un grupo de una sola comuna).

## 9. Limitaciones

1. El pronóstico mejora poco al ingenuo y su intervalo es demasiado estrecho para una epidemia grande: 2023, el año de validación, fue más tranquilo que 2024.
2. Solo hay seis años de serie y dos epidemias; los modelos de aprendizaje automático necesitan más historia (por ejemplo, los años 2017-2019 del INS). El clima público del IDEAM tiene la mitad de las semanas vacías y un cambio de sistema de medición en 2024, así que su efecto no se puede medir con confianza.
3. El modelo de riesgo solo se entrenó con casos del Distrito de 2022-2023: puede no representar una epidemia como la de 2024 ni la forma de clasificar de otras épocas.
4. La afiliación y la edad funcionan como atajos: el modelo marca casi siempre a los mayores de 65 años y al régimen subsidiado y casi nunca a los niños de 0 a 4 años (ver la ficha del modelo).
5. La hospitalización depende de la oferta de camas y del criterio de cada IPS, no solo de la gravedad.
6. Los datos son casos notificados: hay subregistro, sobre todo en los años de menos transmisión.

## 10. Reflexión ética

- **Uso como apoyo, no como decisión.** Un AUC de 0,62 significa que el modelo se equivoca con frecuencia en casos individuales. Nunca debe usarse para negar atención, dar de alta o decidir una hospitalización.
- **Equidad.** Que el régimen subsidiado tenga más riesgo estimado puede ayudar a priorizarlo, pero también puede reforzar un trato diferente basado en la afiliación. Hay que revisar el desempeño por subgrupo antes de cada actualización y no usar el nivel de riesgo para asignar recursos de forma que se reduzca la atención a ningún grupo.
- **Privacidad.** Se trabaja solo con datos anonimizados y publicados en portales abiertos, conforme a la Ley 1581 de 2012; el producto no guarda datos de pacientes (la API calcula el riesgo y no almacena el caso).
- **Transparencia.** Las métricas, los subgrupos y las limitaciones están en `reports/model_card.md`, y cualquier persona puede reproducir los resultados con `make pipeline`.
