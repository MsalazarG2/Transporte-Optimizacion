import pulp
import pandas as pd


def optimizar_distribucion(capacidades, demandas, costos, limite_rionegro_central):
    """
    Optimiza la distribución de medicamentos de RedSalud.

    Parámetros:
        capacidades: diccionario con capacidades de cada centro.
        demandas: diccionario con demandas de cada hospital.
        costos: diccionario con costos por ruta.
        limite_rionegro_central: máximo permitido para Rionegro -> Central.

    Retorna:
        modelo, variables y resultados.
    """

    centros = list(capacidades.keys())
    hospitales = list(demandas.keys())

    # Crear modelo
    modelo = pulp.LpProblem(
        "RedSalud_Distribucion",
        pulp.LpMinimize
    )

    # Variables de decisión
    x = pulp.LpVariable.dicts(
        "x",
        [(i, j) for i in centros for j in hospitales],
        lowBound=0,
        cat="Continuous"
    )

    # --------------------------------------------------
    # FUNCIÓN OBJETIVO
    # --------------------------------------------------

    modelo += pulp.lpSum(
        costos[i][j] * x[(i, j)]
        for i in centros
        for j in hospitales
    )

    # --------------------------------------------------
    # RESTRICCIONES DE CAPACIDAD
    # --------------------------------------------------

    for i in centros:
        modelo += (
            pulp.lpSum(x[(i, j)] for j in hospitales)
            <= capacidades[i],
            f"Capacidad_{i}"
        )

    # --------------------------------------------------
    # RESTRICCIONES DE DEMANDA
    # --------------------------------------------------

    for j in hospitales:
        modelo += (
            pulp.lpSum(x[(i, j)] for i in centros)
            == demandas[j],
            f"Demanda_{j}"
        )

    # --------------------------------------------------
    # RESTRICCIÓN RIONEGRO -> HOSPITAL CENTRAL
    # --------------------------------------------------

    modelo += (
        x[("Rionegro", "Hospital Central")]
        <= limite_rionegro_central,
        "Limite_Rionegro_Central"
    )

    # --------------------------------------------------
    # RESOLVER
    # --------------------------------------------------

    modelo.solve(pulp.PULP_CBC_CMD(msg=False))

    estado = pulp.LpStatus[modelo.status]

    # --------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------

    matriz = []

    for i in centros:
        fila = {"Centro": i}

        for j in hospitales:
            fila[j] = pulp.value(x[(i, j)]) or 0

        fila["Total utilizado"] = sum(
            fila[j] for j in hospitales
        )

        fila["Capacidad"] = capacidades[i]

        fila["Capacidad no utilizada"] = (
            capacidades[i] - fila["Total utilizado"]
        )

        matriz.append(fila)

    resultados = pd.DataFrame(matriz)

    costo_total = (
        pulp.value(modelo.objective)
        if modelo.status == pulp.LpStatusOptimal
        else None
    )

    return modelo, x, resultados, costo_total, estado