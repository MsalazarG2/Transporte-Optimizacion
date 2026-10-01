import streamlit as st
import pandas as pd
import pulp


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

st.set_page_config(
    page_title="RedSalud - Optimización",
    page_icon="💊",
    layout="wide"
)

st.title("💊 RedSalud")
st.subheader("Optimización de la distribución de medicamentos")

st.write(
    """
    Esta aplicación permite determinar el plan de distribución
    de menor costo para RedSalud, teniendo en cuenta las
    capacidades de los centros, las demandas hospitalarias,
    los costos de transporte y la restricción especial
    de Rionegro hacia el Hospital Central.
    """
)


# ==========================================================
# DATOS INICIALES
# ==========================================================

centros = [
    "Medellín",
    "Rionegro",
    "Bello"
]

hospitales = [
    "Hospital Norte",
    "Hospital Central",
    "Hospital Oriente",
    "Hospital Sur"
]


# ==========================================================
# ENTRADA DE CAPACIDADES
# ==========================================================

st.header("1. Capacidades de los centros")

col1, col2, col3 = st.columns(3)

with col1:
    cap_medellin = st.number_input(
        "Capacidad Medellín",
        min_value=0.0,
        value=140.0,
        step=1.0
    )

with col2:
    cap_rionegro = st.number_input(
        "Capacidad Rionegro",
        min_value=0.0,
        value=120.0,
        step=1.0
    )

with col3:
    cap_bello = st.number_input(
        "Capacidad Bello",
        min_value=0.0,
        value=130.0,
        step=1.0
    )

capacidades = {
    "Medellín": cap_medellin,
    "Rionegro": cap_rionegro,
    "Bello": cap_bello
}


# ==========================================================
# ENTRADA DE DEMANDAS
# ==========================================================

st.header("2. Demandas de los hospitales")

col1, col2, col3, col4 = st.columns(4)

with col1:
    dem_norte = st.number_input(
        "Hospital Norte",
        min_value=0.0,
        value=90.0,
        step=1.0
    )

with col2:
    dem_central = st.number_input(
        "Hospital Central",
        min_value=0.0,
        value=100.0,
        step=1.0
    )

with col3:
    dem_oriente = st.number_input(
        "Hospital Oriente",
        min_value=0.0,
        value=80.0,
        step=1.0
    )

with col4:
    dem_sur = st.number_input(
        "Hospital Sur",
        min_value=0.0,
        value=100.0,
        step=1.0
    )

demandas = {
    "Hospital Norte": dem_norte,
    "Hospital Central": dem_central,
    "Hospital Oriente": dem_oriente,
    "Hospital Sur": dem_sur
}


# ==========================================================
# ENTRADA DE COSTOS
# ==========================================================

st.header("3. Costos unitarios de transporte")

st.write(
    "Modifique los costos de cualquier ruta según el escenario que desee analizar."
)

costos_df = pd.DataFrame(
    [
        [8.0, 12.0, 15.0, 11.0],
        [14.0, 10.0, 9.0, 16.0],
        [11.0, 13.0, 14.0, 8.0]
    ],
    index=centros,
    columns=hospitales
)

costos_editados = st.data_editor(
    costos_df,
    use_container_width=True,
    num_rows="fixed",
    min_value=0.0,
    step=1.0
)

costos = {
    centro: {
        hospital: float(costos_editados.loc[centro, hospital])
        for hospital in hospitales
    }
    for centro in centros
}


# ==========================================================
# RESTRICCIÓN ESPECIAL
# ==========================================================

st.header("4. Restricción especial")

limite_rc = st.number_input(
    "Máximo de unidades desde Rionegro hacia Hospital Central",
    min_value=0.0,
    value=30.0,
    step=1.0
)


# ==========================================================
# INDICADORES PREVIOS
# ==========================================================

capacidad_total = sum(capacidades.values())
demanda_total = sum(demandas.values())

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Capacidad total",
        f"{capacidad_total:,.0f}"
    )

with col2:
    st.metric(
        "Demanda total",
        f"{demanda_total:,.0f}"
    )


# ==========================================================
# VALIDACIÓN DE CAPACIDAD
# ==========================================================

if demanda_total > capacidad_total:

    st.error(
        f"""
        ⚠️ No existe capacidad suficiente para satisfacer
        completamente la demanda.

        Demanda total: {demanda_total:,.0f}

        Capacidad total: {capacidad_total:,.0f}

        Déficit: {demanda_total - capacidad_total:,.0f}
        unidades.
        """
    )


# ==========================================================
# BOTÓN DE OPTIMIZACIÓN
# ==========================================================

if st.button(
    "Optimizar distribución",
    type="primary",
    use_container_width=True
):

    # ------------------------------------------------------
    # CREAR MODELO
    # ------------------------------------------------------

    modelo = pulp.LpProblem(
        "RedSalud",
        pulp.LpMinimize
    )

    # Variables
    x = pulp.LpVariable.dicts(
        "x",
        [
            (i, j)
            for i in centros
            for j in hospitales
        ],
        lowBound=0,
        cat="Continuous"
    )

    # ------------------------------------------------------
    # FUNCIÓN OBJETIVO
    # ------------------------------------------------------

    modelo += pulp.lpSum(
        costos[i][j] * x[(i, j)]
        for i in centros
        for j in hospitales
    )

    # ------------------------------------------------------
    # CAPACIDADES
    # ------------------------------------------------------

    for i in centros:

        modelo += (
            pulp.lpSum(
                x[(i, j)]
                for j in hospitales
            )
            <= capacidades[i]
        )

    # ------------------------------------------------------
    # DEMANDAS
    # ------------------------------------------------------

    for j in hospitales:

        modelo += (
            pulp.lpSum(
                x[(i, j)]
                for i in centros
            )
            == demandas[j]
        )

    # ------------------------------------------------------
    # RESTRICCIÓN RIONEGRO - CENTRAL
    # ------------------------------------------------------

    modelo += (
        x[("Rionegro", "Hospital Central")]
        <= limite_rc
    )

    # ------------------------------------------------------
    # RESOLVER
    # ------------------------------------------------------

    modelo.solve(
        pulp.PULP_CBC_CMD(msg=False)
    )

    estado = pulp.LpStatus[modelo.status]

    # ------------------------------------------------------
    # MOSTRAR ESTADO
    # ------------------------------------------------------

    st.header("5. Resultado de la optimización")

    if estado == "Optimal":

        st.success(
            "✅ Solución óptima encontrada."
        )

    elif estado == "Infeasible":

        st.error(
            "❌ El modelo es infactible. "
            "No es posible satisfacer todas las demandas "
            "con las condiciones ingresadas."
        )

    else:

        st.warning(
            f"Estado reportado por el optimizador: {estado}"
        )

    # ------------------------------------------------------
    # RESULTADOS
    # ------------------------------------------------------

    if estado == "Optimal":

        costo_total = pulp.value(
            modelo.objective
        )

        total_utilizado = 0

        filas = []

        for i in centros:

            fila = {
                "Centro": i
            }

            utilizado = 0

            for j in hospitales:

                cantidad = pulp.value(
                    x[(i, j)]
                ) or 0

                fila[j] = cantidad

                utilizado += cantidad

            fila["Total utilizado"] = utilizado

            fila["Capacidad"] = capacidades[i]

            fila["Capacidad no utilizada"] = (
                capacidades[i] - utilizado
            )

            total_utilizado += utilizado

            filas.append(fila)

        resultados = pd.DataFrame(filas)

        # --------------------------------------------------
        # INDICADORES
        # --------------------------------------------------

        porcentaje_utilizacion = (
            total_utilizado / capacidad_total * 100
            if capacidad_total > 0
            else 0
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Costo mínimo total",
                f"{costo_total:,.2f}"
            )

        with col2:

            st.metric(
                "Capacidad utilizada",
                f"{total_utilizado:,.0f}"
            )

        with col3:

            st.metric(
                "Utilización",
                f"{porcentaje_utilizacion:.2f}%"
            )

        # --------------------------------------------------
        # MATRIZ DE DISTRIBUCIÓN
        # --------------------------------------------------

        st.subheader(
            "Matriz óptima de distribución"
        )

        matriz = resultados[
            [
                "Centro",
                "Hospital Norte",
                "Hospital Central",
                "Hospital Oriente",
                "Hospital Sur"
            ]
        ].copy()

        st.dataframe(
            matriz,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------
        # CAPACIDAD POR CENTRO
        # --------------------------------------------------

        st.subheader(
            "Utilización de los centros de distribución"
        )

        st.dataframe(
            resultados[
                [
                    "Centro",
                    "Total utilizado",
                    "Capacidad",
                    "Capacidad no utilizada"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------
        # DETALLE DE DEMANDA
        # --------------------------------------------------

        st.subheader(
            "Verificación de demanda"
        )

        for j in hospitales:

            total_hospital = sum(
                pulp.value(
                    x[(i, j)]
                ) or 0
                for i in centros
            )

            st.write(
                f"**{j}:** "
                f"{total_hospital:,.0f} / "
                f"{demandas[j]:,.0f} unidades"
            )

        # --------------------------------------------------
        # CAPACIDAD NO UTILIZADA
        # --------------------------------------------------

        st.info(
            f"""
            Capacidad total disponible:
            **{capacidad_total:,.0f} unidades**

            Demanda total:
            **{demanda_total:,.0f} unidades**

            Capacidad no utilizada:
            **{capacidad_total - total_utilizado:,.0f} unidades**
            """
        )