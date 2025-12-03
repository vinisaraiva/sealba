# SEALBA – Painel de Sequestro de Carbono e Expansão Agropecuária
# Versão 2.0 (dados 2001–2023)
# ----------------------------------------------------------
# Requisitos:
#   pip install streamlit pandas plotly scikit-learn statsmodels
# Execução:
#   streamlit run sealba_painel.py
# ----------------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =======================
# Configuração geral
# =======================
st.set_page_config(
    page_title="SEALBA – Sequestro de Carbono e Expansão Agropecuária",
    layout="wide",
)

DATA_DIR = Path(__file__).parent

# Paletas de cores padronizadas
COLOR_CLASSES = {
    "Floresta": "#1b7837",
    "Agricultura": "#fdae61",
    "Vegetação Herbácea e Arbustiva": "#d9f0d3",
    "Pastagem": "#d9ef8b",
    "Área Não Vegetada": "#999999",
    "Corpo D'água": "#3288bd",
}

COLOR_CLIMA = {
    "Precipitação (mm)": "#2166ac",
    "ETo (mm)": "#67a9cf",
    "Temperatura média (°C)": "#ef8a62",
    "Umidade relativa (%)": "#fddbc7",
}

# =======================
# Funções utilitárias
# =======================

@st.cache_data
def load_series():
    """Séries anuais agregadas da região SEALBA."""
    df = pd.read_excel(DATA_DIR / "series_anuais.xlsx")
    df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce")
    df.loc[~np.isfinite(df["Ano"]), "Ano"] = np.nan
    df["Ano"] = df["Ano"].astype("Int64")
    return df


@st.cache_data
def load_clima():
    """Dados anuais por município (clima + NPP)."""
    #xlsx_path = DATA_DIR / "Dataset_clima_SeAlBa.xlsx"
    parquet_path = DATA_DIR / "Dataset_clima_SeAlBa.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_excel(xlsx_path)
        # Persistir versão otimizada para futuras execuções
        try:
            df.to_parquet(parquet_path, compression="zstd", index=False)
        except Exception:
            pass
    # Padronizar nomes e tipos
    df.rename(columns={"municipio": "Municipio"}, inplace=True)
    if "ano" in df.columns:
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
        df.loc[~np.isfinite(df["ano"]), "ano"] = np.nan
        df["ano"] = df["ano"].astype("Int64")
    return df


@st.cache_data
def load_uso_media():
    """Médias de uso do solo por município (agro, pasto, floresta)."""
    #xlsx_path = DATA_DIR / "media_agro_past_floresta.xlsx"
    parquet_path = DATA_DIR / "media_agro_past_floresta.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_excel(xlsx_path)
        df["Municipio"] = df["Municipio"].astype(str)
        try:
            df.to_parquet(parquet_path, compression="zstd", index=False)
        except Exception:
            pass
    return df


def small_card(label, value, suffix=""):
    st.markdown(
        f"""
        <div style="padding:0.4rem 0.8rem;border-radius:0.6rem;
                    background-color:#f7f7f9;border:1px solid #e0e0e0;
                    display:inline-block;margin-right:0.6rem;margin-bottom:0.4rem;">
            <span style="font-size:0.75rem;color:#888;">{label}</span><br>
            <span style="font-size:1.1rem;font-weight:600;color:#333;">
                {value}{suffix}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def corr_info(x, y, method="pearson"):
    """Retorna coeficiente de correlação e n (tratando NaN)."""
    s1 = x.astype(float)
    s2 = y.astype(float)
    mask = s1.notna() & s2.notna()
    if mask.sum() < 3:
        return np.nan, mask.sum()
    coef = s1[mask].corr(s2[mask], method=method)
    return coef, mask.sum()


def regressao_linear(x, y):
    """Ajusta regressão linear e retorna slope, intercept e R²."""
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    X = sm.add_constant(x[mask].astype(float))
    model = sm.OLS(y[mask].astype(float), X).fit()
    intercept = model.params["const"]
    slope = model.params[x.name]
    r2 = model.rsquared
    return slope, intercept, r2


# =======================
# Carregar dados
# =======================

series = load_series()
clima = load_clima()
uso_media = load_uso_media()

# =======================
# Sidebar – navegação + autoria
# =======================

#st.sidebar.title("SEALBA – Painel Interativo")
st.sidebar.markdown(
    "Explore a dinâmica de **uso da terra, clima e sequestro de carbono** "
    "na região do SeAlBa (Sergipe, Alagoas e Bahia)."
)

page = st.sidebar.radio(
    "Selecione a seção:",
    [
        "0. Início",
        "1. Uso e Cobertura da Terra",
        "2. Clima",
        "3. NPP e Sequestro de Carbono",
        "4. Análises Estatísticas",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dados: MapBiomas, MODIS/MOD17A3, BR-DWGD. "
    "Período principal: 2001–2023."
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "#### Autoria\n"
    "Vinícius Saraiva Santos  \n"
    "Talia Silva Ribeiro  \n"
    "Breno Arles da Silva Santos  \n"
    "Dian Júnio Bomfim Borges  \n"
    "Tatiane Neres dos Santos Sena  \n"
    "\n"
    "*Doutorandos do PPG Biossistemas – UFSB*"
)

# =======================
# Página 0 – Início
# =======================
if page.startswith("0"):
    st.markdown("## SEALBA – Dinâmica Climática, Uso da Terra e Sequestro de Carbono")

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown(
            "Este painel interativo reúne informações anuais sobre **clima**, "
            "**uso e cobertura da terra** e **produtividade da vegetação** "
            "(NPP) na região do SeAlBa – que abrange municípios de Sergipe, "
            "Alagoas e Bahia.\n\n"
            "Você pode navegar pelas abas para:\n"
            "- acompanhar a expansão agropecuária e a perda de vegetação natural;\n"
            "- visualizar anos de seca e de maior disponibilidade hídrica;\n"
            "- investigar como essas mudanças afetam o sequestro de carbono;\n"
            "- comparar municípios com perfis semelhantes."
        )

        st.markdown("---")
        st.markdown(
            "👈 Use o menu lateral para escolher a seção que deseja explorar."
        )

    with col2:
        mapa_path = DATA_DIR / "mapa_sealba.jpg"
        if mapa_path.exists():
            st.image(mapa_path, caption="Região SEALBA (Sergipe, Alagoas e Bahia)")
        else:
            st.info(
                "Insira um arquivo de mapa chamado **`mapa_sealba.png`** na mesma pasta "
                "para exibir aqui a localização da região SEALBA."
            )

    st.markdown("---")
    st.caption(
        "Este painel faz parte de um estudo sobre manejo e conservação na região semiárida "
        "do SeAlBa, integrando dados satelitais de uso da terra, clima e sequestro de carbono."
    )


# =======================
# Página 1 – Uso da Terra
# =======================
elif page.startswith("1"):
    st.markdown("## Séries Temporais de Uso e Cobertura da Terra (2001–2023)")

    col1, col2 = st.columns([2, 1.2])

    with col1:
        st.markdown(
            "Este gráfico mostra como a área ocupada por cada classe de uso da terra "
            "mudou ao longo dos anos na região do SeAlBa."
        )

        cols_area = [
            "soma_Floresta_km2",
            "soma_Agrop_km2",
            "soma_VegHerbArb_km2",
            "soma_Past_km2",
            "soma_NaoVeg_km2",
            "soma_Agua_km2",
        ]
        rename_map = {
            "soma_Floresta_km2": "Floresta",
            "soma_Agrop_km2": "Agricultura",
            "soma_VegHerbArb_km2": "Vegetação Herbácea e Arbustiva",
            "soma_Past_km2": "Pastagem",
            "soma_NaoVeg_km2": "Área Não Vegetada",
            "soma_Agua_km2": "Corpo D'água",
        }

        df_long = (
            series[["Ano"] + cols_area]
            .rename(columns=rename_map)
            .melt(id_vars="Ano", var_name="Classe de Uso", value_name="Área (km²)")
        )

        fig = px.line(
            df_long,
            x="Ano",
            y="Área (km²)",
            color="Classe de Uso",
            markers=True,
            color_discrete_map=COLOR_CLASSES,
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="v", x=1.02, y=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Linhas que sobem indicam expansão daquela classe; linhas que descem "
            "indicam redução de área ao longo do tempo."
        )

    with col2:
        st.markdown("#### Resumo recente (2021–2023)")
        recent = series[series["Ano"] >= 2021]
        for col, label in [
            ("soma_Agrop_km2", "Agricultura"),
            ("soma_Past_km2", "Pastagem"),
            ("soma_Floresta_km2", "Floresta"),
        ]:
            media = recent[col].mean()
            small_card(label, f"{media:,.0f}", " km²")

        st.markdown("---")
        st.markdown(
            "💬 **Em termos simples:** a agricultura vem aumentando, a pastagem ainda "
            "ocupa grande parte da área e a floresta mostra perda gradual de cobertura."
        )

    st.markdown("---")
    st.markdown("### Tendências de classes selecionadas")

    classe_sel = st.selectbox(
        "Selecione uma classe para visualizar a tendência linear:",
        ["Floresta", "Vegetação Herbácea e Arbustiva", "Agricultura", "Pastagem"],
    )

    if classe_sel == "Floresta":
        col_name = "soma_Floresta_km2"
        y_label = "Área Florestal (km²)"
    elif classe_sel == "Vegetação Herbácea e Arbustiva":
        col_name = "soma_VegHerbArb_km2"
        y_label = "Área Herbácea e Arbustiva (km²)"
    elif classe_sel == "Agricultura":
        col_name = "soma_Agrop_km2"
        y_label = "Área Agrícola (km²)"
    else:
        col_name = "soma_Past_km2"
        y_label = "Área de Pastagem (km²)"

    fig_trend = px.scatter(
        series,
        x="Ano",
        y=col_name,
        trendline="ols",
    )
    fig_trend.update_traces(mode="markers", marker=dict(size=9, opacity=0.8))
    fig_trend.update_layout(
        yaxis_title=y_label,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Estatísticas da tendência
    slope, intercept, r2 = regressao_linear(series["Ano"], series[col_name])
    if not np.isnan(r2):
        st.caption(
            f"A linha reta resume a tendência ao longo do período. "
            f"Coeficiente angular: **{slope:,.1f} km²/ano**; "
            f"R² da regressão: **{r2:.2f}**."
        )
    else:
        st.caption(
            "Não foi possível calcular a regressão linear (dados insuficientes)."
        )


# =======================
# Página 2 – Clima
# =======================
elif page.startswith("2"):
    st.markdown("## Dinâmica Climática na Região do SeAlBa")

    st.markdown(
        "As curvas abaixo mostram como variáveis climáticas médias (chuva, "
        "temperatura, evapotranspiração e umidade) se comportaram na região ao "
        "longo dos anos."
    )

    clima_reg = series[["Ano", "media_pr", "media_eto", "media_tmean", "media_rh"]]

    var_sel = st.multiselect(
        "Selecione variáveis climáticas para plotar:",
        ["Precipitação (mm)", "ETo (mm)", "Temperatura média (°C)", "Umidade relativa (%)"],
        default=["Precipitação (mm)", "Temperatura média (°C)"],
    )

    normalizar = st.checkbox(
        "Normalizar valores entre 0 e 1 (facilita a comparação entre variáveis)",
        value=False,
    )

    rename = {
        "media_pr": "Precipitação (mm)",
        "media_eto": "ETo (mm)",
        "media_tmean": "Temperatura média (°C)",
        "media_rh": "Umidade relativa (%)",
    }

    cols = [k for k, v in rename.items() if v in var_sel]
    df_plot = clima_reg[["Ano"] + cols].rename(columns=rename)

    if normalizar and cols:
        for col in df_plot.columns:
            if col == "Ano":
                continue
            vmin = df_plot[col].min()
            vmax = df_plot[col].max()
            if vmax > vmin:
                df_plot[col] = (df_plot[col] - vmin) / (vmax - vmin)
        y_label = "Valor normalizado (0–1)"
    else:
        y_label = "Valor"

    df_long = df_plot.melt(id_vars="Ano", var_name="Variável", value_name="Valor")

    if not df_long.empty:
        fig = px.line(
            df_long,
            x="Ano",
            y="Valor",
            color="Variável",
            markers=True,
            color_discrete_map=COLOR_CLIMA,
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title=y_label,
        )
        st.plotly_chart(fig, use_container_width=True)

        if normalizar:
            st.caption(
                "Com a normalização, todas as variáveis variam entre 0 e 1. "
                "Isso não altera os padrões ao longo do tempo, apenas coloca "
                "tudo na mesma escala para facilitar a comparação visual."
            )
        else:
            st.caption(
                "As variáveis estão em suas unidades originais. "
                "Precipitação e ETo possuem valores numéricos muito maiores que "
                "temperatura e umidade, por isso dominam o eixo vertical."
            )

    st.markdown("---")
    st.markdown(
        "💬 **Resumo em linguagem simples:** este painel ajuda a enxergar os anos "
        "de seca e de chuva mais abundante, que são o pano de fundo climático das "
        "variações de produtividade da vegetação."
    )


# =======================
# Página 3 – NPP e Sequestro de Carbono
# =======================
elif page.startswith("3"):
    st.markdown("## Sequestro de Carbono – Produtividade Primária Líquida (2001–2023)")

    col1, col2 = st.columns([2, 1.2])

    with col1:
        st.markdown(
            "A curva abaixo representa a quantidade média de carbono fixada pela "
            "vegetação da região em cada ano (NPP médio regional)."
        )
        fig_npp = px.line(
            series,
            x="Ano",
            y="soma_mean_NPP",
            markers=True,
        )
        fig_npp.update_traces(marker=dict(size=9))
        fig_npp.update_layout(
            yaxis_title="NPP médio regional (g C m⁻² ano⁻¹)",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_npp, use_container_width=True)

        st.caption(
            "Picos indicam anos em que a vegetação cresceu mais e fixou mais carbono; "
            "vales indicam anos de menor crescimento, muitas vezes associados a secas."
        )

    with col2:
        st.markdown("#### Destaques da série")
        npp_min = series.loc[series["soma_mean_NPP"].idxmin()]
        npp_max = series.loc[series["soma_mean_NPP"].idxmax()]

        small_card("Ano de menor NPP", int(npp_min["Ano"]))
        small_card("Ano de maior NPP", int(npp_max["Ano"]))
        small_card(
            "Amplitude",
            f"{npp_max['soma_mean_NPP'] - npp_min['soma_mean_NPP']:.0f}",
            " g C m⁻²",
        )

        st.markdown("---")
        st.markdown(
            "💬 **Em termos simples:** o NPP mostra o quanto a vegetação consegue "
            "‘puxar’ carbono da atmosfera em cada ano, funcionando como um termômetro "
            "da saúde da paisagem."
        )

    st.markdown("---")
    st.markdown("### NPP por classe de uso da terra")

    cols_npp = [
        "soma_Floresta_NPP",
        "soma_Agrop_NPP",
        "soma_VegHerbArb_NPP",
        "soma_Past_NPP",
    ]
    rename_npp = {
        "soma_Floresta_NPP": "Floresta",
        "soma_Agrop_NPP": "Agricultura",
        "soma_VegHerbArb_NPP": "Vegetação Herbácea e Arbustiva",
        "soma_Past_NPP": "Pastagem",
    }

    df_npp_long = (
        series[["Ano"] + cols_npp]
        .rename(columns=rename_npp)
        .melt(id_vars="Ano", var_name="Classe de Uso", value_name="NPP total")
    )

    fig_classes = px.line(
        df_npp_long,
        x="Ano",
        y="NPP total",
        color="Classe de Uso",
        markers=True,
        color_discrete_map=COLOR_CLASSES,
    )
    fig_classes.update_traces(marker=dict(size=9))
    fig_classes.update_layout(
        yaxis_title="NPP total da classe (g C m⁻² ano⁻¹)",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="v", x=1.02, y=1),
    )
    st.plotly_chart(fig_classes, use_container_width=True)

    st.caption(
        "Cada linha mostra quanto carbono é fixado por ano em cada tipo de uso da terra. "
        "Diferenças entre elas revelam quais classes contribuem mais para o sequestro total."
    )


# =======================
# Página 4 – Análises Estatísticas
# =======================
elif page.startswith("4"):
    st.markdown("## Análises Estatísticas")

    st.markdown(
        "Nesta seção é possível investigar, de forma simples, como clima, uso da terra "
        "e produtividade da vegetação se relacionam."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "4.1 – Correlações anuais (região)",
            "4.2 – Médias municipais",
            "4.3 – Clusterização de municípios",
        ]
    )

    # 4.1 – Correlações anuais
    with tab1:
        st.markdown("### Correlação entre clima, uso da terra e NPP (2001–2023)")

        st.markdown(
            "Cada ponto do gráfico representa um ano. A inclinação da nuvem de pontos "
            "indica se duas variáveis crescem ou diminuem juntas."
        )

        x_options = {
            "Precipitação média (mm)": "media_pr",
            "ETo média (mm)": "media_eto",
            "Temperatura média (°C)": "media_tmean",
            "Umidade relativa média (%)": "media_rh",
            "Área agrícola (km²)": "soma_Agrop_km2",
            "Área de pastagem (km²)": "soma_Past_km2",
            "Área florestal (km²)": "soma_Floresta_km2",
        }

        y_options = {
            "NPP médio regional": "soma_mean_NPP",
            "NPP total – Agricultura": "soma_Agrop_NPP",
            "NPP total – Pastagem": "soma_Past_NPP",
            "NPP total – Floresta": "soma_Floresta_NPP",
            "NPP total – Veg. Herbácea/Arbustiva": "soma_VegHerbArb_NPP",
        }

        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            x_label = st.selectbox("Variável em X", list(x_options.keys()))
        with col_sel2:
            y_label = st.selectbox("Variável em Y", list(y_options.keys()))

        metodo = st.radio(
            "Método de correlação:",
            ["Pearson (linear)", "Spearman (não paramétrico)"],
            horizontal=True,
        )
        method_internal = "pearson" if "Pearson" in metodo else "spearman"

        x_col = x_options[x_label]
        y_col = y_options[y_label]

        coef, n = corr_info(series[x_col], series[y_col], method=method_internal)

        df_corr = series[["Ano", x_col, y_col]].dropna()

        col_plot, col_stats = st.columns([2, 1])
        with col_plot:
            fig_scatter = px.scatter(
                df_corr,
                x=x_col,
                y=y_col,
                trendline="ols",
                labels={x_col: x_label, y_col: y_label},
            )
            fig_scatter.update_traces(marker=dict(size=10, opacity=0.7))
            fig_scatter.update_layout(margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.caption(
                "Se os pontos formam uma nuvem inclinada para cima, valores altos em X "
                "tendem a vir acompanhados de valores altos em Y. Inclinação para baixo "
                "indica relação inversa."
            )

            csv_corr = df_corr.rename(
                columns={x_col: x_label, y_col: y_label}
            ).to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Baixar dados desta correlação (CSV)",
                data=csv_corr,
                file_name="correlacao_anual_sealba.csv",
                mime="text/csv",
            )

        with col_stats:
            st.markdown("#### Estatísticas da correlação")
            if np.isnan(coef):
                st.write("Correlação não calculada (dados insuficientes).")
            else:
                small_card("n (anos)", n)
                small_card(f"Coeficiente ({method_internal})", f"{coef:.2f}")
                st.markdown("---")
                if coef > 0.5:
                    st.write("🔎 Correlação **positiva forte**.")
                elif coef > 0.3:
                    st.write("🔎 Correlação **positiva moderada**.")
                elif coef < -0.5:
                    st.write("🔎 Correlação **negativa forte**.")
                elif coef < -0.3:
                    st.write("🔎 Correlação **negativa moderada**.")
                else:
                    st.write("🔎 Correlação fraca ou inexistente.")

            st.caption(
                "Use esta aba para testar combinações como “área de agricultura × NPP” "
                "ou “chuva × NPP por classe de uso”."
            )

    # 4.2 – Médias municipais
    with tab2:
        st.markdown("### Correlação entre agropecuária média e NPP médio por município")

        st.markdown(
            "Aqui cada ponto representa um município, usando a média dos anos analisados."
        )

        clima_mun = (
            clima.groupby("Municipio", as_index=False)
            .agg(
                NPP_médio=("NPP", "mean"),
                PR_média=("PR", "mean"),
                Tmean_média=("Tmean", "mean"),
                ETo_média=("ETo", "mean"),
            )
        )

        mun_merged = clima_mun.merge(uso_media, on="Municipio", how="left")
        mun_merged["media_agro_pasto"] = (
            mun_merged["media_agro"].fillna(0) + mun_merged["media_past"].fillna(0)
        )

        col_a, col_b = st.columns(2)
        with col_a:
            x_choice = st.selectbox(
                "Variável de uso da terra (X):",
                [
                    "Agricultura média (km²)",
                    "Pastagem média (km²)",
                    "Agropecuária média (km²)",
                ],
            )
        with col_b:
            y_choice = "NPP médio (g C m⁻² ano⁻¹)"

        mapping_x = {
            "Agricultura média (km²)": "media_agro",
            "Pastagem média (km²)": "media_past",
            "Agropecuária média (km²)": "media_agro_pasto",
        }

        x_var = mapping_x[x_choice]
        y_var = "NPP_médio"

        coef_mun, n_mun = corr_info(
            mun_merged[x_var], mun_merged[y_var], method="spearman"
        )

        df_mun_plot = mun_merged[["Municipio", x_var, y_var]].dropna()

        fig_mun = px.scatter(
            df_mun_plot,
            x=x_var,
            y=y_var,
            hover_name="Municipio",
            labels={x_var: x_choice, y_var: y_choice},
        )
        fig_mun.update_traces(marker=dict(size=10, opacity=0.7))
        fig_mun.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_mun, use_container_width=True)

        st.caption(
            "Pontos mais à direita representam municípios com mais área agrícola ou de pasto; "
            "pontos mais altos representam municípios com maior produtividade média da vegetação."
        )

        csv_mun = df_mun_plot.rename(
            columns={x_var: x_choice, y_var: y_choice}
        ).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Baixar dados desta análise municipal (CSV)",
            data=csv_mun,
            file_name="correlacao_municipios_sealba.csv",
            mime="text/csv",
        )

        st.markdown("#### Estatísticas (Spearman)")
        if not np.isnan(coef_mun):
            small_card("n (municípios)", n_mun)
            small_card("Coeficiente (Spearman)", f"{coef_mun:.2f}")
        else:
            st.write("Correlação não calculada (dados insuficientes).")

        st.markdown(
            "💬 **Em linguagem simples:** esta análise mostra se municípios mais "
            "agropecuários tendem a ter mais ou menos sequestro médio de carbono."
        )

    # 4.3 – Clusterização
    with tab3:
        st.markdown("### Clusterização de municípios")

        st.markdown(
            "Nesta aba, municípios com comportamentos parecidos são agrupados em "
            "clusters, considerando clima, uso da terra e produtividade."
        )

        vars_cluster = [
            "NPP_médio",
            "PR_média",
            "Tmean_média",
            "ETo_média",
            "media_agro",
            "media_past",
            "media_floresta",
        ]

        df_cluster = mun_merged.dropna(subset=vars_cluster).copy()
        X = df_cluster[vars_cluster].values

        n_clusters = st.slider("Número de clusters (k):", 2, 6, 3)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_cluster["cluster"] = kmeans.fit_predict(X_scaled).astype(int)

        st.markdown("#### Distribuição dos municípios por cluster")
        counts = df_cluster["cluster"].value_counts().sort_index()
        for c, v in counts.items():
            small_card(f"Cluster {c}", v, " municípios")

        st.markdown("---")
        st.markdown("#### Mapa conceitual: uso da terra × NPP médio")

        fig_clu = px.scatter(
            df_cluster,
            x="media_agro",
            y="NPP_médio",
            color="cluster",
            hover_name="Municipio",
            labels={
                "media_agro": "Agricultura média (km²)",
                "NPP_médio": "NPP médio (g C m⁻² ano⁻¹)",
            },
        )
        fig_clu.update_traces(marker=dict(size=10, opacity=0.8))
        fig_clu.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_clu, use_container_width=True)

        st.caption(
            "Cada cor representa um tipo de município. Isso ajuda a identificar, por exemplo, "
            "grupos com muita agricultura e menor NPP, ou com mais floresta e maior NPP."
        )
