# SEALBA Panel – Streamlit (evento científico, sem upload) – v2.3
# Foco: clareza científica, legendas/ajuda, correlação com resumo e nível de análise
# ----------------------------------------------------
# Uso:
#   streamlit run SEALBA_panel_app.py
# Requisitos:
#   pip install -r requirements_sealba_panel.txt
# ----------------------------------------------------

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =======================
# Config & constantes
# =======================
st.set_page_config(page_title="SEALBA – Painel Socioambiental", layout="wide")
DATASET_PATH = Path(__file__).parent / "sealba_dataset.xlsx"
# IMPORTANTE: o arquivo acima deve conter a planilha "master_municipio_ano"
# com colunas (exemplos): ano, uf, municipio, pib_total_mil_reais, pib_percapita_reais,
# vab_agropecuaria_mil, precip_media_mm (ou pr_mean), idh_total, tmean, ur_mean, evt_mean etc.

AUTHORS = ["Dian Júnio B. Borges", "Talia S. Ribeiro", "Breno A. S. Santos", "Tatiane N. S. Sena", "Vinicius S. Santos"]
SEALBA_UFS = {"AL", "BA", "SE"}    # apenas estados participantes da SEALBA
CORR_STRONG_THR = 0.5              # |r| destacado no heatmap de correlação

# Detectar statsmodels (necessário para trendlines OLS/LOWESS no plotly express)
has_statsmodels = importlib.util.find_spec("statsmodels") is not None


# =======================
# Funções auxiliares
# =======================
@st.cache_data(show_spinner=False)
def load_master(path: Path) -> pd.DataFrame:
    """
    Lê o arquivo mestre e tenta garantir tipos adequados de colunas.
    Espera encontrar a sheet "master_municipio_ano".
    """
    df = pd.read_excel(path, sheet_name="master_municipio_ano")
    df.columns = [c.strip() for c in df.columns]
    # ano como inteiro (nullable)
    if "ano" in df.columns:
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")

    # converter numérico quando possível (exceto chaves categóricas)
    for c in df.columns:
        if df[c].dtype == object and c not in ["uf", "municipio"]:
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def numeric_columns(df: pd.DataFrame) -> list:
    """Lista de colunas numéricas (exclui 'ano' para evitar eixos com período sendo tratado como métrica)."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "ano"]


def fmt_number(val):
    """Formatação amigável de números em métricas."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if isinstance(val, (int, np.integer)):
        return f"{val:,}".replace(",", ".")
    try:
        return f"{val:,.0f}".replace(",", ".")
    except Exception:
        return str(val)


def analysis_level_label(df_filtered: pd.DataFrame) -> str:
    """
    Retorna um texto dizendo o nível da análise baseado nas UFs presentes no recorte filtrado.
    """
    if "uf" in df_filtered.columns:
        ufs = df_filtered["uf"].dropna().unique().tolist()
    else:
        ufs = []
    if len(ufs) == 0:
        return "Nível: regional (SEALBA)"
    elif set(ufs).issubset(SEALBA_UFS) and len(ufs) == len(SEALBA_UFS):
        return "Nível: regional (SEALBA – AL, BA e SE)"
    elif len(ufs) == 1:
        return f"Nível: intraestadual ({ufs[0]})"
    else:
        return "Nível: multiestados (AL/BA/SE)"


def corr_summary_tables(df_sub: pd.DataFrame, method: str = "spearman"):
    """
    Calcula a matriz de correlação e retorna:
    - corr: DataFrame de correlação
    - cm_sorted: pares ordenados por |r| desc
    - top_pos: top 10 r positivos
    - top_neg: top 10 r negativos
    """
    if df_sub.empty or df_sub.shape[1] < 2:
        return None, None, None, None

    corr = df_sub.corr(method=method)
    # Transformar em pares (var1, var2, r), removendo diagonal e duplicatas
    melted = (
        corr.reset_index()
        .melt(id_vars="index", var_name="var2", value_name="r")
        .rename(columns={"index": "var1"})
    )
    melted = melted[melted["var1"] < melted["var2"]]
    melted["abs_r"] = melted["r"].abs()

    cm_sorted = melted.sort_values("abs_r", ascending=False)
    top_pos = melted.sort_values("r", ascending=False).head(10)
    top_neg = melted.sort_values("r", ascending=True).head(10)
    return corr, cm_sorted, top_pos, top_neg


# =======================
# Carregar dados
# =======================
if not DATASET_PATH.exists():
    st.error(
        "Dataset padrão não encontrado ao lado do app: **SEALBA_dataset_master.xlsx**.\n\n"
        "Coloque o arquivo na mesma pasta do `SEALBA_panel_app.py` e garanta que exista a aba "
        "`master_municipio_ano` com as colunas necessárias."
    )
    st.stop()

df = load_master(DATASET_PATH)

# Restringe apenas a SEALBA
if "uf" in df.columns:
    df = df[df["uf"].isin(SEALBA_UFS)].copy()

# Título
st.title("SEALBA – Painel Socioambiental")
st.caption(
    "Protótipo interativo para análise de dados socioambientais e econômicos (2014–2023). "
    "Estados: AL, BA e SE. Sem upload — dados do artigo embutidos."
)

# =======================
# Sidebar – Filtros
# =======================
st.sidebar.header("Filtros")
ufs_disponiveis = (
    sorted([u for u in df["uf"].dropna().unique().tolist() if u in SEALBA_UFS])
    if "uf" in df.columns
    else []
)
uf_sel = st.sidebar.multiselect("UF (apenas SEALBA)", options=ufs_disponiveis, default=ufs_disponiveis)

df1 = df[df["uf"].isin(uf_sel)] if uf_sel else df.copy()

munis = sorted(df1["municipio"].dropna().unique().tolist()) if "municipio" in df1.columns else []
muni_sel = st.sidebar.multiselect("Municípios (opcional)", options=munis, default=[])
if muni_sel:
    df1 = df1[df1["municipio"].isin(muni_sel)]

if "ano" in df1.columns and df1["ano"].notna().any():
    min_y = int(df1["ano"].min())
    max_y = int(df1["ano"].max())
    per = st.sidebar.slider("Período (ano)", min_value=min_y, max_value=max_y, value=(min_y, max_y))
    df1 = df1[(df1["ano"] >= per[0]) & (df1["ano"] <= per[1])]

# =======================
# KPIs (topo) + status
# =======================
with st.status("Visão geral dos dados filtrados", state="complete"):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Municípios", df1["municipio"].nunique())
    with c2:
        st.metric(
            "Período",
            f"{int(df1['ano'].min())}–{int(df1['ano'].max())}" if df1["ano"].notna().any() else "—",
        )
    with c3:
        val = df1["pib_total_mil_reais"].mean() if "pib_total_mil_reais" in df1.columns else None
        st.metric("PIB total (média, R$ mil)", fmt_number(val))
    with c4:
        val = df1["pib_percapita_reais"].mean() if "pib_percapita_reais" in df1.columns else None
        st.metric("PIB per capita (média, R$)", fmt_number(val))
    with c5:
        # caso a coluna específica de precip municipal não exista, tenta pr_mean como fallback
        col_prec = "precip_media_mm" if "precip_media_mm" in df1.columns else ("pr_mean" if "pr_mean" in df1.columns else None)
        val = df1[col_prec].mean() if col_prec else None
        st.metric("Precipitação média municipal (mm)", fmt_number(val))
    st.write(analysis_level_label(df1))

st.markdown("---")

# =======================
# Abas
# =======================
tab_dash, tab_ts, tab_scatter, tab_corr, tab_rank, tab_autores = st.tabs(
    ["📊 Painel", "📈 Série Temporal", "🔬 Dispersão / Tendência", "🔗 Correlação (heatmap)", "🏆 Rankings", "👥 Autores"]
)

# -----------------------
# Aba: Painel (overview)
# -----------------------
with tab_dash:
    st.subheader("Painel – visão geral")
    st.caption("Cada ponto nos gráficos de dispersão representa **município×ano**; séries temporais podem ser médias regionais.")
    a1, a2 = st.columns(2)

    # Série temporal rápida (PIB total médio)
    with a1:
        if {"ano", "pib_total_mil_reais"}.issubset(df1.columns):
            ser = df1.groupby("ano", as_index=False)["pib_total_mil_reais"].mean(numeric_only=True)
            fig = px.line(ser, x="ano", y="pib_total_mil_reais", markers=True)
            fig.update_layout(
                height=320, margin=dict(l=10, r=10, t=30, b=10),
                yaxis_title="PIB total (média, R$ mil)"
            )
            st.plotly_chart(fig, use_container_width=True)
            with st.status("📌 Interpretação", state="complete"):
                st.write("**Média SEALBA por ano**. Use os filtros de UF/municípios para mudar a composição desta média.")
        else:
            st.info("Sem dados para série temporal de PIB total.")

    # Dispersão rápida: VAB agro vs precip municipal
    with a2:
        if {"vab_agropecuaria_mil"}.issubset(df1.columns) and ("precip_media_mm" in df1.columns or "pr_mean" in df1.columns):
            xcol = "precip_media_mm" if "precip_media_m m" in df1.columns else "pr_mean"
            # corrigir typo se ocorrer
            if "precip_media_m m" in df1.columns:
                df1 = df1.rename(columns={"precip_media_m m": "precip_media_mm"})
                xcol = "precip_media_mm"
            xcol = "precip_media_mm" if "precip_media_mm" in df1.columns else "pr_mean"

            d2 = df1[[xcol, "vab_agropecuaria_mil", "uf", "municipio", "ano"]].dropna()
            trendline = None
            if has_statsmodels:
                trendline = "ols"  # pode alternar para "lowess"

            fig2 = px.scatter(
                d2, x=xcol, y="vab_agropecuaria_mil",
                color="uf", hover_data=["municipio", "ano"], trendline=trendline
            )
            if not has_statsmodels:
                fig2.update_layout(title="(Instale 'statsmodels' para exibir linha de tendência OLS/LOWESS)")

            fig2.update_layout(
                height=320, margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title=("Precipitação média municipal (mm)" if xcol == "precip_media_mm" else "Precipitação regional (mm)"),
                yaxis_title="VAB Agro (R$ mil)"
            )
            st.plotly_chart(fig2, use_container_width=True)
            with st.status("📌 Leitura do gráfico", state="complete"):
                st.write("**Cada ponto = município×ano**; cores distinguem **UFs**. A linha (se habilitada) resume a tendência OLS/LOWESS.")
        else:
            st.info("Sem dados suficientes para dispersão (VAB agro × precipitação).")

    st.markdown("---")
    b1, b2 = st.columns(2)

    # Série temporal regional de precip (pr_mean)
    with b1:
        if {"ano", "pr_mean"}.issubset(df1.columns):
            serp = df1.groupby("ano", as_index=False)["pr_mean"].mean(numeric_only=True)
            fig3 = px.line(serp, x="ano", y="pr_mean", markers=True)
            fig3.update_layout(
                height=280, margin=dict(l=10, r=10, t=30, b=10),
                yaxis_title="Precipitação regional (mm)"
            )
            st.plotly_chart(fig3, use_container_width=True)
            with st.status("📌 Nota metodológica", state="complete"):
                st.write("**Média anual regional (SEALBA)** com base nos dados filtrados.")
        else:
            st.info("Sem dados para série de precipitação regional (pr_mean).")

    # Boxplot por UF – PIB per capita (último ano do filtro)
    with b2:
        if {"uf", "ano", "pib_percapita_reais"}.issubset(df1.columns) and df1["ano"].notna().any():
            last_y = int(df1["ano"].max())
            bx = df1[df1["ano"] == last_y][["uf", "pib_percapita_reais"]].dropna()
            if not bx.empty:
                fig4 = px.box(bx, x="uf", y="pib_percapita_reais", points="suspectedoutliers")
                fig4.update_layout(
                    height=280, margin=dict(l=10, r=10, t=30, b=10),
                    yaxis_title=f"PIB per capita (R$) — {last_y}"
                )
                st.plotly_chart(fig4, use_container_width=True)
                with st.status("📌 Leitura do boxplot", state="complete"):
                    st.write("**Comparação intra-SEALBA por UF** no último ano filtrado. Pontos fora do box podem indicar outliers.")
            else:
                st.info("Sem dados para boxplot de PIB per capita no ano selecionado.")
        else:
            st.info("Sem dados para boxplot de PIB per capita.")

# -----------------------
# Aba: Série temporal
# -----------------------
with tab_ts:
    st.subheader("Série temporal agregada")
    st.caption("Escolha **SEALBA (média)** para visão regional ou **Por UF** para série desagregada por estado.")
    if "ano" not in df1.columns:
        st.info("Sem coluna 'ano' para série temporal.")
    else:
        candidates = numeric_columns(df1)
        prefer = ["pib_total_mil_reais", "pib_percapita_reais", "vab_agropecuaria_mil", "pr_mean", "precip_media_mm", "idh_total"]
        defaults = [c for c in prefer if c in candidates] or (candidates[:1] if candidates else [])
        var_ts = st.selectbox("Variável", options=candidates, index=(candidates.index(defaults[0]) if defaults else 0))
        by = st.radio("Agregação", ["SEALBA (média)", "Por UF"], horizontal=True)
        if by == "SEALBA (média)":
            ser = df1.groupby("ano", as_index=False)[var_ts].mean(numeric_only=True)
            fig = px.line(ser, x="ano", y=var_ts, markers=True)
        else:
            ser = df1.groupby(["ano", "uf"], as_index=False)[var_ts].mean(numeric_only=True)
            fig = px.line(ser, x="ano", y=var_ts, color="uf", markers=True)

        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis_title=var_ts)
        st.plotly_chart(fig, use_container_width=True)
        with st.status("📌 Interpretação", state="complete"):
            st.write(analysis_level_label(df1))

# -----------------------
# Aba: Dispersão / Tendência
# -----------------------
with tab_scatter:
    st.subheader("Dispersão e linha de tendência")
    st.caption("Selecione pares de variáveis. **Cada ponto = município×ano**. Cor pode ser UF, município ou ano.")
    candidates = numeric_columns(df1)
    if len(candidates) < 2:
        st.info("Selecione um recorte com ao menos duas variáveis numéricas.")
    else:
        # sugestões padrão
        x_default = "precip_media_mm" if "precip_media_mm" in candidates else ( "pr_mean" if "pr_mean" in candidates else candidates[0] )
        y_default = "vab_agropecuaria_mil" if "vab_agropecuaria_mil" in candidates else candidates[min(1, len(candidates)-1)]

        xvar = st.selectbox("Eixo X", options=candidates, index=candidates.index(x_default))
        yvar = st.selectbox("Eixo Y", options=candidates, index=candidates.index(y_default))
        color_by = st.selectbox("Colorir por", options=[c for c in ["uf", "municipio", "ano"] if c in df1.columns], index=0)

        trend_opt = ["Sem linha"]
        if has_statsmodels:
            trend_opt += ["OLS (linear)", "LOWESS (suavizada)"]
        model = st.radio("Tendência", trend_opt, horizontal=True, index=(1 if has_statsmodels else 0))

        trend = None
        if model.startswith("OLS") and has_statsmodels:
            trend = "ols"
        elif model.startswith("LOWESS") and has_statsmodels:
            trend = "lowess"

        # Monta colunas únicas (narwhals/plotly requer nomes únicos)
        cols = [xvar, yvar, color_by, "municipio", "uf", "ano"]
        cols_unique = list(dict.fromkeys([c for c in cols if c in df1.columns]))
        d = df1[cols_unique].dropna()

        if d.empty:
            st.info("Sem dados suficientes para esse par de variáveis.")
        else:
            hover_base = [c for c in ["municipio", "uf", "ano"] if c in d.columns and c != color_by]
            fig = px.scatter(
                d, x=xvar, y=yvar,
                color=(color_by if color_by in d.columns else None),
                hover_data=hover_base,
                trendline=trend
            )
            if (model != "Sem linha") and not has_statsmodels:
                fig.update_layout(title="(Instale 'statsmodels' para habilitar linhas de tendência OLS/LOWESS)")
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), xaxis_title=xvar, yaxis_title=yvar)
            st.plotly_chart(fig, use_container_width=True)
            with st.status("📌 Dica de leitura", state="complete"):
                st.write("Use a cor por **UF** para diferenças espaciais ou por **ano** para entender mudança temporal.")

# -----------------------
# Aba: Correlação (heatmap)
# -----------------------
with tab_corr:
    st.subheader("Matriz de correlação")
    st.caption("Escolha as variáveis. **Pearson** (linear) ou **Spearman** (monotônica). Destaque para |r| ≥ 0.5.")
    candidates = numeric_columns(df1)
    prefer = ["pib_total_mil_reais", "pib_percapita_reais", "vab_agropecuaria_mil", "precip_media_mm", "pr_mean", "tmean", "evt_mean", "ur_mean", "idh_total"]
    default_vars = [c for c in prefer if c in candidates] or candidates[:6]
    cols_sel = st.multiselect("Variáveis", candidates, default=default_vars)
    method = st.radio("Método", ["spearman", "pearson"], index=0, horizontal=True)

    if len(cols_sel) >= 2:
        sub = df1[cols_sel].dropna()
        n_obs = len(sub)
        if not sub.empty:
            corr = sub.corr(method=method)

            # Heatmap com anotações (negrito quando |r| >= limiar)
            import plotly.figure_factory as ff
            z = corr.values
            x = corr.columns.tolist()
            y = corr.index.tolist()
            ann = np.vectorize(lambda v: f"**{v:.2f}**" if abs(v) >= CORR_STRONG_THR else f"{v:.2f}")(z)

            fig = ff.create_annotated_heatmap(
                z=z, x=x, y=y, colorscale="RdBu", showscale=True, reversescale=True, zmin=-1, zmax=1,
                annotation_text=ann
            )
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Resumo lateral (top correlações)
            _, cm_sorted, top_pos, top_neg = corr_summary_tables(sub, method=method)
            cL, cR = st.columns(2)
            with cL:
                st.markdown("**Top correlações positivas (r)**")
                if top_pos is not None and not top_pos.empty:
                    st.dataframe(top_pos[["var1", "var2", "r"]].reset_index(drop=True))
            with cR:
                st.markdown("**Top correlações negativas (r)**")
                if top_neg is not None and not top_neg.empty:
                    st.dataframe(top_neg[["var1", "var2", "r"]].reset_index(drop=True))

            with st.status("📌 Nota metodológica", state="complete"):
                st.write(f"{analysis_level_label(df1)} · Método: **{method}** · Observações válidas: **n = {n_obs}** · Destaque: **|r| ≥ {CORR_STRONG_THR}**.")
        else:
            st.info("Sem dados após remoção de NAs para as variáveis selecionadas.")
    else:
        st.info("Selecione ao menos duas variáveis para a correlação.")

# -----------------------
# Aba: Rankings
# -----------------------
with tab_rank:
    st.subheader("Rankings por ano (nível municipal)")
    st.caption("Ordena **municípios** por variável selecionada, para o ano escolhido.")
    candidates = numeric_columns(df1)
    prefer = ["pib_percapita_reais", "pib_total_mil_reais", "vab_agropecuaria_mil", "precip_media_mm", "idh_total"]
    defaults = [c for c in prefer if c in candidates] or (candidates[:1] if candidates else [])

    if not candidates:
        st.info("Não há variáveis numéricas disponíveis para ranking.")
    else:
        var = st.selectbox("Variável para ranking", options=candidates, index=candidates.index(defaults[0]))
        if "ano" in df1.columns and df1["ano"].notna().any():
            years = sorted(df1["ano"].dropna().unique().tolist())
            ysel = st.selectbox("Ano", options=years, index=len(years) - 1)
            dd = df1[df1["ano"] == ysel][["municipio", "uf", var]].dropna().copy()
            if dd.empty:
                st.info("Sem dados para esse ano/variável.")
            else:
                dd["municipio_uf"] = dd["municipio"] + " / " + dd["uf"]
                topn = st.slider("Top N", min_value=5, max_value=30, value=10, step=1)
                dd_top = dd.sort_values(var, ascending=False).head(topn)
                dd_bot = dd.sort_values(var, ascending=True).head(topn)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**TOP {topn} — {var}**")
                    if not dd_top.empty:
                        figt = px.bar(dd_top[::-1], x=var, y="municipio_uf", orientation="h")
                        figt.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="", xaxis_title=var)
                        st.plotly_chart(figt, use_container_width=True)
                with c2:
                    st.markdown(f"**BOTTOM {topn} — {var}**")
                    if not dd_bot.empty:
                        figb = px.bar(dd_bot[::-1], x=var, y="municipio_uf", orientation="h")
                        figb.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="", xaxis_title=var)
                        st.plotly_chart(figb, use_container_width=True)

                with st.status("📌 Interpretação", state="complete"):
                    st.write("Ranking **municipal**. Use os filtros de UF/município para restringir o universo analisado.")
        else:
            st.info("Sem coluna 'ano' para selecionar ranking anual.")

# -----------------------
# Aba: Autores
# -----------------------
with tab_autores:
    st.subheader("Autores do artigo")
    st.markdown("A autoria desta pesquisa é composta por:")
    for i, a in enumerate(AUTHORS, start=1):
        st.markdown(f"- {i}. **{a}**")

st.markdown("---")
st.caption(
    "Fonte: IBGE (PIB Municipal), Embrapa/IBGE (municípios SEALBA), arquivos do projeto (uso/cobertura, clima). "
    "Estados: AL, BA e SE. Destaques em correlação: |r| ≥ 0.5."
)
