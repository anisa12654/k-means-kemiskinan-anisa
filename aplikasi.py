import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px
import json

# ==============================
# KONFIGURASI HALAMAN
# ==============================

st.set_page_config(
    page_title="Analisis K-Means Kemiskinan",
    layout="wide"
)

# ==============================
# HEADER
# ==============================

col1,col2,col3,col4 = st.columns([0.3,0.3,0.3,4])

with col1:
    st.image("Logo-Darmajaya-baru.png",width=65)

with col2:
    st.image("SI.png",width=65)

with col3:
    st.image("logo_unggul_SI.png",width=65)

with col4:
    st.markdown("### ANALISIS KLASTERISASI DAN FLUKTUASI ANGKA KEMISKINAN DI INDONESIA")
    st.markdown("###### METODE K-MEANS CLUSTERING - SISTEM INFORMASI IIB DARMAJAYA")
    st.info("Dashboard Visualisasi Data Kemiskinan")

st.markdown("---")

# ==============================
# LOAD DATA
# ==============================

@st.cache_data
def load_data():

    df = pd.read_excel("hasil_cluster_3.xlsx")

    # bersihkan nama kolom
    df.columns = df.columns.str.lower().str.strip()

    # hapus kolom kosong
    df = df.loc[:, ~df.columns.str.contains("unnamed")]

    # rename agar mudah dipakai
    df = df.rename(columns={
        "gk (garis kemiskinan)":"gk"
    })

    return df

df = load_data()

# ==============================
# SIDEBAR
# ==============================

with st.sidebar:

    st.header("Pengaturan")

    k = st.slider(
        "Jumlah Cluster (K)",
        2,6,3
    )

# ==============================
# TAB
# ==============================

tab1,tab2,tab3 = st.tabs([
    "📊 Dashboard",
    "📈 Fluktuasi Kemiskinan",
    "👤 Biodata"
])

# ==============================
# TAB DASHBOARD
# ==============================

with tab1:

    st.subheader("Data Kemiskinan")

    st.dataframe(df,use_container_width=True)

    # ==========================
    # KMEANS
    # ==========================

    X = df[["gk","pengeluaran"]]

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    df["cluster"] = model.fit_predict(X)

    centroid = model.cluster_centers_

    colA,colB = st.columns(2)

    # ==========================
    # SCATTER
    # ==========================

    with colA:

        st.subheader("Visualisasi Cluster")

        fig,ax = plt.subplots()

        colors = ["green","yellow","red","blue","purple"]

        for i in range(k):

            subset = df[df["cluster"]==i]

            ax.scatter(
                subset["gk"],
                subset["pengeluaran"],
                color=colors[i],
                label=f"Cluster {i}"
            )

        ax.scatter(
            centroid[:,0],
            centroid[:,1],
            color="black",
            marker="X",
            s=200,
            label="Centroid"
        )

        ax.set_xlabel("Garis Kemiskinan")
        ax.set_ylabel("Pengeluaran")
        ax.legend()

        st.pyplot(fig)

    # ==========================
    # BAR CHART
    # ==========================

    with colB:

        st.subheader("Jumlah Data per Cluster")

        jumlah = df["cluster"].value_counts().sort_index()

        fig2,ax2 = plt.subplots()

        ax2.bar(
            [f"Cluster {i}" for i in jumlah.index],
            jumlah.values
        )

        st.pyplot(fig2)

    st.markdown("---")

    # ==========================
    # PETA
    # ==========================

    st.subheader("Peta Persebaran Kemiskinan")

    try:

        with open("indonesia-prov.geojson") as f:
            geojson = json.load(f)

        df_map = df.copy()

        df_map["provinsi"] = df_map["provinsi"].str.upper().str.strip()

        fig_map = px.choropleth(
            df_map,
            geojson=geojson,
            locations="provinsi",
            featureidkey="properties.Propinsi",
            color="cluster",
            hover_name="provinsi",
            color_continuous_scale="RdYlGn_r"
        )

        fig_map.update_geos(
            fitbounds="locations",
            visible=False
        )

        st.plotly_chart(
            fig_map,
            use_container_width=True
        )

    except:
        st.warning("File geojson belum tersedia")

# ==============================
# TAB FLUKTUASI
# ==============================

with tab2:

    st.subheader("Fluktuasi Garis Kemiskinan")

    trend = df.groupby("tahun")["gk"].mean().reset_index()

    fig3,ax3 = plt.subplots()

    ax3.plot(
        trend["tahun"],
        trend["gk"],
        marker="o"
    )

    ax3.set_xlabel("Tahun")
    ax3.set_ylabel("Rata-rata GK")
    ax3.grid(True)

    st.pyplot(fig3)

# ==============================
# TAB BIODATA
# ==============================

with tab3:

    st.subheader("Biodata Pengembang")

    st.write("Nama : Anisa Dirgahayatul Kasanah")
    st.write("NPM : 2211050024")
    st.write("Program Studi : Sistem Informasi")
