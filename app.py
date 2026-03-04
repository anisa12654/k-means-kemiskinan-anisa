import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px
import json

# ==============================
# 1. KONFIGURASI HALAMAN
# ==============================
st.set_page_config(
    page_title="Analisis K-Means Clustering Kemiskinan",
    layout="wide"
)

# ==============================
# HEADER
# ==============================
col_logo1, col_logo2, col_logo3, col_title = st.columns([0.3, 0.3, 0.3, 4])

with col_logo1:
    st.image("logo/Logo-Darmajaya-new.png", width=65)

with col_logo2:
    st.image("logo/SI.png", width=65)

with col_logo3:
    st.image("logo/logo unggul SI.png", width=65)

with col_title:
    st.markdown("### ANALISIS KLASTERISASI DAN FLUKTUASI ANGKA KEMISKINAN DI INDONESIA")
    st.markdown("###### METODE K-MEANS CLUSTERING - PRODI SISTEM INFORMASI IIB DARMAJAYA")
    st.info("Visualisasi Data Kemiskinan Berbasis Clustering")

st.markdown("---")

# ==============================
# SESSION STATE
# ==============================
if "df" not in st.session_state:
    st.session_state.df = None

if "k_val" not in st.session_state:
    st.session_state.k_val = 3

# ==============================
# FUNGSI LOAD DATA
# ==============================
def load_data_excel_aman(file):
    try:
        df = pd.read_excel(file, engine="openpyxl")
        df.columns = df.columns.astype(str).str.strip().str.lower()

        col_prov = next((c for c in df.columns if "provinsi" in c), None)
        col_gk = next((c for c in df.columns if "gk" in c or "garis" in c), None)
        col_peng = next((c for c in df.columns if "pengeluaran" in c), None)
        col_thn = next((c for c in df.columns if "tahun" in c), None)

        new_df = pd.DataFrame()

        if col_prov:
            new_df["provinsi"] = df[col_prov]

        if col_gk:
            new_df["gk"] = pd.to_numeric(df[col_gk], errors="coerce")

        if col_peng:
            new_df["pengeluaran"] = pd.to_numeric(df[col_peng], errors="coerce")

        if col_thn:
            new_df["tahun"] = pd.to_numeric(df[col_thn], errors="coerce")

        return new_df.dropna(subset=["gk", "pengeluaran"])

    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None

# ==============================
# TABS
# ==============================
tab_dasbor, tab_fluktuasi, tab_biodata = st.tabs(
    ["📊 Dasbor", "📈 Fluktuasi Kemiskinan", "👤 Biodata"]
)

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.header("Tahapan Kerja")
    selection = st.radio(
        "",
        ["Kumpulan Data K-Means", "Nilai K", "Visualisasi"]
    )

# ==============================
# TAB DASBOR
# ==============================
with tab_dasbor:

    # ==========================
    # KUMPULAN DATA
    # ==========================
    if selection == "Kumpulan Data K-Means":

        st.subheader("📊 Data Kemiskinan")

        file_otomatis = "data/hasil_cluster_3.xlsx"

        try:
            with open(file_otomatis, "rb") as f:
                data = load_data_excel_aman(f)
                if data is not None:
                    st.session_state.df = data
                    st.success("File otomatis berhasil dimuat")
        except FileNotFoundError:
            st.warning("File otomatis tidak ditemukan. Silakan upload manual.")
            up = st.file_uploader("Upload File Excel", type=["xlsx"])
            if up:
                data = load_data_excel_aman(up)
                if data is not None:
                    st.session_state.df = data
                    st.success("Data berhasil dimuat")

        if st.session_state.df is not None:
            df = st.session_state.df
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Data", len(df))
            c2.metric("Rata-rata GK", f"Rp {df['gk'].mean():,.0f}")
            c3.metric("Rata-rata Pengeluaran", f"Rp {df['pengeluaran'].mean():,.0f}")

            st.dataframe(df, use_container_width=True)
        else:
            st.info("Silakan upload data terlebih dahulu.")

    # ==========================
    # NILAI K
    # ==========================
    elif selection == "Nilai K":

        st.subheader("⚙️ Tentukan Nilai K")
        st.session_state.k_val = st.number_input(
            "Masukkan Jumlah Cluster (K):",
            min_value=2,
            max_value=10,
            value=st.session_state.k_val
        )

        st.success(f"Nilai K = {st.session_state.k_val}")

    # ==========================
    # VISUALISASI
    # ==========================
    elif selection == "Visualisasi":

        if st.session_state.df is None:
            st.warning("Silakan upload data terlebih dahulu.")
        else:

            df_final = st.session_state.df.copy()
            k = st.session_state.k_val

            # K-Means
            X = df_final[["gk", "pengeluaran"]].values
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            df_final["cluster"] = km.fit_predict(X)
            centroids = km.cluster_centers_

            warna_map = {0: "green", 1: "#FFD700", 2: "red", 3: "blue", 4: "purple"}

            res1, res2, res3, res4 = st.tabs(
                ["📍 Sebaran Titik", "📊 Diagram Batang", "🗺️ Peta Persebaran", "📄 Data"]
            )

            # ======================
            # SCATTER
            # ======================
            with res1:
                fig, ax = plt.subplots()
                for i in range(k):
                    subset = df_final[df_final["cluster"] == i]
                    ax.scatter(
                        subset["gk"],
                        subset["pengeluaran"],
                        c=warna_map.get(i, "gray"),
                        label=f"Cluster {i}",
                        edgecolors="white"
                    )

                ax.scatter(
                    centroids[:, 0],
                    centroids[:, 1],
                    c="black",
                    marker="X",
                    s=200,
                    label="Centroid"
                )

                ax.set_xlabel("Garis Kemiskinan")
                ax.set_ylabel("Pengeluaran")
                ax.legend()
                st.pyplot(fig)

            # ======================
            # BAR CHART
            # ======================
            with res2:
                counts = df_final["cluster"].value_counts().sort_index()
                fig2, ax2 = plt.subplots()
                ax2.bar(
                    [f"Cluster {i}" for i in counts.index],
                    counts.values,
                    color=[warna_map.get(i, "gray") for i in counts.index]
                )
                st.pyplot(fig2)

            # ======================
            # PETA PERSEBARAN
            # ======================
            with res3:
                st.markdown("### 🗺️ Peta Persebaran Klaster Kemiskinan")

                try:
                    with open("data/indonesia-prov.geojson") as f:
                        geojson_data = json.load(f)

                    df_final["provinsi"] = df_final["provinsi"].str.title()

                    fig_map = px.choropleth(
                    df_final,
                    geojson=geojson_data,
                    locations="provinsi",
                    featureidkey="properties.Propinsi",
                    color="cluster",
                    color_discrete_map={
                        0: "green",
                        1: "yellow",
                        2: "red"
                    }
                )

                    fig_map.update_geos(fitbounds="locations", visible=False)
                    st.plotly_chart(fig_map, use_container_width=True)

                except Exception as e:
                    st.error(f"Gagal memuat peta: {e}")

            # ======================
            # DATA
            # ======================
            with res4:
                st.dataframe(df_final, use_container_width=True)

# ==============================
# TAB FLUKTUASI
# ==============================
with tab_fluktuasi:

    if st.session_state.df is not None:
        df = st.session_state.df

        if "tahun" in df.columns:
            df_trend = df.groupby("tahun")["gk"].mean().reset_index()

            fig3, ax3 = plt.subplots(figsize=(10, 5))
            ax3.plot(df_trend["tahun"], df_trend["gk"], marker="o", color="red")
            ax3.set_xlabel("Tahun")
            ax3.set_ylabel("Rata-rata Garis Kemiskinan")
            ax3.grid(True)

            st.pyplot(fig3)
        else:
            st.warning("Kolom tahun tidak ditemukan.")
    else:
        st.info("Upload data terlebih dahulu.")

# ==============================
# TAB BIODATA
# ==============================
with tab_biodata:
    st.subheader("👤 Biodata Pengembang")
    st.write("Nama  : Anisa Dirgahayatul Kasanah")
    st.write("NPM   : 2211050024")
    st.write("Prodi : Sistem Informasi")