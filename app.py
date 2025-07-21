import streamlit as st
import pandas as pd
import google.generativeai as genai
import math
import time
import psycopg2
import psycopg2.extras
from datetime import datetime
import altair as alt

# --- KONFIGURASI DAN KONEKSI ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

@st.cache_resource
def get_db_conn():
    try:
        conn = psycopg2.connect(**st.secrets["postgres"])
        return conn
    except psycopg2.OperationalError as e:
        st.error(f"Gagal terhubung ke database PostgreSQL: {e}")
        return None

# --- FUNGSI-FUNGSI UTAMA ---
def init_db(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS seo_keyword_intents (
                top_query TEXT PRIMARY KEY,
                keyword_intent VARCHAR(50),
                tanggal_data_diupdate TIMESTAMP
            );
        """)
    conn.commit()

def fetch_existing_intents(conn):
    try:
        df_intents = pd.read_sql_query("SELECT top_query, keyword_intent FROM seo_keyword_intents;", conn)
        df_intents['top_query'] = df_intents['top_query'].str.lower()
        return df_intents
    except (Exception, psycopg2.Error) as e:
        st.warning(f"Tidak dapat mengambil data intent dari database: {e}")
        return pd.DataFrame(columns=['top_query', 'keyword_intent'])

def save_to_db(conn, df_to_save):
    df_filtered = df_to_save.dropna(subset=['keyword_intent'])
    df_filtered = df_filtered[df_filtered['keyword_intent'] != 'Unknown'][['Top queries', 'keyword_intent']].copy()
    if df_filtered.empty: return 0
    now = datetime.now()
    sql_insert = """
    INSERT INTO seo_keyword_intents (top_query, keyword_intent, tanggal_data_diupdate)
    VALUES (%s, %s, %s)
    ON CONFLICT (top_query) DO UPDATE SET
        keyword_intent = EXCLUDED.keyword_intent,
        tanggal_data_diupdate = EXCLUDED.tanggal_data_diupdate;
    """
    data_to_insert = [(row['Top queries'], row['keyword_intent'], now) for _, row in df_filtered.iterrows()]
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql_insert, data_to_insert)
    conn.commit()
    return len(data_to_insert)

def detect_intents_batch(keywords):
    prompt = (
        "Untuk setiap keyword di bawah ini, klasifikasikan intent-nya. "
        "Anda HARUS memilih HANYA SATU dari empat opsi berikut: Informasional, Komersial, Navigasional, Transaksional.\n\n"
        "Berikan jawaban HANYA dalam format ini: - keyword: intent\n\n"
        "Contoh:\n"
        "- cara membuat kue: Informasional\n- review hp terbaik 2024: Komersial\n"
        "- login facebook: Navigasional\n- harga tiket pesawat jakarta bali: Transaksional\n\n"
        "Berikut adalah keyword yang harus dianalisis:\n"
    )
    prompt += "\n".join([f"- {kw}" for kw in keywords])
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        raw = response.text.strip()
        intents = {
            parts[0].replace("-", "").strip().lower(): parts[1].strip().capitalize()
            for line in raw.splitlines() if ":" in line and (parts := line.split(":", 1))
        }
        return intents
    except Exception as e:
        st.error(f"[Gemini ERROR]: {e}")
        return {}

def detect_all_intents_batched(keywords, batch_size=100, delay=5):
    total = len(keywords)
    all_intents = {}
    progress_bar = st.progress(0, text="Memulai proses batch...")
    total_batches = math.ceil(total / batch_size)
    for i in range(0, total, batch_size):
        batch = keywords[i:i+batch_size]
        current_batch_num = i // batch_size + 1
        progress_bar.progress(current_batch_num / total_batches, text=f"Memproses batch {current_batch_num} dari {total_batches}...")
        try:
            result = detect_intents_batch(batch)
            all_intents.update(result)
            if i + batch_size < total:
                time.sleep(delay)
        except Exception as e:
            st.error(f"Terjadi error pada batch ke-{current_batch_num}: {e}")
            st.warning("Proses dihentikan. Data yang berhasil dianalisis sebelum error akan tetap disimpan.")
            break
    progress_bar.empty()
    return all_intents

# --- ANTARMUKA PENGGUNA (STREAMLIT UI) ---
st.set_page_config(page_title="SEO Optimizer", layout="wide")
st.title("SEO Analysis Dashboard")

def clear_state_on_upload():
    if 'df' in st.session_state:
        del st.session_state.df

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"], on_change=clear_state_on_upload)

if uploaded_file is not None:
    if 'df' not in st.session_state:
        with st.spinner("Membaca dan memproses file CSV..."):
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            keyword_col = "Top queries"
            if keyword_col not in df.columns: st.error(f"Kolom '{keyword_col}' tidak ditemukan."); st.stop()

            ### =================================================================== ###
            ### ### PERBAIKAN: Pastikan semua kolom metrik bersih dan numerik ### ###
            ### =================================================================== ###
            metric_cols = [
                'Last 3 months CTR', 'Previous 3 months CTR',
                'Last 3 months Position', 'Previous 3 months Position',
                'Last 3 months Impressions', 'Previous 3 months Impressions',
                'Last 3 months Clicks', 'Previous 3 months Clicks'
            ]

            # Pastikan semua kolom ada untuk menghindari error
            for col in metric_cols:
                if col not in df.columns:
                    df[col] = 0

            # Konversi semua kolom metrik menjadi numerik.
            # `errors='coerce'` akan mengubah nilai non-numerik (teks) menjadi NaN (kosong).
            for col in metric_cols:
                # Khusus untuk CTR yang mungkin dalam format persen
                if 'CTR' in col and df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace('%', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Isi semua sel kosong (NaN) yang mungkin muncul dengan 0.
            df[metric_cols] = df[metric_cols].fillna(0)

            # Normalisasi kolom CTR setelah dipastikan numerik
            if 'Last 3 months CTR' in df.columns and df['Last 3 months CTR'].max() > 1:
                df['Last 3 months CTR'] = df['Last 3 months CTR'] / 100
            if 'Previous 3 months CTR' in df.columns and df['Previous 3 months CTR'].max() > 1:
                df['Previous 3 months CTR'] = df['Previous 3 months CTR'] / 100
            ### =================================================================== ###
            ### ### AKHIR BLOK PERBAIKAN                                        ### ###
            ### =================================================================== ###

            # Kalkulasi sekarang aman karena semua input sudah numerik
            df['Needs Optimization'] = ((df['Last 3 months CTR'] < df['Previous 3 months CTR'] * 0.9) |
                                        ((df['Last 3 months CTR'] < 0.02) & (df['Last 3 months Position'] < 3) & (df['Last 3 months Impressions'] > 5000)) |
                                        ((df['Last 3 months Clicks'] < df['Previous 3 months Clicks']) & (df['Last 3 months Impressions'] > df['Previous 3 months Impressions'])))

        with st.spinner("Mencocokkan data dengan database..."):
            conn = get_db_conn()
            if conn:
                df_existing_intents = fetch_existing_intents(conn)
                if not df_existing_intents.empty:
                    df['query_lower'] = df[keyword_col].str.lower()
                    df = pd.merge(df, df_existing_intents, left_on='query_lower', right_on='top_query', how='left')
                    df.drop(columns=['query_lower', 'top_query'], inplace=True, errors='ignore')

            if 'keyword_intent' not in df.columns: df['keyword_intent'] = 'Unknown'
            else: df['keyword_intent'].fillna('Unknown', inplace=True)

        st.session_state.df = df
        st.rerun()


    df = st.session_state.df
    keyword_col = "Top queries"

    st.sidebar.header("Tindakan")
    unknown_intent_count = (df['keyword_intent'] == 'Unknown').sum()
    st.sidebar.write(f"**{unknown_intent_count}** keyword belum memiliki intent (dari keseluruhan data).")

    if st.sidebar.button(f" Generate & Save Intent", disabled=(unknown_intent_count == 0)):
        df_unknown = df[df['keyword_intent'] == 'Unknown']
        keywords_to_process = df_unknown[keyword_col].unique().tolist()
        if not keywords_to_process:
            st.info("Semua keyword sudah memiliki intent.")
        else:
            new_intents_dict = detect_all_intents_batched(keywords_to_process, delay=20)
            if new_intents_dict:
                with st.spinner("Menyimpan hasil AI langsung ke database..."):
                    conn = get_db_conn()
                    if conn:
                        init_db(conn)
                        df_to_save = pd.DataFrame(list(new_intents_dict.items()), columns=['Top queries', 'keyword_intent'])
                        rows_saved = save_to_db(conn, df_to_save)
                        if rows_saved > 0:
                            st.success(f"{rows_saved} intent baru berhasil disimpan ke database!")
                df_state = st.session_state.df
                df_new_intents = pd.DataFrame(list(new_intents_dict.items()), columns=['query_lower_case', 'new_intent'])
                df_state['query_lower_case'] = df_state[keyword_col].str.lower()
                df_merged = pd.merge(df_state, df_new_intents, on='query_lower_case', how='left')
                df_merged['keyword_intent'] = df_merged['new_intent'].combine_first(df_merged['keyword_intent'])
                df_merged.drop(columns=['query_lower_case', 'new_intent'], inplace=True)
                st.session_state.df = df_merged
                st.info("Tampilan akan diperbarui...")
                time.sleep(2)
                st.rerun()
            else:
                st.warning("Tidak ada hasil baru dari AI untuk disimpan atau diperbarui.")

    st.sidebar.markdown("---")
    st.sidebar.header("Filters")
    only_optimize = st.sidebar.checkbox("Hanya tampilkan yang 'Needs Optimization'", value=True)
    
    display_df = df.copy()
    if only_optimize:
        display_df = display_df[display_df['Needs Optimization']]

    st.subheader("Edit Intent Keyword")
    all_intents_list = sorted(df['keyword_intent'].unique().tolist())
    selected_intents = st.multiselect("Filter berdasarkan Intent:", options=all_intents_list, default=all_intents_list)
    if selected_intents:
        display_df = display_df[display_df['keyword_intent'].isin(selected_intents)]
    else:
        display_df = display_df.head(0)
    unknown_in_view = (display_df['keyword_intent'] == 'Unknown').sum()
    st.info(f"Anda dapat mengubah **Keyword Intent** di bawah. Tampilan saat ini memiliki **{unknown_in_view}** keyword tanpa intent.")
    
    edited_df = st.data_editor(
        display_df,
        column_config={
            "keyword_intent": st.column_config.SelectboxColumn("Keyword Intent", help="Pilih intent manual", width="medium", options=["Informasional", "Komersial", "Navigasional", "Transaksional", "Unknown"], required=True),
            "Top queries": st.column_config.TextColumn(disabled=True),
            "Needs Optimization": st.column_config.CheckboxColumn(disabled=True),
        },
        use_container_width=True, hide_index=True,
    )

    if st.button("Simpan Perubahan Manual"):
        comparison_df = df.merge(pd.DataFrame(edited_df), on=keyword_col, how="inner", suffixes=('_original', '_edited'))
        changed_rows = comparison_df[comparison_df['keyword_intent_original'] != comparison_df['keyword_intent_edited']]
        if changed_rows.empty:
            st.warning("Tidak ada perubahan yang terdeteksi.")
        else:
            df_changes = pd.DataFrame({'Top queries': changed_rows[keyword_col], 'keyword_intent': changed_rows['keyword_intent_edited']})
            conn = get_db_conn()
            if conn:
                with st.spinner(f"Menyimpan {len(df_changes)} perubahan ke database..."):
                    init_db(conn)
                    rows_affected = save_to_db(conn, df_changes)
                    st.session_state.df.set_index(keyword_col, inplace=True)
                    st.session_state.df['keyword_intent'].update(df_changes.set_index('Top queries')['keyword_intent'])
                    st.session_state.df.reset_index(inplace=True)
                    st.success(f"{rows_affected} perubahan berhasil disimpan!")
                    st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.download_button("Download Data Tampilan", display_df.to_csv(index=False).encode('utf-8'), "seo_analyzed_data.csv", "text/csv")
    
    st.markdown("---")

    ### ====================================================================== ###
    ### ### PERUBAHAN: BLOK VISUALISASI BARU MENGGUNAKAN ALTAIR               ### ###
    ### ====================================================================== ###

    df_viz = display_df[display_df['keyword_intent'] != 'Unknown'].copy()

    if not df_viz.empty:
        intent_agg = df_viz.groupby('keyword_intent')[[
            'Previous 3 months Impressions', 'Last 3 months Impressions',
            'Previous 3 months Clicks', 'Last 3 months Clicks'
        ]].sum().reset_index()

        # Fungsi untuk membuat grafik berkelompok dengan Altair
        def create_grouped_bar_chart(data, value_col_prefix, title):
            # Mengubah format data dari 'wide' ke 'long'
            data_long = data.melt(
                id_vars='keyword_intent',
                value_vars=[f'{value_col_prefix} Previous 3 months', f'{value_col_prefix} Last 3 months'],
                var_name='Periode',
                value_name='Total'
            )
            
            chart = alt.Chart(data_long).mark_bar().encode(
                x=alt.X('keyword_intent:N', title='Intent', sort='-y'),
                y=alt.Y('Total:Q', title=f'Total {value_col_prefix}'),
                color=alt.Color('Periode:N', title='Periode'),
                xOffset='Periode:N' # Kunci untuk membuat grouped bar chart
            ).properties(
                title=title
            )
            return chart

        # Membuat dan menampilkan grafik Impresi
        impressions_chart_data = intent_agg[['keyword_intent', 'Previous 3 months Impressions', 'Last 3 months Impressions']].rename(columns={
            'Previous 3 months Impressions': 'Impresi Previous 3 months',
            'Last 3 months Impressions': 'Impresi Last 3 months'
        })
        impressions_chart = create_grouped_bar_chart(impressions_chart_data, 'Impresi', 'Perbandingan Total Impresi')
        st.altair_chart(impressions_chart, use_container_width=True)

        # Membuat dan menampilkan grafik Klik
        clicks_chart_data = intent_agg[['keyword_intent', 'Previous 3 months Clicks', 'Last 3 months Clicks']].rename(columns={
            'Previous 3 months Clicks': 'Klik Previous 3 months',
            'Last 3 months Clicks': 'Klik Last 3 months'
        })
        clicks_chart = create_grouped_bar_chart(clicks_chart_data, 'Klik', 'Perbandingan Total Klik')
        st.altair_chart(clicks_chart, use_container_width=True)

    else:
        st.info("Tidak ada data untuk ditampilkan dalam visualisasi berdasarkan filter Anda saat ini.")
    

else:
    st.info(" Selamat datang! Silakan upload file CSV dari Google Search Console untuk memulai analisis.")