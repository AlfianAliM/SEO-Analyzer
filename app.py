import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="SEO Optimizer", layout="wide")
st.title(" SEO Analyz Dashboard")

uploaded_file = st.file_uploader("Upload file SEO CSV (Queries or Pages)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Deteksi mode: queries vs pages
    entity_col = df.columns[0]
    if entity_col.lower().startswith("top quer"):
        mode = "query"
        st.subheader(" Mode: Query-Based Analysis")
    elif entity_col.lower().startswith("top page"):
        mode = "page"
        st.subheader(" Mode: Page-Based Analysis")
    else:
        st.error(" File tidak dikenali. Kolom pertama harus 'Top queries' atau 'Top pages'")
        st.stop()

    # Convert CTR columns
    def parse_ctr(col):
        if df[col].dtype == object:
            if df[col].str.contains('%').any():
                df[col] = df[col].str.replace('%', '', regex=False).astype(float) / 100
            else:
                df[col] = df[col].astype(float)
        return df

    df = parse_ctr('Last 3 months CTR')
    df = parse_ctr('Previous 3 months CTR')

    # Logic flags
    df['CTR Drop'] = df['Last 3 months CTR'] < df['Previous 3 months CTR'] * 0.9
    # df['Low CTR High Position'] = (df['Last 3 months CTR'] < 0.02) & \
    #                               (df['Last 3 months Position'] < 3) & \
    #                               (df['Last 3 months Impressions'] > 5000)
    df['Low CTR High Position'] = (
        (df['Last 3 months CTR'] < 0.02) &
        (df['Last 3 months Position'] < 3) &
        (df['Last 3 months Impressions'] > 5000)
    )
    df['Click Down While Impr Up'] = (df['Last 3 months Clicks'] < df['Previous 3 months Clicks']) & \
                                     (df['Last 3 months Impressions'] > df['Previous 3 months Impressions'])

    df['Needs Optimization'] = df[['CTR Drop', 'Low CTR High Position', 'Click Down While Impr Up']].any(axis=1)

    # Derived metrics
    df["CTR Drop %"] = (df["Previous 3 months CTR"] - df["Last 3 months CTR"]) * 100
    df["Click Loss"] = df["Previous 3 months Clicks"] - df["Last 3 months Clicks"]
    df["CTR Gap"] = df["Previous 3 months CTR"] - df["Last 3 months CTR"]

    # Sidebar Filters
    st.sidebar.header("Filters")
    only_optimize = st.sidebar.checkbox(" Show only 'Needs Optimization'", value=True)

    sort_option = st.sidebar.selectbox(
        " Sort by Priority",
        options=[
            "Default",
            "Impressions (Last 3 months)",
            "CTR Drop (%)",
            "Click Loss",
            "CTR Gap",
        ]
    )

    # Filter
    display_df = df[df['Needs Optimization']] if only_optimize else df

    # Sort
    if sort_option == "Impressions (Last 3 months)":
        display_df = display_df.sort_values(by="Last 3 months Impressions", ascending=False)
    elif sort_option == "CTR Drop (%)":
        display_df = display_df.sort_values(by="CTR Drop %", ascending=False)
    elif sort_option == "Click Loss":
        display_df = display_df.sort_values(by="Click Loss", ascending=False)
    elif sort_option == "CTR Gap":
        display_df = display_df.sort_values(by="CTR Gap", ascending=False)

    # Display table
    st.markdown("###  Overview")
    st.dataframe(
        df.style
        .format({
            'Last 3 months CTR': '{:.2%}',
            'Previous 3 months CTR': '{:.2%}',
            'Last 3 months Position': '{:.2f}',
            'Previous 3 months Position': '{:.2f}',
            'CTR Drop %': '{:.2f}',
            'CTR Gap': '{:.2%}',
            'Position Change': '{:.2f}'
        })
        .applymap(lambda val: 'background-color: #ffe6e6' if val is True else '', subset=['CTR Drop', 'Low CTR High Position', 'Click Down While Impr Up'])
        .applymap(lambda val: 'background-color: #fff3cd' if val is True else '', subset=['Needs Optimization']),
        use_container_width=True
    )

    # st.dataframe(
    #     display_df.style
    #     .applymap(lambda val: 'background-color: #ffe6e6' if val is True else '', subset=['CTR Drop', 'Low CTR High Position', 'Click Down While Impr Up'])
    #     .applymap(lambda val: 'background-color: #fff3cd' if val is True else '', subset=['Needs Optimization']),
    #     use_container_width=True
    # )

    # Summary
    st.markdown("###  Summary Stats")
    st.write(f"**Total Items Displayed:** {len(display_df)}")
    st.write(f"**Total Needing Optimization (All Data):** {df['Needs Optimization'].sum()}")

    # Download
    st.markdown("###  Download Filtered CSV")
    st.download_button(
        label="Download CSV",
        data=display_df.to_csv(index=False).encode('utf-8'),
        file_name=f"seo_{mode}_filtered.csv",
        mime='text/csv'
    )

    # Visualization
    # st.markdown("### CTR Drop Distribution")
    # fig, ax = plt.subplots()
    # ax.hist(display_df["CTR Drop %"], bins=20, color="#f28e2c", edgecolor="black")
    # ax.set_xlabel("CTR Drop (%)")
    # ax.set_ylabel("Number of Items")
    # ax.set_title("Distribution of CTR Drop")
    # st.pyplot(fig)


