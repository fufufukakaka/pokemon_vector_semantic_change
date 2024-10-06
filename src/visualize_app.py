import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# CSVファイルの読み込み
df = pd.read_csv("output/aligned_pokemon_embeddings_tsne.csv")

# 表示オプションの選択肢を作成
display_option = st.radio(
    "表示オプションを選択してください:",
    ("シーズンスライダーを表示", "全シーズンを表示"),
)

# ポケモン名の入力欄を作成
search_word = st.text_input("追跡したいポケモンを入力:", "カイリュー")

# 検索した単語に関連するデータをフィルタリング
df["highlight"] = df["word"].str.contains(search_word)
df["color"] = df["highlight"].apply(lambda x: "highlight" if x else "normal")

# フィルタリングしたポケモンデータ
filtered_df = df[df["highlight"]]

# 間引き率のスライダー (全シーズンを表示する場合のみ有効)
if display_option == "全シーズンを表示":
    sample_fraction = st.slider("表示するデータポイントの割合を選んでください", 0.01, 1.0, 0.1, step=0.01)

# 表示するかしないかの制御
if filtered_df.empty:
    st.write(f"'{search_word}' はデータに存在しません。")
else:
    if display_option == "シーズンスライダーを表示":
        # シーズンスライダーを表示する場合のアニメーション
        fig = px.scatter(
            df,
            x="x",
            y="y",
            text="word",
            hover_name="word",
            animation_frame="season",
            title=f"ポケモン '{search_word}' の対戦における意味変化",
            color="color",  # ハイライト情報を色に反映
            color_discrete_map={
                "highlight": "red",
                "normal": "lightgray",
            },  # カスタムカラーの設定
        )

        # アニメーションの速度とUI調整
        fig.update_layout(
            width=800,  # 図の幅を設定
            height=600,  # 図の高さを設定
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {
                                        "duration": 500,
                                        "redraw": True,
                                    },  # フレームごとの表示時間を設定 (ミリ秒)
                                    "fromcurrent": True,
                                    "transition": {"duration": 300},
                                },
                            ],  # 遷移の時間を設定 (ミリ秒)
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
        )
    else:
        # 全シーズンを一度に表示する場合（間引き適用）
        sampled_df = df.sample(frac=sample_fraction)

        # 全シーズンを一度に表示する場合
        fig = px.scatter(
            sampled_df,
            x="x",
            y="y",
            text="word",
            hover_name="word",  # テキストをホバー時のみ表示
            hover_data={'season': True},
            color="color",  # ハイライト情報を色に反映
            color_discrete_map={
                "highlight": "red",
                "normal": "lightgray",
            },  # カスタムカラーの設定
            title=f"全シーズンでのポケモン '{search_word}' の意味変化",
        )
        fig.update_layout(
            width=1200,  # 図の幅を設定
            height=800,  # 図の高さを設定
        )

    # プロットを表示
    st.plotly_chart(fig, use_container_width=True)
