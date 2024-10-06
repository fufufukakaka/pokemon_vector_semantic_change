import os
import random

import pandas as pd
from gensim.models import Word2Vec


# ポケモンの列名を取得する関数
def get_pokemon_columns():
    return [f"ポケモン_{i}" for i in range(1, 7)]


# CSVファイルからデータを読み込み、ポケモンのシーケンスを作成する関数
def load_pokemon_sequences(file_path):
    df = pd.read_csv(file_path)
    pokemon_columns = get_pokemon_columns()
    pokemon_sequences = df[pokemon_columns].values.tolist()

    # 各シーケンスの順番をシャッフル (順序は入れ替え可能なのでランダム化)
    for sequence in pokemon_sequences:
        random.shuffle(sequence)

    return pokemon_sequences


def main():
    # データの保存先ディレクトリを指定（必要に応じて変更）
    data_dir = "data"
    save_dir = "embeddings"
    os.makedirs(save_dir, exist_ok=True)

    # シーズン範囲
    seasons = range(1, 23)

    # シーズンごとにデータを処理してword2vecを学習し、モデルを保存
    for season in seasons:
        # ファイルパスを生成
        file_name = f"s{season}_single_ranked_teams.csv"
        file_path = os.path.join(data_dir, file_name)

        if os.path.exists(file_path):
            print(f"Processing {file_name}...")

            # ポケモンのシーケンスを取得
            pokemon_sequences = load_pokemon_sequences(file_path)

            # word2vec モデルを学習
            model = Word2Vec(
                sentences=pokemon_sequences,
                vector_size=100,
                window=6,
                min_count=1,
                sg=1,
            )

            # モデルを保存
            embedding_file = os.path.join(
                save_dir, f"s{season}_single_ranked_teams_embedding"
            )
            model.save(embedding_file)

            print(f"Saved model for season {season} as {embedding_file}")
        else:
            print(f"File {file_name} not found, skipping season {season}.")


if __name__ == "__main__":
    main()
