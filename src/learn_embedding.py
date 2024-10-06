import os
import random

import click
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
from scipy.linalg import orthogonal_procrustes
from sklearn.manifold import TSNE


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


# シーズンごとの埋め込みモデルを読み込む
def load_embedding(season, save_dir):
    embedding_file = os.path.join(save_dir, f"s{season}_single_ranked_teams_embedding")
    model = Word2Vec.load(embedding_file)
    return model.wv


# 他のシーズンの埋め込みを基準に合わせて整合させる関数
def align_embeddings(base_embedding, target_embedding):
    # 共通の単語（ポケモン名）を取得
    common_vocab = list(
        set(base_embedding.index_to_key) & set(target_embedding.index_to_key)
    )

    # 共通語彙のベクトルを取得
    base_vectors = np.array([base_embedding[word] for word in common_vocab])
    target_vectors = np.array([target_embedding[word] for word in common_vocab])

    # プロクルステス変換で整合
    R, _ = orthogonal_procrustes(target_vectors, base_vectors)
    aligned_vectors = np.dot(target_embedding.vectors, R)

    # aligned_vectors を KeyedVectors に戻す
    aligned_kv = KeyedVectors(vector_size=base_embedding.vector_size)
    aligned_kv.add_vectors(target_embedding.index_to_key, aligned_vectors)

    return aligned_kv


def rename_with_season(embedding, season):
    new_keyed_vectors = KeyedVectors(vector_size=embedding.vector_size)
    new_vocab = [f"{word}({season})" for word in embedding.index_to_key]
    new_keyed_vectors.add_vectors(new_vocab, embedding.vectors)
    return new_keyed_vectors


@click.command()
@click.option("--data_dir", default="data", help="Directory to save data")
@click.option("--save_dir", default="embeddings", help="Directory to save embeddings")
@click.option("--max_season", default=23, help="Maximum season to process")
@click.option("--base_season", default=1, help="Base season for alignment")
def main(data_dir, save_dir, max_season, base_season):
    # データの保存先ディレクトリを指定（必要に応じて変更）
    os.makedirs(save_dir, exist_ok=True)

    # シーズン範囲
    seasons = range(1, max_season)

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

    print("Alignment of embeddings...")
    # 基準シーズンの埋め込みを取得
    base_embedding = load_embedding(base_season, save_dir)

    # シーズンごとの整合済み埋め込みを保存
    aligned_embeddings = {base_season: base_embedding}

    for season in seasons:
        if season != base_season:
            target_embedding = load_embedding(season, save_dir)
            aligned_embedding = align_embeddings(base_embedding, target_embedding)
            aligned_embeddings[season] = aligned_embedding

    # すべてのシーズンの埋め込みを連結
    all_words = []
    all_vectors = []

    for season, embedding in aligned_embeddings.items():
        renamed_embedding = rename_with_season(embedding, season)
        all_words.extend(renamed_embedding.index_to_key)
        all_vectors.extend(renamed_embedding.vectors)

    # t-SNE で次元削減
    print("Dimensionality reduction with t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(np.array(all_vectors))

    df = pd.DataFrame(
        {"word": all_words, "x": reduced_vectors[:, 0], "y": reduced_vectors[:, 1]}
    )

    # シーズンごとにフィルタリング用の情報を追加
    df["season"] = [word.split("(")[-1].replace(")", "") for word in df["word"]]

    df.to_csv("output/aligned_pokemon_embeddings_tsne.csv", index=False)

    print("Saved aligned embeddings to output/aligned_pokemon_embeddings_tsne.csv")


if __name__ == "__main__":
    main()
