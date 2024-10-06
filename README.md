# pokemon_vector_semantic_change
https://sv.pokedb.tokyo/guide/opendata で公開されているデータを用いて、意味変化の大きいポケモンとそうでないポケモンについて分析します

## データのダウンロード

```bash
uv run python src/get_battle_team_data.py
```

## 埋め込み表現の学習とシーズンを同じ時空間に align させる

```bash
uv run python src/learn_embedding.py \
--data_dir data \
--save_dir embeddings \
--max_season 23 \
--base_season 1
```
