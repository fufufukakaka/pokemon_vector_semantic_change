# pokemon_vector_semantic_change
https://sv.pokedb.tokyo/guide/opendata で公開されているデータを用いて、意味変化の大きいポケモンとそうでないポケモンについて分析します

## データのダウンロード

```bash
uv run python src/get_battle_team_data.py
```

## 埋め込み表現の学習

```bash
uv run python src/learn_embedding.py
```