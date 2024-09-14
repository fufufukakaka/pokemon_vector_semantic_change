import os
import requests

# 保存先のディレクトリを指定
save_dir = "data"
os.makedirs(save_dir, exist_ok=True)

# シーズンとモードに基づいてURLを生成
def download_data(season_range, mode_list, file_format='csv'):
    base_url = "https://sv.pokedb.tokyo/opendata/s{season}_{mode}_ranked_teams.{file_format}"
    
    for season in season_range:
        for mode in mode_list:
            # URLを生成
            url = base_url.format(season=season, mode=mode, file_format=file_format)
            # 保存ファイル名を生成
            file_name = f"s{season}_{mode}_ranked_teams.{file_format}"
            file_path = os.path.join(save_dir, file_name)
            
            try:
                # データをダウンロード
                response = requests.get(url)
                response.raise_for_status()  # ステータスコードが200番台以外の場合に例外を発生させる
                
                # ダウンロードしたデータをファイルに保存
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                print(f"Downloaded and saved: {file_name}")
            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP error occurred: {http_err} - {url}")
            except Exception as err:
                print(f"Other error occurred: {err} - {url}")

def main():
    # シーズン1から21までのデータをシングルバトル(single)でダウンロード
    seasons = range(1, 22)  # シーズン1から21まで
    modes = ['single']  # シングル
    download_data(seasons, modes)


if __name__ == "__main__":
    main()