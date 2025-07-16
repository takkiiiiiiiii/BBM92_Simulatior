import numpy as np

# 入力ファイルと出力ファイルの名前を定義
input_filename = 'qber_simul_bbm92_test2.npy'
output_filename = 'qber_simul_bbm92_test99.npy'

try:
    # 1. ファイルからNumPy配列を読み込む
    data = np.load(input_filename)

    # 2. 読み込んだデータの各値に10を加算する
    modified_data = data + 0.01

    # 3. 新しいファイル名でデータを保存する
    np.save(output_filename, modified_data)
    print(data)
    print(modified_data)
    
    print(f"処理が完了しました。'{input_filename}'を読み込み、10を加算した結果を'{output_filename}'に保存しました。")

except FileNotFoundError:
    print(f"エラー: 入力ファイル '{input_filename}' が見つかりません。コードを実行する前に、ファイルが存在することを確認してください。")
except Exception as e:
    print(f"予期せぬエラーが発生しました: {e}")
