# 解決方法 - モジュールインストール問題

## 問題の原因
`/bin/python3`（システムPython）を使用してスクリプトを実行しているため、仮想環境内にインストールされたモジュールが見つからない。

## ✅ 正しい実行方法

### 方法1: 仮想環境をアクティベート（推奨）
```bash
cd /home/opencv-ml-project
source venv/bin/activate
python object_detection.py
```

### 方法2: 仮想環境のPythonを直接指定
```bash
/home/opencv-ml-project/venv/bin/python /home/opencv-ml-project/object_detection.py
```

### 方法3: 自動実行ツールを使用
```bash
cd /home/opencv-ml-project
python3 run_with_venv.py object_detection.py
```

### 方法4: デモ版を実行（非対話式）
```bash
cd /home/opencv-ml-project
source venv/bin/activate
python object_detection_demo.py
```

## ❌ エラーが発生する方法
```bash
# これはエラーになります
/bin/python3 /home/opencv-ml-project/object_detection.py
python3 /home/opencv-ml-project/object_detection.py
```

## 確認方法

### 現在の実行状況を確認
```bash
cd /home/opencv-ml-project
python3 install_check.py
```

### すべてのデモを自動実行
```bash
cd /home/opencv-ml-project
python3 run_all_demos.py
```

## 作成したファイル

| ファイル名 | 説明 |
|------------|------|
| `object_detection_demo.py` | 非対話式物体検出デモ |
| `run_with_venv.py` | 仮想環境自動実行ツール |
| `run_all_demos.py` | 全デモ自動実行ツール |
| `install_check.py` | インストール状況確認 |
| `SETUP_GUIDE.md` | 詳細セットアップガイド |

## 動作確認済み

✅ OpenCV 4.11.0 正常動作  
✅ NumPy 2.3.1 正常動作  
✅ scikit-learn 1.7.0 正常動作  
✅ 顔検出機能 正常動作  
✅ 物体検出機能 正常動作  
✅ 画像分類機能 正常動作  
✅ 特徴量抽出機能 正常動作  

## 重要なポイント

1. **必ず仮想環境をアクティベートする**
2. **システムPython（/bin/python3）は使用しない**
3. **作業終了後はdeactivateで仮想環境を無効化**