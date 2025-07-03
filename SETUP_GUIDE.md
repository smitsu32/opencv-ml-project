# OpenCV機械学習プロジェクト セットアップガイド

## 問題の解決方法

### 状況確認
- ✅ 仮想環境内にはすべてのモジュールが正常にインストールされています
- ❌ システムPythonにはモジュールがインストールされていません
- ❌ 仮想環境をアクティベートせずに実行するとエラーが発生します

### 正しい実行方法

#### 方法1: 仮想環境を手動でアクティベート（推奨）

```bash
# 1. プロジェクトディレクトリに移動
cd /home/opencv-ml-project

# 2. 仮想環境をアクティベート
source venv/bin/activate

# 3. スクリプトを実行
python demo_test.py
python face_detection.py
python object_detection.py
python image_classification.py

# 4. 作業終了後、仮想環境を無効化
deactivate
```

#### 方法2: 自動実行ツールを使用

```bash
# デモテストを実行
python3 run_with_venv.py demo_test.py

# または対話式で選択
python3 run_with_venv.py
```

### よくあるエラーと解決方法

#### エラー: ModuleNotFoundError: No module named 'cv2'
**原因**: 仮想環境がアクティベートされていない

**解決方法**:
```bash
source venv/bin/activate
python your_script.py
```

#### エラー: /bin/python3 で実行してもモジュールが見つからない
**原因**: システムPythonを直接使用している

**解決方法**:
```bash
# 仮想環境のPythonを直接指定
/home/opencv-ml-project/venv/bin/python your_script.py

# または仮想環境をアクティベート
source venv/bin/activate
python your_script.py
```

### 実行例

#### 1. デモテスト（推奨）
```bash
cd /home/opencv-ml-project
source venv/bin/activate
python demo_test.py
```

#### 2. 顔検出
```bash
cd /home/opencv-ml-project
source venv/bin/activate
python face_detection.py
```

#### 3. 物体検出
```bash
cd /home/opencv-ml-project
source venv/bin/activate
python object_detection.py
```

#### 4. 画像分類
```bash
cd /home/opencv-ml-project
source venv/bin/activate
python image_classification.py
```

### 確認コマンド

#### 仮想環境が正しく動作するか確認
```bash
source venv/bin/activate
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
python -c "import numpy as np; print('NumPy version:', np.__version__)"
```

#### インストール済みパッケージの確認
```bash
source venv/bin/activate
pip list
```

### トラブルシューティング

#### 仮想環境を再作成する場合
```bash
# 古い仮想環境を削除
rm -rf venv

# 新しい仮想環境を作成
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### パッケージを再インストールする場合
```bash
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### 重要なポイント

1. **必ず仮想環境をアクティベートしてから実行**
2. **システムPythonではなく仮想環境のPythonを使用**
3. **作業終了後はdeactivateで仮想環境を無効化**
4. **エラーが発生した場合は、まず仮想環境がアクティベートされているか確認**