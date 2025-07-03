# 仮想環境アクティベート方法ガイド

## 基本的な手順

### 1. プロジェクトディレクトリに移動
```bash
cd /home/opencv-ml-project
```

### 2. 仮想環境をアクティベート
```bash
source venv/bin/activate
```

### 3. アクティベート確認
アクティベートされると、プロンプトの先頭に `(venv)` が表示されます：
```bash
(venv) root@DESKTOP-MRUAG5M:/home/opencv-ml-project#
```

### 4. スクリプトを実行
```bash
python object_detection.py
python demo_test.py
python face_detection.py
# など
```

### 5. 作業終了後、仮想環境を無効化
```bash
deactivate
```

## 完全な実行例

```bash
# 1. ディレクトリ移動
cd /home/opencv-ml-project

# 2. 仮想環境アクティベート
source venv/bin/activate

# 3. 確認（プロンプトに(venv)が表示される）
# (venv) root@DESKTOP-MRUAG5M:/home/opencv-ml-project#

# 4. スクリプト実行
python object_detection.py

# 5. 終了時
deactivate
```

## 仮想環境の状態確認方法

### アクティベートされているか確認
```bash
echo $VIRTUAL_ENV
# アクティベートされている場合: /home/opencv-ml-project/venv
# されていない場合: 何も表示されない
```

### 使用中のPythonパス確認
```bash
which python
# アクティベート時: /home/opencv-ml-project/venv/bin/python
# 非アクティベート時: /usr/bin/python3
```

### インストール済みパッケージ確認
```bash
pip list
```

## よくあるエラーと対処法

### エラー1: `bash: venv/bin/activate: No such file or directory`
**原因**: 仮想環境が作成されていない、または間違ったディレクトリにいる

**対処法**:
```bash
# 正しいディレクトリに移動
cd /home/opencv-ml-project

# 仮想環境の存在確認
ls -la venv/bin/activate

# 存在しない場合は仮想環境を再作成
python3 -m venv venv
```

### エラー2: `ModuleNotFoundError: No module named 'cv2'`
**原因**: 仮想環境がアクティベートされていない

**対処法**:
```bash
# 仮想環境をアクティベート
source venv/bin/activate

# プロンプトに(venv)が表示されることを確認
```

### エラー3: 仮想環境をアクティベートしたのにモジュールが見つからない
**原因**: パッケージがインストールされていない

**対処法**:
```bash
# 仮想環境内でパッケージを再インストール
source venv/bin/activate
pip install -r requirements.txt
```

## 便利なエイリアス設定

### .bashrcに追加（オプション）
```bash
# ホームディレクトリの.bashrcファイルに追加
echo 'alias opencv_activate="cd /home/opencv-ml-project && source venv/bin/activate"' >> ~/.bashrc
source ~/.bashrc

# 使用方法
opencv_activate
```

## 自動化スクリプト

### 一発実行スクリプト
```bash
# 仮想環境アクティベート + スクリプト実行
./ubuntu_run.sh

# または
python3 run_with_venv.py demo_test.py
```

## トラブルシューティング

### 仮想環境が壊れた場合
```bash
# 仮想環境を削除
rm -rf venv

# 新しい仮想環境を作成
python3 -m venv venv

# アクティベート
source venv/bin/activate

# パッケージを再インストール
pip install -r requirements.txt
```

### 権限エラーの場合
```bash
# ファイルの権限を確認
ls -la venv/bin/activate

# 実行権限を付与（必要に応じて）
chmod +x venv/bin/activate
```