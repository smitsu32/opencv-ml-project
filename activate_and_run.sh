#!/bin/bash
# OpenCV機械学習プロジェクトの実行スクリプト

# 仮想環境をアクティベート
source venv/bin/activate

echo "仮想環境がアクティベートされました"
echo "利用可能なスクリプト:"
echo "1. python face_detection.py"
echo "2. python object_detection.py"
echo "3. python image_classification.py"
echo "4. python run_example.py"
echo ""
echo "実行したいスクリプトを選択してください:"

# 実行可能にする
chmod +x activate_and_run.sh