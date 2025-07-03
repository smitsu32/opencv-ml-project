#!/bin/bash
# Ubuntu環境用実行スクリプト

echo "Ubuntu環境でOpenCVプロジェクトを実行"
echo "========================================"

# 現在のディレクトリに移動
cd "$(dirname "$0")"

# 仮想環境をアクティベート
if [ -d "venv" ]; then
    echo "仮想環境をアクティベート中..."
    source venv/bin/activate
    
    echo "利用可能なスクリプト:"
    echo "1. demo_test.py (推奨)"
    echo "2. object_detection_demo.py"
    echo "3. face_detection.py"
    echo "4. image_classification.py"
    echo ""
    
    echo "例: python demo_test.py"
    echo ""
    
    # シェルを起動して仮想環境を維持
    exec bash
else
    echo "エラー: 仮想環境が見つかりません"
    echo "以下を実行してセットアップしてください:"
    echo "python3 -m venv venv"
    echo "source venv/bin/activate"
    echo "pip install -r requirements.txt"
fi
