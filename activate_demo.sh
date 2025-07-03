#!/bin/bash
# 仮想環境アクティベートのデモスクリプト

echo "=== 仮想環境アクティベート デモ ==="
echo ""

echo "1. 現在のディレクトリ確認:"
pwd
echo ""

echo "2. アクティベート前の状態:"
echo "   VIRTUAL_ENV: ${VIRTUAL_ENV:-'設定されていません'}"
echo "   Python パス: $(which python3)"
echo ""

echo "3. 仮想環境をアクティベート中..."
source venv/bin/activate
echo ""

echo "4. アクティベート後の状態:"
echo "   VIRTUAL_ENV: ${VIRTUAL_ENV}"
echo "   Python パス: $(which python)"
echo ""

echo "5. モジュールインポートテスト:"
python -c "import cv2; print('   ✓ OpenCV:', cv2.__version__)"
python -c "import numpy as np; print('   ✓ NumPy:', np.__version__)"
echo ""

echo "6. 簡単なデモ実行:"
python -c "
import cv2
import numpy as np

# 小さな画像を作成
img = np.zeros((50, 50, 3), dtype=np.uint8)
img[:, :] = [255, 0, 0]  # 青色

print('   ✓ 画像作成成功: 50x50 青色画像')
print('   ✓ すべて正常に動作しています!')
"

echo ""
echo "=== デモ完了 ==="
echo ""
echo "次は以下のコマンドで実際のスクリプトを実行できます:"
echo "   python demo_test.py"
echo "   python object_detection_demo.py"
echo "   python face_detection.py"
echo ""
echo "作業終了時は 'deactivate' で仮想環境を無効化してください"