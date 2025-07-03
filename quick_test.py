#!/usr/bin/env python3
"""
簡単なインポートテスト
"""

def test_imports():
    """必要なモジュールのインポートテスト"""
    print("Ubuntu環境でのインポートテスト")
    print("=" * 30)
    
    tests = [
        ("OpenCV", "import cv2", "cv2.__version__"),
        ("NumPy", "import numpy as np", "np.__version__"),
        ("Matplotlib", "import matplotlib", "matplotlib.__version__"),
        ("scikit-learn", "from sklearn import __version__ as sklearn_version", "sklearn_version"),
        ("Pillow", "import PIL", "PIL.__version__"),
        ("imutils", "import imutils", "'0.5.4'")  # imutilsはバージョン属性がない
    ]
    
    success_count = 0
    
    for name, import_cmd, version_cmd in tests:
        try:
            exec(import_cmd)
            version = eval(version_cmd)
            print(f"✓ {name}: {version}")
            success_count += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
    
    print(f"\n結果: {success_count}/{len(tests)} 成功")
    
    if success_count == len(tests):
        print("\n🎉 すべてのモジュールが正常にインポートできました!")
        return True
    else:
        print("\n⚠️  一部のモジュールでエラーが発生しました")
        return False

def test_opencv_functionality():
    """OpenCVの基本機能テスト"""
    print("\nOpenCV機能テスト")
    print("=" * 20)
    
    try:
        import cv2
        import numpy as np
        
        # 画像作成
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = [255, 0, 0]  # 青
        
        # グレースケール変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # エッジ検出
        edges = cv2.Canny(gray, 50, 150)
        
        print("✓ 画像作成")
        print("✓ 色空間変換")
        print("✓ エッジ検出")
        
        # Haar Cascade分類器のテスト
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if not face_cascade.empty():
            print("✓ Haar Cascade分類器")
        else:
            print("✗ Haar Cascade分類器")
        
        return True
        
    except Exception as e:
        print(f"✗ OpenCV機能テストでエラー: {e}")
        return False

def main():
    print("Ubuntu OpenCV プロジェクト 簡単テスト")
    print("=" * 40)
    
    import_success = test_imports()
    opencv_success = test_opencv_functionality()
    
    print("\n" + "=" * 40)
    
    if import_success and opencv_success:
        print("✅ すべてのテストが成功しました!")
        print("\n次のステップ:")
        print("1. source venv/bin/activate")
        print("2. python demo_test.py")
        print("3. python object_detection_demo.py")
    else:
        print("❌ 一部のテストが失敗しました")
        print("\n対処方法:")
        print("1. システムパッケージをインストール:")
        print("   sudo apt update")
        print("   sudo apt install python3-dev libopencv-dev")
        print("2. 仮想環境を再作成:")
        print("   rm -rf venv")
        print("   python3 -m venv venv")
        print("   source venv/bin/activate")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()