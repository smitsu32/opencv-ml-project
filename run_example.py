#!/usr/bin/env python3
"""
OpenCV機械学習プロジェクトの実行例
仮想環境内で実行してください
"""

import sys
import os

# 仮想環境のパスを追加
venv_path = os.path.join(os.path.dirname(__file__), 'venv', 'lib', 'python3.12', 'site-packages')
if os.path.exists(venv_path):
    sys.path.insert(0, venv_path)

def main():
    print("OpenCV機械学習プロジェクトの実行例")
    print("=" * 40)
    
    print("\n1. 基本的な機能テスト")
    try:
        import cv2
        import numpy as np
        print("✓ OpenCV と NumPy のインポートが成功しました")
        print(f"  - OpenCV バージョン: {cv2.__version__}")
        print(f"  - NumPy バージョン: {np.__version__}")
    except ImportError as e:
        print(f"✗ ライブラリのインポートに失敗: {e}")
        return
    
    print("\n2. Haar Cascade分類器のテスト")
    try:
        # 顔検出用のHaar Cascade分類器をロード
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("✗ 顔検出用分類器の読み込みに失敗")
        else:
            print("✓ 顔検出用分類器の読み込みが成功")
        
        # 目検出用のHaar Cascade分類器をロード
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        if eye_cascade.empty():
            print("✗ 目検出用分類器の読み込みに失敗")
        else:
            print("✓ 目検出用分類器の読み込みが成功")
    except Exception as e:
        print(f"✗ Haar Cascade分類器のテスト中にエラー: {e}")
    
    print("\n3. 機械学習ライブラリのテスト")
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        print("✓ scikit-learn のインポートが成功しました")
        
        # 簡単なモデルのテスト
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 2, 100)
        
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        print("✓ Random Forest モデルの訓練が成功")
        
        svm = SVC(kernel='linear', random_state=42)
        svm.fit(X, y)
        print("✓ SVM モデルの訓練が成功")
        
    except Exception as e:
        print(f"✗ 機械学習ライブラリのテスト中にエラー: {e}")
    
    print("\n4. 画像処理機能のテスト")
    try:
        # 簡単な画像を作成
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = [255, 0, 0]  # 青色で塗りつぶし
        
        # グレースケール変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("✓ 色空間変換が成功")
        
        # エッジ検出
        edges = cv2.Canny(gray, 50, 150)
        print("✓ エッジ検出が成功")
        
        # 輪郭検出
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("✓ 輪郭検出が成功")
        
    except Exception as e:
        print(f"✗ 画像処理機能のテスト中にエラー: {e}")
    
    print("\n" + "=" * 40)
    print("セットアップが完了しました！")
    print("\n実行方法:")
    print("1. 仮想環境をアクティベート:")
    print("   source venv/bin/activate")
    print("\n2. 各スクリプトを実行:")
    print("   python face_detection.py")
    print("   python object_detection.py")
    print("   python image_classification.py")
    print("\n3. 仮想環境を無効化:")
    print("   deactivate")

if __name__ == "__main__":
    main()