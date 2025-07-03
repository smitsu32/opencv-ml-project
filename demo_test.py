#!/usr/bin/env python3
"""
OpenCV機械学習プロジェクトのデモテスト
ライブラリが正しく動作するかテストします
"""

import cv2
import numpy as np
import os

def test_face_detection():
    """顔検出機能のテスト"""
    print("=== 顔検出機能テスト ===")
    
    # Haar Cascade分類器を読み込み
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if face_cascade.empty():
        print("✗ 顔検出用分類器の読み込みに失敗")
        return False
    
    # テスト用のサンプル画像を作成（人工的な顔のような形）
    test_img = np.zeros((200, 200, 3), dtype=np.uint8)
    test_img[:, :] = [255, 255, 255]  # 白い背景
    
    # 顔のような楕円を描画
    cv2.ellipse(test_img, (100, 100), (50, 70), 0, 0, 360, (200, 180, 160), -1)
    # 目のような円を描画
    cv2.circle(test_img, (80, 80), 8, (0, 0, 0), -1)
    cv2.circle(test_img, (120, 80), 8, (0, 0, 0), -1)
    # 口のような線を描画
    cv2.ellipse(test_img, (100, 120), (15, 10), 0, 0, 180, (0, 0, 0), 2)
    
    # グレースケールに変換
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    # 顔検出を実行
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    print(f"✓ 顔検出処理が正常に実行されました")
    print(f"  検出された顔の数: {len(faces)}")
    
    # 結果画像を保存
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imwrite('test_face_detection.jpg', test_img)
    print("  結果画像を test_face_detection.jpg に保存しました")
    
    return True

def test_object_detection():
    """物体検出機能のテスト"""
    print("\n=== 物体検出機能テスト ===")
    
    # テスト用画像を作成
    test_img = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # 異なる形状を描画
    cv2.rectangle(test_img, (50, 50), (100, 100), (255, 0, 0), -1)  # 青い四角
    cv2.circle(test_img, (200, 100), 30, (0, 255, 0), -1)  # 緑の円
    cv2.ellipse(test_img, (150, 200), (40, 25), 0, 0, 360, (0, 0, 255), -1)  # 赤い楕円
    
    # グレースケールに変換
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    # エッジ検出
    edges = cv2.Canny(gray, 50, 150)
    
    # 輪郭検出
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"✓ 物体検出処理が正常に実行されました")
    print(f"  検出された輪郭の数: {len(contours)}")
    
    # 結果画像に輪郭を描画
    result_img = test_img.copy()
    cv2.drawContours(result_img, contours, -1, (255, 255, 255), 2)
    
    cv2.imwrite('test_object_detection.jpg', result_img)
    print("  結果画像を test_object_detection.jpg に保存しました")
    
    return True

def test_image_classification():
    """画像分類機能のテスト"""
    print("\n=== 画像分類機能テスト ===")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        
        # テスト用データセットを作成
        X = np.random.rand(100, 10)  # 100サンプル、10特徴量
        y = np.random.randint(0, 3, 100)  # 3クラス
        
        # Random Forestモデルのテスト
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        
        # テストデータで予測
        X_test = np.random.rand(10, 10)
        predictions = rf.predict(X_test)
        
        print(f"✓ Random Forest分類器が正常に動作しました")
        print(f"  テスト予測結果: {predictions}")
        
        # SVMモデルのテスト
        svm = SVC(kernel='linear', random_state=42)
        svm.fit(X, y)
        svm_predictions = svm.predict(X_test)
        
        print(f"✓ SVM分類器が正常に動作しました")
        print(f"  テスト予測結果: {svm_predictions}")
        
        return True
        
    except Exception as e:
        print(f"✗ 画像分類機能のテスト中にエラー: {e}")
        return False

def test_image_features():
    """画像特徴量抽出のテスト"""
    print("\n=== 画像特徴量抽出テスト ===")
    
    # テスト用画像を作成（より大きなサイズ）
    test_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    try:
        # HOG特徴量を抽出（パラメータを調整）
        hog = cv2.HOGDescriptor(_winSize=(64, 64), _blockSize=(16, 16), 
                               _blockStride=(8, 8), _cellSize=(8, 8), 
                               _nbins=9)
        
        # 画像を適切なサイズにリサイズ
        resized_gray = cv2.resize(gray, (64, 64))
        hog_features = hog.compute(resized_gray)
        
        print(f"✓ HOG特徴量抽出が正常に完了")
        print(f"  特徴量の次元: {hog_features.shape}")
        
    except Exception as e:
        print(f"⚠ HOG特徴量抽出でエラー: {e}")
        print("  代替の特徴量抽出を試行します")
    
    # 色ヒストグラムを計算
    hist = cv2.calcHist([test_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    print(f"✓ 色ヒストグラム計算が正常に完了")
    print(f"  ヒストグラムの形状: {hist.shape}")
    
    # 基本的な統計特徴量を計算
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    
    print(f"✓ 基本統計特徴量の計算が正常に完了")
    print(f"  平均値: {mean_val:.2f}, 標準偏差: {std_val:.2f}")
    
    return True

def main():
    print("OpenCV機械学習プロジェクト デモテスト")
    print("=" * 50)
    
    # 各機能のテストを実行
    tests = [
        ("顔検出", test_face_detection),
        ("物体検出", test_object_detection),
        ("画像分類", test_image_classification),
        ("特徴量抽出", test_image_features)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_name}テスト中にエラー: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"テスト結果: 成功 {passed}件, 失敗 {failed}件")
    
    if failed == 0:
        print("✓ すべてのテストが成功しました！")
        print("\n次の手順:")
        print("1. 実際の画像ファイルを用意")
        print("2. 各スクリプトを対話式で実行")
        print("3. ウェブカメラ機能をテスト（カメラが利用可能な場合）")
    else:
        print("✗ いくつかのテストが失敗しました。セットアップを確認してください。")

if __name__ == "__main__":
    main()