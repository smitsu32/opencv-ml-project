#!/usr/bin/env python3
"""
改良された顔検出機能のテストスクリプト
"""

import sys
import os
sys.path.insert(0, '/home/opencv-ml-project/venv/lib/python3.12/site-packages')

import cv2
import numpy as np
from app import WebOpenCVProcessor

def create_test_face_image():
    """テスト用の顔画像を作成"""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:, :] = [240, 240, 240]  # 薄いグレー背景
    
    # メインの顔（中央）
    face_center = (150, 150)
    
    # 1. 顔の輪郭（肌色の楕円）
    cv2.ellipse(img, face_center, (60, 80), 0, 0, 360, (220, 200, 180), -1)
    cv2.ellipse(img, face_center, (60, 80), 0, 0, 360, (180, 160, 140), 3)
    
    # 2. 額の部分を強調
    cv2.ellipse(img, (150, 120), (40, 30), 0, 0, 180, (200, 180, 160), -1)
    
    # 3. 目の領域（より大きく明確に）
    # 左目
    cv2.ellipse(img, (125, 135), (12, 8), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (125, 135), 6, (0, 0, 0), -1)
    cv2.circle(img, (127, 133), 2, (255, 255, 255), -1)
    
    # 右目
    cv2.ellipse(img, (175, 135), (12, 8), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (175, 135), 6, (0, 0, 0), -1)
    cv2.circle(img, (177, 133), 2, (255, 255, 255), -1)
    
    # 4. 眉毛
    cv2.ellipse(img, (125, 120), (15, 5), 0, 0, 180, (100, 80, 60), 3)
    cv2.ellipse(img, (175, 120), (15, 5), 0, 0, 180, (100, 80, 60), 3)
    
    # 5. 鼻
    cv2.ellipse(img, (150, 155), (8, 15), 0, 0, 360, (200, 180, 160), -1)
    cv2.ellipse(img, (150, 155), (8, 15), 0, 0, 360, (180, 160, 140), 2)
    
    # 6. 口
    cv2.ellipse(img, (150, 180), (20, 12), 0, 0, 180, (150, 100, 100), -1)
    cv2.ellipse(img, (150, 180), (20, 12), 0, 0, 180, (100, 50, 50), 2)
    
    # 7. 頬を強調
    cv2.ellipse(img, (110, 160), (15, 10), 0, 0, 360, (230, 210, 190), -1)
    cv2.ellipse(img, (190, 160), (15, 10), 0, 0, 360, (230, 210, 190), -1)
    
    # 8. 顎のライン
    cv2.ellipse(img, (150, 210), (45, 25), 0, 0, 180, (200, 180, 160), -1)
    
    # 9. 髪の毛（上部）
    cv2.ellipse(img, (150, 90), (70, 40), 0, 0, 180, (80, 60, 40), -1)
    
    # 10. 影とハイライトを追加してより立体的に
    # 右側に影
    cv2.ellipse(img, (180, 150), (20, 60), 0, 0, 180, (190, 170, 150), -1)
    # 左側にハイライト  
    cv2.ellipse(img, (120, 140), (15, 50), 0, 0, 180, (240, 220, 200), -1)
    
    return img

def test_face_detection():
    """顔検出機能をテスト"""
    print("顔検出機能のテストを開始します...")
    
    # テスト画像を作成
    test_img = create_test_face_image()
    
    # 画像を保存
    test_path = "test_face_image.jpg"
    cv2.imwrite(test_path, test_img)
    print(f"テスト画像を {test_path} に保存しました")
    
    # 顔検出処理を実行
    processor = WebOpenCVProcessor()
    
    # 従来の方法でテスト
    print("\n=== 従来のパラメータでテスト ===")
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces_old = processor.face_cascade.detectMultiScale(gray, 1.3, 5)
    print(f"従来の方法: {len(faces_old)}個の顔を検出")
    
    # 改良された方法でテスト
    print("\n=== 改良されたパラメータでテスト ===")
    faces1 = processor.face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(20, 20))
    faces2 = processor.face_cascade.detectMultiScale(gray, 1.05, 2, minSize=(15, 15))
    faces3 = processor.face_cascade.detectMultiScale(gray, 1.3, 4, minSize=(30, 30))
    
    print(f"パラメータセット1 (1.1, 3): {len(faces1)}個")
    print(f"パラメータセット2 (1.05, 2): {len(faces2)}個")
    print(f"パラメータセット3 (1.3, 4): {len(faces3)}個")
    
    all_faces = list(faces1) + list(faces2) + list(faces3)
    print(f"統合結果: {len(all_faces)}個の候補")
    
    # 画像解析
    print(f"\n=== 画像情報 ===")
    print(f"画像サイズ: {test_img.shape}")
    print(f"平均輝度: {np.mean(gray):.1f}")
    print(f"標準偏差: {np.std(gray):.1f}")
    
    # 結果画像を作成
    result_img = test_img.copy()
    
    # 全ての検出結果を描画
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    labels = ["Set1", "Set2", "Set3"]
    
    for i, faces in enumerate([faces1, faces2, faces3]):
        for (x, y, w, h) in faces:
            cv2.rectangle(result_img, (x, y), (x+w, y+h), colors[i], 2)
            cv2.putText(result_img, labels[i], (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
    
    cv2.imwrite("test_face_detection_result.jpg", result_img)
    print("結果画像を test_face_detection_result.jpg に保存しました")
    
    # WebOpenCVProcessorの改良版メソッドをテスト
    print("\n=== WebOpenCVProcessor.detect_faces() テスト ===")
    try:
        result_filename, message = processor.detect_faces(test_path)
        print(f"結果: {message}")
        print(f"結果画像: {result_filename}")
    except Exception as e:
        print(f"エラー: {e}")
    
    # クリーンアップ
    if os.path.exists(test_path):
        os.remove(test_path)

if __name__ == "__main__":
    test_face_detection()