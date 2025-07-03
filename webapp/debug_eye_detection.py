#!/usr/bin/env python3
"""
目検出の詳細なデバッグスクリプト
"""

import sys
import os
sys.path.insert(0, '/home/opencv-ml-project/venv/lib/python3.12/site-packages')

import cv2
import numpy as np

def debug_eye_detection():
    """目検出のデバッグ"""
    
    # Haar Cascade分類器をロード
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # デモ画像を読み込み
    img_path = '/home/opencv-ml-project/webapp/static/uploads/demo_face_46b653a1.jpg'
    if not os.path.exists(img_path):
        print(f"画像が見つかりません: {img_path}")
        return
    
    img = cv2.imread(img_path)
    if img is None:
        print("画像の読み込みに失敗しました")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"画像サイズ: {img.shape}")
    print(f"グレースケール画像の平均輝度: {np.mean(gray):.2f}")
    print(f"グレースケール画像の標準偏差: {np.std(gray):.2f}")
    
    # 顔検出
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(20, 20))
    print(f"検出された顔の数: {len(faces)}")
    
    for i, (x, y, w, h) in enumerate(faces):
        print(f"\n顔 {i+1}: x={x}, y={y}, w={w}, h={h}")
        
        # 顔の領域を取得
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        print(f"  顔領域サイズ: {roi_gray.shape}")
        print(f"  顔領域の平均輝度: {np.mean(roi_gray):.2f}")
        print(f"  顔領域の標準偏差: {np.std(roi_gray):.2f}")
        
        # 複数のパラメータで目検出を試行
        eye_params = [
            (1.1, 2, (5, 5), (50, 50)),
            (1.05, 1, (3, 3), (40, 40)),
            (1.2, 3, (8, 8), (30, 30)),
            (1.03, 1, (2, 2), (25, 25)),
            (1.15, 2, (4, 4), (35, 35)),
            (1.08, 1, (2, 2), (20, 20))
        ]
        
        all_eyes = []
        for j, (scale, neighbors, min_size, max_size) in enumerate(eye_params):
            eyes = eye_cascade.detectMultiScale(roi_gray, scale, neighbors, 
                                              minSize=min_size, maxSize=max_size)
            print(f"  パラメータセット {j+1} (scale={scale}, neighbors={neighbors}): {len(eyes)}個の目")
            all_eyes.extend(eyes)
        
        print(f"  全パラメータ統合: {len(all_eyes)}個の目候補")
        
        # 重複除去
        unique_eyes = []
        for eye in all_eyes:
            ex, ey, ew, eh = eye
            is_duplicate = False
            
            for existing_eye in unique_eyes:
                eex, eey, eew, eeh = existing_eye
                eye_center_x, eye_center_y = ex + ew//2, ey + eh//2
                existing_center_x, existing_center_y = eex + eew//2, eey + eeh//2
                eye_distance = ((eye_center_x - existing_center_x)**2 + (eye_center_y - existing_center_y)**2)**0.5
                
                if eye_distance < min(ew, eh, eew, eeh) * 0.3:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_eyes.append((ex, ey, ew, eh))
        
        print(f"  重複除去後: {len(unique_eyes)}個の目")
        
        # 目の詳細情報を表示
        for j, (ex, ey, ew, eh) in enumerate(unique_eyes):
            print(f"    目 {j+1}: ex={ex}, ey={ey}, ew={ew}, eh={eh}")
            print(f"    目の領域サイズ: {ew}x{eh}")
        
        # 顔領域の画像を保存（デバッグ用）
        cv2.imwrite(f'debug_face_{i+1}.jpg', roi_color)
        
        # 目の候補を全て描画したデバッグ画像を作成
        debug_roi = roi_color.copy()
        for ex, ey, ew, eh in all_eyes:
            cv2.rectangle(debug_roi, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)
        cv2.imwrite(f'debug_eyes_all_{i+1}.jpg', debug_roi)
        
        # 最終的な目の検出結果を描画
        final_roi = roi_color.copy()
        for ex, ey, ew, eh in unique_eyes:
            cv2.rectangle(final_roi, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
        cv2.imwrite(f'debug_eyes_final_{i+1}.jpg', final_roi)
    
    print("\nデバッグ画像を保存しました:")
    print("- debug_face_*.jpg: 各顔の領域")
    print("- debug_eyes_all_*.jpg: 全ての目候補")
    print("- debug_eyes_final_*.jpg: 最終的な目検出結果")

if __name__ == "__main__":
    debug_eye_detection()