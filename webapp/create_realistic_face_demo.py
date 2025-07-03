#!/usr/bin/env python3
"""
現実的な顔画像を使用したデモ画像を作成
"""

import sys
import os
sys.path.insert(0, '/home/opencv-ml-project/venv/lib/python3.12/site-packages')

import cv2
import numpy as np

def create_simple_test_faces():
    """シンプルなテスト用顔画像を複数作成"""
    
    # 画像1: 単一の大きな顔
    img1 = np.ones((300, 300, 3), dtype=np.uint8) * 240
    
    # 顔の基本構造（より大きく、明確に）
    cx, cy = 150, 150
    
    # 顔の輪郭（肌色）
    cv2.circle(img1, (cx, cy), 80, (200, 180, 160), -1)
    cv2.circle(img1, (cx, cy), 80, (0, 0, 0), 2)
    
    # 額部分
    cv2.ellipse(img1, (cx, cy-20), (60, 40), 0, 0, 180, (220, 200, 180), -1)
    
    # 左目（白い部分と黒い瞳）
    cv2.ellipse(img1, (cx-30, cy-20), (15, 10), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img1, (cx-30, cy-20), 7, (0, 0, 0), -1)
    cv2.circle(img1, (cx-28, cy-22), 2, (255, 255, 255), -1)
    cv2.ellipse(img1, (cx-30, cy-20), (15, 10), 0, 0, 360, (0, 0, 0), 1)
    
    # 右目（白い部分と黒い瞳）
    cv2.ellipse(img1, (cx+30, cy-20), (15, 10), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img1, (cx+30, cy-20), 7, (0, 0, 0), -1)
    cv2.circle(img1, (cx+32, cy-22), 2, (255, 255, 255), -1)
    cv2.ellipse(img1, (cx+30, cy-20), (15, 10), 0, 0, 360, (0, 0, 0), 1)
    
    # 眉毛
    cv2.ellipse(img1, (cx-30, cy-35), (18, 6), 0, 0, 180, (80, 60, 40), -1)
    cv2.ellipse(img1, (cx+30, cy-35), (18, 6), 0, 0, 180, (80, 60, 40), -1)
    
    # 鼻
    cv2.ellipse(img1, (cx, cy), (8, 15), 0, 0, 360, (180, 160, 140), -1)
    
    # 口
    cv2.ellipse(img1, (cx, cy+30), (20, 10), 0, 0, 180, (150, 100, 100), -1)
    
    cv2.imwrite('test_face_single.jpg', img1)
    print("単一顔画像を test_face_single.jpg に保存")
    
    
    # 画像2: 3つの顔（より小さく配置）
    img2 = np.ones((400, 600, 3), dtype=np.uint8) * 250
    
    face_centers = [(150, 150), (450, 150), (300, 300)]
    
    for i, (cx, cy) in enumerate(face_centers):
        # 顔の輪郭
        cv2.circle(img2, (cx, cy), 60, (210, 190, 170), -1)
        cv2.circle(img2, (cx, cy), 60, (0, 0, 0), 2)
        
        # 額部分
        cv2.ellipse(img2, (cx, cy-15), (45, 30), 0, 0, 180, (230, 210, 190), -1)
        
        # 左目
        cv2.ellipse(img2, (cx-22, cy-15), (12, 8), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img2, (cx-22, cy-15), 5, (0, 0, 0), -1)
        cv2.circle(img2, (cx-21, cy-16), 1, (255, 255, 255), -1)
        cv2.ellipse(img2, (cx-22, cy-15), (12, 8), 0, 0, 360, (0, 0, 0), 1)
        
        # 右目
        cv2.ellipse(img2, (cx+22, cy-15), (12, 8), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img2, (cx+22, cy-15), 5, (0, 0, 0), -1)
        cv2.circle(img2, (cx+23, cy-16), 1, (255, 255, 255), -1)
        cv2.ellipse(img2, (cx+22, cy-15), (12, 8), 0, 0, 360, (0, 0, 0), 1)
        
        # 眉毛
        cv2.ellipse(img2, (cx-22, cy-25), (15, 4), 0, 0, 180, (100, 80, 60), -1)
        cv2.ellipse(img2, (cx+22, cy-25), (15, 4), 0, 0, 180, (100, 80, 60), -1)
        
        # 鼻
        cv2.ellipse(img2, (cx, cy), (6, 12), 0, 0, 360, (190, 170, 150), -1)
        
        # 口
        cv2.ellipse(img2, (cx, cy+22), (15, 8), 0, 0, 180, (160, 120, 120), -1)
        
        # 顔番号
        cv2.putText(img2, f'Face {i+1}', (cx-30, cy-80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.imwrite('test_face_triple.jpg', img2)
    print("3つ顔画像を test_face_triple.jpg に保存")
    
    
    # 画像3: 非常にシンプルな矩形ベースの顔（Haarに最適化）
    img3 = np.ones((300, 300, 3), dtype=np.uint8) * 245
    
    cx, cy = 150, 150
    
    # 顔の矩形領域
    cv2.rectangle(img3, (cx-70, cy-80), (cx+70, cy+80), (220, 220, 220), -1)
    cv2.rectangle(img3, (cx-70, cy-80), (cx+70, cy+80), (0, 0, 0), 2)
    
    # 左目（矩形ベース）
    cv2.rectangle(img3, (cx-40, cy-25), (cx-15, cy-10), (255, 255, 255), -1)
    cv2.rectangle(img3, (cx-40, cy-25), (cx-15, cy-10), (0, 0, 0), 2)
    cv2.rectangle(img3, (cx-35, cy-22), (cx-20, cy-13), (0, 0, 0), -1)
    
    # 右目（矩形ベース）
    cv2.rectangle(img3, (cx+15, cy-25), (cx+40, cy-10), (255, 255, 255), -1)
    cv2.rectangle(img3, (cx+15, cy-25), (cx+40, cy-10), (0, 0, 0), 2)
    cv2.rectangle(img3, (cx+20, cy-22), (cx+35, cy-13), (0, 0, 0), -1)
    
    # 鼻（矩形）
    cv2.rectangle(img3, (cx-5, cy-5), (cx+5, cy+10), (200, 200, 200), -1)
    
    # 口（矩形）
    cv2.rectangle(img3, (cx-20, cy+25), (cx+20, cy+35), (100, 100, 100), -1)
    
    cv2.imwrite('test_face_simple_rect.jpg', img3)
    print("矩形ベース顔画像を test_face_simple_rect.jpg に保存")
    
    return ['test_face_single.jpg', 'test_face_triple.jpg', 'test_face_simple_rect.jpg']

def test_all_face_images():
    """作成した全ての画像で顔・目検出をテスト"""
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    test_images = create_simple_test_faces()
    
    for img_path in test_images:
        print(f"\\n=== {img_path} をテスト ===")
        
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 顔検出
        faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))
        print(f"検出された顔: {len(faces)}個")
        
        total_eyes = 0
        for i, (x, y, w, h) in enumerate(faces):
            roi_gray = gray[y:y+h, x:x+w]
            
            # 複数パラメータで目検出
            eyes1 = eye_cascade.detectMultiScale(roi_gray, 1.1, 2, minSize=(5, 5), maxSize=(50, 50))
            eyes2 = eye_cascade.detectMultiScale(roi_gray, 1.05, 1, minSize=(3, 3), maxSize=(40, 40))
            eyes3 = eye_cascade.detectMultiScale(roi_gray, 1.2, 3, minSize=(8, 8), maxSize=(30, 30))
            
            all_eyes = list(eyes1) + list(eyes2) + list(eyes3)
            
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
            
            face_eyes = len(unique_eyes)
            total_eyes += face_eyes
            print(f"  顔 {i+1}: {face_eyes}個の目")
            
            # 結果を画像に描画
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            for (ex, ey, ew, eh) in unique_eyes:
                cv2.rectangle(img[y:y+h, x:x+w], (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        print(f"総検出目数: {total_eyes}個")
        
        # 結果画像を保存
        result_path = img_path.replace('.jpg', '_result.jpg')
        cv2.imwrite(result_path, img)
        print(f"結果画像: {result_path}")

if __name__ == "__main__":
    test_all_face_images()