#!/usr/bin/env python3
"""
代替の顔検出デモ用画像作成
より確実に検出される画像を作成
"""

import cv2
import numpy as np

def create_simple_face_pattern():
    """シンプルな顔パターンを作成（検出されやすい）"""
    # より大きな画像で、明確なコントラストを持つ顔を作成
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[:, :] = [255, 255, 255]  # 白い背景
    
    # 顔の基本形状（楕円）- 黒で描画してコントラストを強化
    cv2.ellipse(img, (200, 200), (80, 110), 0, 0, 360, (128, 128, 128), -1)
    cv2.ellipse(img, (200, 200), (80, 110), 0, 0, 360, (0, 0, 0), 4)
    
    # 目の領域を明確に
    # 左目
    cv2.ellipse(img, (170, 170), (20, 15), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (170, 170), (20, 15), 0, 0, 360, (0, 0, 0), 3)
    cv2.circle(img, (170, 170), 8, (0, 0, 0), -1)
    
    # 右目
    cv2.ellipse(img, (230, 170), (20, 15), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (230, 170), (20, 15), 0, 0, 360, (0, 0, 0), 3)
    cv2.circle(img, (230, 170), 8, (0, 0, 0), -1)
    
    # 鼻を明確に
    cv2.ellipse(img, (200, 200), (10, 20), 0, 0, 360, (100, 100, 100), -1)
    cv2.ellipse(img, (200, 200), (10, 20), 0, 0, 360, (0, 0, 0), 2)
    
    # 口を明確に
    cv2.ellipse(img, (200, 240), (25, 15), 0, 0, 180, (50, 50, 50), -1)
    cv2.ellipse(img, (200, 240), (25, 15), 0, 0, 180, (0, 0, 0), 3)
    
    # 眉毛を追加
    cv2.ellipse(img, (170, 150), (20, 8), 0, 0, 180, (0, 0, 0), 4)
    cv2.ellipse(img, (230, 150), (20, 8), 0, 0, 180, (0, 0, 0), 4)
    
    return img

def create_multiple_faces():
    """複数の顔を含む画像を作成"""
    img = np.zeros((500, 600, 3), dtype=np.uint8)
    img[:, :] = [240, 240, 240]  # 薄いグレー背景
    
    # 3つの顔を配置
    face_positions = [(150, 150), (450, 150), (300, 350)]
    
    for i, (cx, cy) in enumerate(face_positions):
        # 顔の輪郭
        cv2.ellipse(img, (cx, cy), (60, 80), 0, 0, 360, (200, 180, 160), -1)
        cv2.ellipse(img, (cx, cy), (60, 80), 0, 0, 360, (0, 0, 0), 3)
        
        # 目
        cv2.ellipse(img, (cx-25, cy-20), (12, 8), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, (cx-25, cy-20), 5, (0, 0, 0), -1)
        
        cv2.ellipse(img, (cx+25, cy-20), (12, 8), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, (cx+25, cy-20), 5, (0, 0, 0), -1)
        
        # 鼻
        cv2.ellipse(img, (cx, cy), (6, 12), 0, 0, 360, (180, 160, 140), -1)
        
        # 口
        cv2.ellipse(img, (cx, cy+25), (15, 8), 0, 0, 180, (150, 100, 100), -1)
        
        # 顔番号を表示
        cv2.putText(img, f'Face {i+1}', (cx-30, cy-100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return img

# 代替案: よりHaar Cascadeが認識しやすいパターンを作成
def create_haar_friendly_face():
    """Haar Cascade分類器が認識しやすい顔パターンを作成"""
    img = np.ones((300, 300, 3), dtype=np.uint8) * 240
    
    # Haar特徴量は明暗のパターンを検出するので、より明確なコントラストを作成
    
    # 顔領域（明るい部分）
    cv2.rectangle(img, (75, 75), (225, 225), (220, 220, 220), -1)
    
    # 目の領域（暗い部分）- Haar特徴量の目パターンに対応
    cv2.rectangle(img, (95, 120), (125, 140), (80, 80, 80), -1)
    cv2.rectangle(img, (175, 120), (205, 140), (80, 80, 80), -1)
    
    # 目の中央（さらに暗い）
    cv2.rectangle(img, (105, 125), (115, 135), (30, 30, 30), -1)
    cv2.rectangle(img, (185, 125), (195, 135), (30, 30, 30), -1)
    
    # 鼻の影（暗い部分）
    cv2.rectangle(img, (140, 160), (160, 180), (120, 120, 120), -1)
    
    # 口の領域（暗い部分）
    cv2.rectangle(img, (125, 190), (175, 205), (90, 90, 90), -1)
    
    # 顔の境界を強化
    cv2.rectangle(img, (75, 75), (225, 225), (0, 0, 0), 2)
    
    return img

if __name__ == "__main__":
    # 複数のテスト画像を作成
    
    # 1. シンプルな顔パターン
    face1 = create_simple_face_pattern()
    cv2.imwrite('demo_face_simple.jpg', face1)
    print("シンプル顔画像を demo_face_simple.jpg に保存")
    
    # 2. 複数の顔
    face2 = create_multiple_faces()
    cv2.imwrite('demo_face_multiple.jpg', face2)
    print("複数顔画像を demo_face_multiple.jpg に保存")
    
    # 3. Haar Cascade向け顔パターン
    face3 = create_haar_friendly_face()
    cv2.imwrite('demo_face_haar.jpg', face3)
    print("Haar向け顔画像を demo_face_haar.jpg に保存")
    
    print("\n3つのテスト画像を作成しました。")
    print("これらの画像で顔検出をテストしてみてください。")