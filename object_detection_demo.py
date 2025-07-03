import cv2
import numpy as np
import os

class ObjectDetectorDemo:
    def __init__(self):
        self.classes = [
            'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee'
        ]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
    
    def create_test_image(self):
        """テスト用画像を作成"""
        # 300x300の画像を作成
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[:, :] = [240, 240, 240]  # 薄いグレー背景
        
        # 異なる形状とサイズのオブジェクトを描画
        # 青い四角形
        cv2.rectangle(img, (50, 50), (120, 120), (255, 0, 0), -1)
        cv2.rectangle(img, (45, 45), (125, 125), (0, 0, 0), 2)
        
        # 緑の円
        cv2.circle(img, (200, 100), 40, (0, 255, 0), -1)
        cv2.circle(img, (200, 100), 40, (0, 0, 0), 2)
        
        # 赤い楕円
        cv2.ellipse(img, (150, 200), (50, 30), 0, 0, 360, (0, 0, 255), -1)
        cv2.ellipse(img, (150, 200), (50, 30), 0, 0, 360, (0, 0, 0), 2)
        
        # 黄色い三角形
        pts = np.array([[80, 250], [40, 200], [120, 200]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], (0, 255, 255))
        cv2.polylines(img, [pts], True, (0, 0, 0), 2)
        
        # 紫の多角形
        pts = np.array([[250, 200], [280, 180], [290, 220], [260, 240], [240, 220]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], (255, 0, 255))
        cv2.polylines(img, [pts], True, (0, 0, 0), 2)
        
        return img
    
    def detect_objects_simple(self, img):
        """シンプルな物体検出"""
        # グレースケールに変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ガウシアンフィルタでノイズを除去
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 適応的閾値処理
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 輪郭検出
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 結果画像を作成
        result_img = img.copy()
        detected_objects = []
        
        # 一定サイズ以上の輪郭のみを検出
        min_area = 500
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # 形状を判定
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 3:
                    shape = "Triangle"
                    color = (0, 255, 255)
                elif len(approx) == 4:
                    shape = "Rectangle"
                    color = (255, 0, 0)
                elif len(approx) > 4:
                    shape = "Circle/Ellipse"
                    color = (0, 255, 0)
                else:
                    shape = "Unknown"
                    color = (128, 128, 128)
                
                detected_objects.append({
                    'shape': shape,
                    'area': area,
                    'bbox': (x, y, w, h)
                })
                
                # 矩形を描画
                cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(result_img, f'{shape}', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result_img, detected_objects
    
    def detect_edges_and_shapes(self, img):
        """エッジ検出と形状認識"""
        # グレースケールに変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Cannyエッジ検出
        edges = cv2.Canny(gray, 50, 150)
        
        # 輪郭検出
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 結果画像を作成
        result_img = img.copy()
        
        # 形状認識
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # 最小面積
                # 輪郭の近似
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # 形状判定
                if len(approx) == 3:
                    shape = "Triangle"
                elif len(approx) == 4:
                    shape = "Rectangle"
                elif len(approx) > 4:
                    shape = "Circle"
                else:
                    shape = "Unknown"
                
                # 矩形を描画
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(result_img, shape, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return result_img, edges

def main():
    print("物体検出システム デモ")
    print("=" * 30)
    
    detector = ObjectDetectorDemo()
    
    # テスト用画像を作成
    print("1. テスト用画像を作成中...")
    test_img = detector.create_test_image()
    cv2.imwrite('test_input.jpg', test_img)
    print("   テスト画像を test_input.jpg に保存しました")
    
    # 物体検出を実行
    print("\n2. 物体検出を実行中...")
    result_img, objects = detector.detect_objects_simple(test_img)
    cv2.imwrite('object_detection_result.jpg', result_img)
    
    print(f"   検出された物体の数: {len(objects)}")
    for i, obj in enumerate(objects):
        print(f"   物体 {i+1}: {obj['shape']}, 面積: {obj['area']:.0f}")
    
    print("   結果画像を object_detection_result.jpg に保存しました")
    
    # エッジ検出と形状認識を実行
    print("\n3. エッジ検出と形状認識を実行中...")
    shape_result, edges = detector.detect_edges_and_shapes(test_img)
    cv2.imwrite('shape_detection_result.jpg', shape_result)
    cv2.imwrite('edges_result.jpg', edges)
    
    print("   形状認識結果を shape_detection_result.jpg に保存しました")
    print("   エッジ検出結果を edges_result.jpg に保存しました")
    
    print("\n" + "=" * 30)
    print("物体検出デモが完了しました！")
    print("生成されたファイル:")
    print("- test_input.jpg (入力画像)")
    print("- object_detection_result.jpg (物体検出結果)")
    print("- shape_detection_result.jpg (形状認識結果)")
    print("- edges_result.jpg (エッジ検出結果)")

if __name__ == "__main__":
    main()