import cv2
import numpy as np
import os

class ObjectDetector:
    def __init__(self):
        # YOLOv4の設定ファイルと重みファイルのパス
        self.config_path = "yolov4.cfg"
        self.weights_path = "yolov4.weights"
        self.names_path = "coco.names"
        
        # クラス名を読み込み
        self.classes = self.load_classes()
        
        # 色をランダムに生成
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
    def load_classes(self):
        """COCO データセットのクラス名を読み込み"""
        # COCOデータセットのクラス名（80クラス）
        classes = [
            'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
            'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        return classes
    
    def load_yolo_model(self):
        """YOLOモデルを読み込み（事前学習済み重みが必要）"""
        try:
            net = cv2.dnn.readNet(self.weights_path, self.config_path)
            return net
        except:
            print("YOLOモデルファイルが見つかりません。代替手法を使用します。")
            return None
    
    def detect_objects_simple(self, image_path):
        """シンプルな物体検出（背景差分法）"""
        # 画像を読み込み
        img = cv2.imread(image_path)
        if img is None:
            print(f"画像が読み込めませんでした: {image_path}")
            return None
            
        # グレースケールに変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ガウシアンフィルタでノイズを除去
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 適応的閾値処理
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 輪郭検出
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 一定サイズ以上の輪郭のみを検出
        min_area = 1000
        detected_objects = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                detected_objects.append((x, y, w, h))
                
                # 矩形を描画
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f'Object {len(detected_objects)}', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img, detected_objects
    
    def detect_moving_objects(self):
        """動体検出（ウェブカメラ使用）"""
        cap = cv2.VideoCapture(0)
        
        # 背景差分器を初期化
        backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        print("動体検出を開始します。'q'キーで終了します。")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 背景差分を適用
            fgMask = backSub.apply(frame)
            
            # ノイズ除去
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
            
            # 輪郭検出
            contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 動体に矩形を描画
            moving_objects = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # 最小面積
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    moving_objects += 1
            
            # 動体数を表示
            cv2.putText(frame, f'Moving Objects: {moving_objects}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # フレームを表示
            cv2.imshow('Original', frame)
            cv2.imshow('Foreground Mask', fgMask)
            
            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def detect_edges_and_shapes(self, image_path):
        """エッジ検出と形状認識"""
        # 画像を読み込み
        img = cv2.imread(image_path)
        if img is None:
            print(f"画像が読み込めませんでした: {image_path}")
            return None
            
        # グレースケールに変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Cannyエッジ検出
        edges = cv2.Canny(gray, 50, 150)
        
        # 輪郭検出
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 形状認識
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # 最小面積
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
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img, shape, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return img, edges

def main():
    detector = ObjectDetector()
    
    print("物体検出システム")
    print("1. 静止画像から物体検出")
    print("2. ウェブカメラから動体検出")
    print("3. エッジ検出と形状認識")
    
    choice = input("選択してください (1, 2, または 3): ")
    
    if choice == "1":
        image_path = input("画像ファイルのパスを入力してください: ")
        if os.path.exists(image_path):
            result_img, objects = detector.detect_objects_simple(image_path)
            if result_img is not None:
                print(f"検出された物体の数: {len(objects)}")
                cv2.imshow('Object Detection Result', result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                # 結果を保存するか確認
                save = input("結果を保存しますか? (y/n): ")
                if save.lower() == 'y':
                    output_path = "object_detection_result.jpg"
                    cv2.imwrite(output_path, result_img)
                    print(f"結果が {output_path} に保存されました。")
        else:
            print("指定された画像ファイルが見つかりません。")
    
    elif choice == "2":
        detector.detect_moving_objects()
    
    elif choice == "3":
        image_path = input("画像ファイルのパスを入力してください: ")
        if os.path.exists(image_path):
            result_img, edges = detector.detect_edges_and_shapes(image_path)
            if result_img is not None:
                cv2.imshow('Shape Detection Result', result_img)
                cv2.imshow('Edges', edges)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("指定された画像ファイルが見つかりません。")
    
    else:
        print("無効な選択です。")

if __name__ == "__main__":
    main()