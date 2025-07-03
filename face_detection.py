import cv2
import numpy as np
import os

class FaceDetector:
    def __init__(self):
        # Haar Cascade分類器を読み込み
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def detect_faces_in_image(self, image_path):
        """画像ファイルから顔を検出"""
        # 画像を読み込み
        img = cv2.imread(image_path)
        if img is None:
            print(f"画像が読み込めませんでした: {image_path}")
            return None
            
        # グレースケールに変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 顔を検出
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # 検出された顔に矩形を描画
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # 顔の領域内で目を検出
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        return img, len(faces)
    
    def detect_faces_from_webcam(self):
        """ウェブカメラからリアルタイム顔検出"""
        cap = cv2.VideoCapture(0)
        
        print("ウェブカメラから顔検出を開始します。'q'キーで終了します。")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # グレースケールに変換
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 顔を検出
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # 検出された顔に矩形を描画
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # 顔の領域内で目を検出
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # 顔の数を表示
            cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # フレームを表示
            cv2.imshow('Face Detection', frame)
            
            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = FaceDetector()
    
    print("顔検出システム")
    print("1. 画像ファイルから顔検出")
    print("2. ウェブカメラからリアルタイム顔検出")
    
    choice = input("選択してください (1 または 2): ")
    
    if choice == "1":
        image_path = input("画像ファイルのパスを入力してください: ")
        if os.path.exists(image_path):
            result_img, face_count = detector.detect_faces_in_image(image_path)
            if result_img is not None:
                print(f"検出された顔の数: {face_count}")
                cv2.imshow('Face Detection Result', result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                # 結果を保存するか確認
                save = input("結果を保存しますか? (y/n): ")
                if save.lower() == 'y':
                    output_path = "face_detection_result.jpg"
                    cv2.imwrite(output_path, result_img)
                    print(f"結果が {output_path} に保存されました。")
        else:
            print("指定された画像ファイルが見つかりません。")
    
    elif choice == "2":
        detector.detect_faces_from_webcam()
    
    else:
        print("無効な選択です。")

if __name__ == "__main__":
    main()