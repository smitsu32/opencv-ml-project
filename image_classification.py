import cv2
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class ImageClassifier:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_extractor = None
        
    def extract_features(self, image_path):
        """画像から特徴量を抽出"""
        # 画像を読み込み
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # グレースケールに変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 画像をリサイズ（統一サイズ）
        resized = cv2.resize(gray, (64, 64))
        
        # HOG特徴量を抽出
        hog = cv2.HOGDescriptor()
        hog_features = hog.compute(resized)
        
        # 色ヒストグラムを計算
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_features = hist.flatten()
        
        # LBP (Local Binary Pattern) 特徴量
        lbp_features = self.calculate_lbp(resized)
        
        # 特徴量を結合
        if hog_features is not None:
            features = np.concatenate([hog_features.flatten(), hist_features, lbp_features])
        else:
            features = np.concatenate([hist_features, lbp_features])
            
        return features
    
    def calculate_lbp(self, gray_image):
        """Local Binary Pattern特徴量を計算"""
        height, width = gray_image.shape
        lbp = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                center = gray_image[i, j]
                binary = []
                
                # 8近傍の画素値を取得
                neighbors = [
                    gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                    gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                    gray_image[i+1, j-1], gray_image[i, j-1]
                ]
                
                # 中心画素と比較してバイナリ値を決定
                for neighbor in neighbors:
                    binary.append(1 if neighbor >= center else 0)
                
                # バイナリ値を10進数に変換
                lbp[i, j] = sum([binary[k] * (2**k) for k in range(8)])
        
        # ヒストグラムを計算
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        return hist
    
    def load_dataset(self, dataset_path):
        """データセットを読み込み"""
        features = []
        labels = []
        
        # データセットディレクトリ内のサブディレクトリ（クラス）を取得
        classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        
        print(f"検出されたクラス: {classes}")
        
        for class_name in classes:
            class_path = os.path.join(dataset_path, class_name)
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"クラス '{class_name}' の画像数: {len(image_files)}")
            
            for image_file in image_files:
                image_path = os.path.join(class_path, image_file)
                feature = self.extract_features(image_path)
                
                if feature is not None:
                    features.append(feature)
                    labels.append(class_name)
        
        return np.array(features), np.array(labels)
    
    def train_model(self, features, labels, model_type='random_forest'):
        """モデルを訓練"""
        # ラベルをエンコード
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # データを訓練用とテスト用に分割
        X_train, X_test, y_train, y_test = train_test_split(
            features, encoded_labels, test_size=0.2, random_state=42
        )
        
        # モデルを選択
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.model = SVC(kernel='rbf', random_state=42)
        else:
            raise ValueError("サポートされていないモデルタイプです")
        
        # モデルを訓練
        print("モデルを訓練中...")
        self.model.fit(X_train, y_train)
        
        # テストデータで評価
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"モデルの精度: {accuracy:.2f}")
        print("\n分類レポート:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return accuracy
    
    def predict_image(self, image_path):
        """画像を分類"""
        if self.model is None:
            print("モデルが訓練されていません。")
            return None
            
        # 特徴量を抽出
        features = self.extract_features(image_path)
        if features is None:
            print("画像の特徴量を抽出できませんでした。")
            return None
        
        # 予測
        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0] if hasattr(self.model, 'predict_proba') else None
        
        # ラベルをデコード
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        
        return predicted_class, probabilities
    
    def save_model(self, model_path):
        """モデルを保存"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"モデルが {model_path} に保存されました。")
    
    def load_model(self, model_path):
        """モデルを読み込み"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        
        print(f"モデルが {model_path} から読み込まれました。")
    
    def create_sample_dataset(self, output_path):
        """サンプルデータセットを作成（シンプルな幾何学図形）"""
        os.makedirs(output_path, exist_ok=True)
        
        # 円、四角、三角形のクラスを作成
        classes = ['circle', 'rectangle', 'triangle']
        
        for class_name in classes:
            class_path = os.path.join(output_path, class_name)
            os.makedirs(class_path, exist_ok=True)
            
            # 各クラスで20枚の画像を生成
            for i in range(20):
                img = np.zeros((100, 100, 3), dtype=np.uint8)
                
                if class_name == 'circle':
                    center = (50, 50)
                    radius = np.random.randint(20, 40)
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    cv2.circle(img, center, radius, color, -1)
                    
                elif class_name == 'rectangle':
                    x, y = np.random.randint(10, 30), np.random.randint(10, 30)
                    w, h = np.random.randint(30, 50), np.random.randint(30, 50)
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
                    
                elif class_name == 'triangle':
                    pts = np.array([[50, 20], [20, 80], [80, 80]], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    cv2.fillPoly(img, [pts], color)
                
                # ノイズを追加
                noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
                img = cv2.add(img, noise)
                
                # 画像を保存
                cv2.imwrite(os.path.join(class_path, f'{class_name}_{i:02d}.png'), img)
        
        print(f"サンプルデータセットが {output_path} に作成されました。")

def main():
    classifier = ImageClassifier()
    
    print("画像分類システム")
    print("1. サンプルデータセットを作成")
    print("2. データセットでモデルを訓練")
    print("3. 画像を分類")
    print("4. モデルを保存")
    print("5. モデルを読み込み")
    
    while True:
        choice = input("\n選択してください (1-5, qで終了): ")
        
        if choice == "1":
            output_path = input("データセットの出力パスを入力してください (例: ./sample_dataset): ")
            classifier.create_sample_dataset(output_path)
        
        elif choice == "2":
            dataset_path = input("データセットのパスを入力してください: ")
            if os.path.exists(dataset_path):
                model_type = input("モデルタイプを選択してください (random_forest/svm): ")
                features, labels = classifier.load_dataset(dataset_path)
                if len(features) > 0:
                    classifier.train_model(features, labels, model_type)
                else:
                    print("データセットが空です。")
            else:
                print("指定されたパスが存在しません。")
        
        elif choice == "3":
            image_path = input("分類する画像のパスを入力してください: ")
            if os.path.exists(image_path):
                result = classifier.predict_image(image_path)
                if result:
                    predicted_class, probabilities = result
                    print(f"予測結果: {predicted_class}")
                    
                    if probabilities is not None:
                        print("各クラスの確率:")
                        for i, prob in enumerate(probabilities):
                            class_name = classifier.label_encoder.classes_[i]
                            print(f"  {class_name}: {prob:.3f}")
                    
                    # 画像を表示
                    img = cv2.imread(image_path)
                    cv2.putText(img, f'Predicted: {predicted_class}', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Classification Result', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                print("指定された画像ファイルが見つかりません。")
        
        elif choice == "4":
            model_path = input("モデルの保存パスを入力してください (例: model.pkl): ")
            classifier.save_model(model_path)
        
        elif choice == "5":
            model_path = input("モデルファイルのパスを入力してください: ")
            if os.path.exists(model_path):
                classifier.load_model(model_path)
            else:
                print("指定されたモデルファイルが見つかりません。")
        
        elif choice.lower() == "q":
            break
        
        else:
            print("無効な選択です。")

if __name__ == "__main__":
    main()