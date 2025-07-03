from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import base64
from PIL import Image
import io
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# アップロード可能な拡張子
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_directories():
    """必要なディレクトリを作成"""
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

class WebOpenCVProcessor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def detect_faces(self, image_path):
        """顔検出処理"""
        img = cv2.imread(image_path)
        if img is None:
            return None, "画像の読み込みに失敗しました"
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 複数のパラメータで顔検出を試行
        faces = []
        
        # パラメータセット1: 標準的な検出
        faces1 = self.face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(20, 20))
        faces.extend(faces1)
        
        # パラメータセット2: より敏感な検出
        faces2 = self.face_cascade.detectMultiScale(gray, 1.05, 2, minSize=(15, 15))
        faces.extend(faces2)
        
        # パラメータセット3: 大きな顔を検出
        faces3 = self.face_cascade.detectMultiScale(gray, 1.3, 4, minSize=(30, 30))
        faces.extend(faces3)
        
        # 重複除去
        if len(faces) > 0:
            # NumPy配列に変換
            faces_array = np.array(faces)
            
            # 重複する顔の検出結果を統合
            unique_faces = []
            for face in faces_array:
                x, y, w, h = face
                is_duplicate = False
                
                for existing in unique_faces:
                    ex, ey, ew, eh = existing
                    # 重複判定：中心点が近い場合は重複
                    center_x, center_y = x + w//2, y + h//2
                    ex_center_x, ex_center_y = ex + ew//2, ey + eh//2
                    distance = ((center_x - ex_center_x)**2 + (center_y - ex_center_y)**2)**0.5
                    
                    if distance < min(w, h) * 0.3:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_faces.append((x, y, w, h))
            
            faces = unique_faces
        
        # 検出された顔に矩形を描画
        detected_eyes_count = 0
        for (x, y, w, h) in faces:
            # 顔の矩形を描画
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
            cv2.putText(img, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # 顔の領域内で目を検出
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            # 目検出のパラメータを複数パターンで調整
            eyes1 = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 2, minSize=(5, 5), maxSize=(50, 50))
            eyes2 = self.eye_cascade.detectMultiScale(roi_gray, 1.05, 1, minSize=(3, 3), maxSize=(40, 40))
            eyes3 = self.eye_cascade.detectMultiScale(roi_gray, 1.2, 3, minSize=(8, 8), maxSize=(30, 30))
            eyes4 = self.eye_cascade.detectMultiScale(roi_gray, 1.03, 1, minSize=(2, 2), maxSize=(25, 25))
            
            all_eyes = list(eyes1) + list(eyes2) + list(eyes3) + list(eyes4)
            
            # 目の重複除去
            unique_eyes = []
            for eye in all_eyes:
                ex, ey, ew, eh = eye
                is_duplicate = False
                
                for existing_eye in unique_eyes:
                    eex, eey, eew, eeh = existing_eye
                    eye_center_x, eye_center_y = ex + ew//2, ey + eh//2
                    existing_center_x, existing_center_y = eex + eew//2, eey + eeh//2
                    eye_distance = ((eye_center_x - existing_center_x)**2 + (eye_center_y - existing_center_y)**2)**0.5
                    
                    # 目の重複判定をより精細に
                    if eye_distance < min(ew, eh, eew, eeh) * 0.3:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_eyes.append((ex, ey, ew, eh))
            
            # 目の矩形を描画
            for (ex, ey, ew, eh) in unique_eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                cv2.putText(roi_color, 'Eye', (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                detected_eyes_count += 1
        
        # 結果画像を保存
        result_filename = f"face_result_{uuid.uuid4().hex[:8]}.jpg"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        cv2.imwrite(result_path, img)
        
        # 詳細なメッセージを作成
        if len(faces) > 0:
            expected_eyes = len(faces) * 2
            message = f"検出された顔の数: {len(faces)}個, 目の数: {detected_eyes_count}個 (期待値: {expected_eyes}個)"
            if detected_eyes_count < expected_eyes:
                message += f" - 一部の目が検出されない可能性があります。画像の明度や角度を調整してください。"
        else:
            message = "顔が検出されませんでした。より明確な顔の特徴を持つ画像をお試しください。"
        
        return result_filename, message
    
    def detect_objects(self, image_path):
        """物体検出処理"""
        img = cv2.imread(image_path)
        if img is None:
            return None, "画像の読み込みに失敗しました"
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 複数の手法を試して最適な検出を行う
        detected_objects = []
        
        # 手法1: 適応的閾値処理
        blur1 = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh1 = cv2.adaptiveThreshold(blur1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 手法2: 大津の手法による閾値処理
        blur2 = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh2 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 手法3: Cannyエッジ検出 + 形態学的演算
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours3, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 全ての輪郭を統合
        all_contours = contours1 + contours2 + contours3
        
        # 画像サイズに応じて最小面積を動的に設定
        img_area = img.shape[0] * img.shape[1]
        min_area = max(100, img_area * 0.001)  # 画像の0.1%以上の面積
        max_area = img_area * 0.5  # 画像の50%以下の面積
        
        # 重複除去のためのリスト
        valid_objects = []
        
        for contour in all_contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # 重複チェック（既存の検出結果と重複していないか）
                is_duplicate = False
                for existing in valid_objects:
                    ex, ey, ew, eh = existing['bbox']
                    # 重複判定：中心点が近い場合は重複とみなす
                    center_x, center_y = x + w//2, y + h//2
                    ex_center_x, ex_center_y = ex + ew//2, ey + eh//2
                    distance = ((center_x - ex_center_x)**2 + (center_y - ex_center_y)**2)**0.5
                    if distance < min(w, h, ew, eh) * 0.5:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    # 形状判定
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # より厳密な形状判定
                    vertices = len(approx)
                    aspect_ratio = float(w) / h if h > 0 else 1
                    
                    if vertices == 3:
                        shape = "Triangle"
                        color = (0, 255, 255)  # 黄色
                    elif vertices == 4:
                        # 正方形と長方形を区別
                        if 0.8 <= aspect_ratio <= 1.2:
                            shape = "Square"
                        else:
                            shape = "Rectangle"
                        color = (255, 0, 0)  # 青色
                    elif vertices >= 5:
                        # 円形度で円と楕円を区別
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                        if circularity > 0.7:
                            shape = "Circle"
                        else:
                            shape = "Ellipse"
                        color = (0, 255, 0)  # 緑色
                    else:
                        shape = "Unknown"
                        color = (128, 128, 128)  # 灰色
                    
                    obj_info = {
                        'shape': shape,
                        'area': area,
                        'bbox': (x, y, w, h),
                        'vertices': vertices
                    }
                    valid_objects.append(obj_info)
                    detected_objects.append(obj_info)
        
        # 検出結果を画像に描画
        for obj in detected_objects:
            x, y, w, h = obj['bbox']
            shape = obj['shape']
            
            # 形状に応じて色を設定
            if shape == "Triangle":
                color = (0, 255, 255)
            elif shape in ["Square", "Rectangle"]:
                color = (255, 0, 0)
            elif shape in ["Circle", "Ellipse"]:
                color = (0, 255, 0)
            else:
                color = (128, 128, 128)
            
            # 矩形とラベルを描画
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, f'{shape}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 面積も表示
            cv2.putText(img, f'Area: {int(obj["area"])}', (x, y+h+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 結果画像を保存
        result_filename = f"object_result_{uuid.uuid4().hex[:8]}.jpg"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        cv2.imwrite(result_path, img)
        
        # 詳細な結果メッセージを作成
        if detected_objects:
            shape_counts = {}
            for obj in detected_objects:
                shape = obj['shape']
                shape_counts[shape] = shape_counts.get(shape, 0) + 1
            
            result_details = []
            for shape, count in shape_counts.items():
                result_details.append(f"{shape}: {count}個")
            
            message = f"検出された物体の総数: {len(detected_objects)}個 ({', '.join(result_details)})"
        else:
            message = "物体が検出されませんでした。画像の明度や形状を確認してください。"
        
        return result_filename, message
    
    def detect_edges(self, image_path):
        """エッジ検出処理"""
        img = cv2.imread(image_path)
        if img is None:
            return None, "画像の読み込みに失敗しました"
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # エッジを3チャンネルに変換して保存
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        result_filename = f"edge_result_{uuid.uuid4().hex[:8]}.jpg"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        cv2.imwrite(result_path, edges_colored)
        
        return result_filename, "エッジ検出が完了しました"

# グローバル処理インスタンス
processor = WebOpenCVProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'ファイルが選択されていません'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'ファイルが選択されていません'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        return jsonify({
            'success': True,
            'filename': unique_filename,
            'original_name': file.filename
        })
    
    return jsonify({'error': '許可されていないファイル形式です'})

@app.route('/process/<processing_type>/<filename>')
def process_image(processing_type, filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'ファイルが見つかりません'})
    
    try:
        if processing_type == 'face':
            result_filename, message = processor.detect_faces(file_path)
        elif processing_type == 'object':
            result_filename, message = processor.detect_objects(file_path)
        elif processing_type == 'edge':
            result_filename, message = processor.detect_edges(file_path)
        else:
            return jsonify({'error': '無効な処理タイプです'})
        
        if result_filename:
            return jsonify({
                'success': True,
                'result_image': result_filename,
                'message': message,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        else:
            return jsonify({'error': message})
    
    except Exception as e:
        return jsonify({'error': f'処理中にエラーが発生しました: {str(e)}'})

@app.route('/results/<filename>')
def get_result(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/uploads/<filename>')
def get_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/demo')
def demo():
    """デモ画像で各機能をテスト"""
    return render_template('demo.html')

@app.route('/features')
def features():
    """技術詳細ページ"""
    return render_template('features.html')

@app.route('/create_demo_image/<image_type>')
def create_demo_image(image_type):
    """デモ用画像を作成"""
    try:
        if image_type == 'face':
            # 検証済みの成功パターンを使用（test_face_triple.jpgベース）
            img = np.ones((400, 600, 3), dtype=np.uint8) * 250
            
            face_centers = [(150, 150), (450, 150), (300, 300)]
            
            for i, (cx, cy) in enumerate(face_centers):
                # 顔の輪郭
                cv2.circle(img, (cx, cy), 60, (210, 190, 170), -1)
                cv2.circle(img, (cx, cy), 60, (0, 0, 0), 2)
                
                # 額部分
                cv2.ellipse(img, (cx, cy-15), (45, 30), 0, 0, 180, (230, 210, 190), -1)
                
                # 左目
                cv2.ellipse(img, (cx-22, cy-15), (12, 8), 0, 0, 360, (255, 255, 255), -1)
                cv2.circle(img, (cx-22, cy-15), 5, (0, 0, 0), -1)
                cv2.circle(img, (cx-21, cy-16), 1, (255, 255, 255), -1)
                cv2.ellipse(img, (cx-22, cy-15), (12, 8), 0, 0, 360, (0, 0, 0), 1)
                
                # 右目
                cv2.ellipse(img, (cx+22, cy-15), (12, 8), 0, 0, 360, (255, 255, 255), -1)
                cv2.circle(img, (cx+22, cy-15), 5, (0, 0, 0), -1)
                cv2.circle(img, (cx+23, cy-16), 1, (255, 255, 255), -1)
                cv2.ellipse(img, (cx+22, cy-15), (12, 8), 0, 0, 360, (0, 0, 0), 1)
                
                # 眉毛
                cv2.ellipse(img, (cx-22, cy-25), (15, 4), 0, 0, 180, (100, 80, 60), -1)
                cv2.ellipse(img, (cx+22, cy-25), (15, 4), 0, 0, 180, (100, 80, 60), -1)
                
                # 鼻
                cv2.ellipse(img, (cx, cy), (6, 12), 0, 0, 360, (190, 170, 150), -1)
                
                # 口
                cv2.ellipse(img, (cx, cy+22), (15, 8), 0, 0, 180, (160, 120, 120), -1)
                
                # 顔番号
                cv2.putText(img, f'Face {i+1}', (cx-30, cy-80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
        elif image_type == 'shapes':
            # より検出しやすい幾何学図形を作成
            img = np.zeros((400, 400, 3), dtype=np.uint8)
            img[:, :] = [240, 240, 240]  # 薄いグレー背景
            
            # 各図形を明確に分離して配置し、黒い縁取りを追加
            
            # 1. 青い四角形（左上）
            cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)
            cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), 3)
            
            # 2. 緑の円（右上）
            cv2.circle(img, (300, 100), 50, (0, 255, 0), -1)
            cv2.circle(img, (300, 100), 50, (0, 0, 0), 3)
            
            # 3. 赤い三角形（左下）
            pts1 = np.array([[100, 250], [50, 350], [150, 350]], np.int32)
            cv2.fillPoly(img, [pts1], (0, 0, 255))
            cv2.polylines(img, [pts1], True, (0, 0, 0), 3)
            
            # 4. 黄色い正方形（右下）
            cv2.rectangle(img, (250, 250), (350, 350), (0, 255, 255), -1)
            cv2.rectangle(img, (250, 250), (350, 350), (0, 0, 0), 3)
            
            # 5. 紫の楕円（中央）
            cv2.ellipse(img, (200, 200), (60, 30), 0, 0, 360, (255, 0, 255), -1)
            cv2.ellipse(img, (200, 200), (60, 30), 0, 0, 360, (0, 0, 0), 3)
            
        else:
            return jsonify({'error': '無効な画像タイプです'})
        
        # デモ画像を保存
        demo_filename = f"demo_{image_type}_{uuid.uuid4().hex[:8]}.jpg"
        demo_path = os.path.join(app.config['UPLOAD_FOLDER'], demo_filename)
        cv2.imwrite(demo_path, img)
        
        return jsonify({
            'success': True,
            'filename': demo_filename
        })
    
    except Exception as e:
        return jsonify({'error': f'デモ画像作成エラー: {str(e)}'})

if __name__ == '__main__':
    ensure_directories()
    print("OpenCV Webアプリを開始します...")
    print("ブラウザで http://localhost:5000 にアクセスしてください")
    app.run(debug=True, host='0.0.0.0', port=5000)