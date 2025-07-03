// グローバル変数
let uploadedFileName = null;
let currentImageType = null;

// DOM要素
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const processButtons = document.querySelectorAll('.process-btn');
const demoButtons = document.querySelectorAll('.demo-btn');
const resultsSection = document.getElementById('resultsSection');
const loading = document.getElementById('loading');
const errorMessage = document.getElementById('errorMessage');

// 初期化
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    updateProcessButtons();
});

// イベントリスナーの設定
function setupEventListeners() {
    // ファイルアップロード
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
    
    // 処理ボタン
    processButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const type = btn.dataset.type;
            if (uploadedFileName) {
                processImage(type, uploadedFileName);
            }
        });
    });
    
    // デモボタン
    demoButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const demoType = btn.dataset.demo;
            createDemoImage(demoType);
        });
    });
}

// ドラッグ&ドロップ処理
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// ファイル選択処理
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// ファイル処理
function handleFile(file) {
    // ファイルサイズチェック (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('ファイルサイズが大きすぎます（最大16MB）');
        return;
    }
    
    // ファイル形式チェック
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
    if (!allowedTypes.includes(file.type)) {
        showError('サポートされていないファイル形式です');
        return;
    }
    
    uploadFile(file);
}

// ファイルアップロード
function uploadFile(file) {
    showLoading();
    
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.success) {
            uploadedFileName = data.filename;
            showFileInfo(data.original_name, file.size);
            updateProcessButtons();
        } else {
            showError(data.error);
        }
    })
    .catch(error => {
        hideLoading();
        showError('アップロードエラー: ' + error.message);
    });
}

// ファイル情報表示
function showFileInfo(originalName, size) {
    const sizeKB = (size / 1024).toFixed(1);
    fileInfo.innerHTML = `
        <h4>✅ アップロード完了</h4>
        <p><strong>ファイル名:</strong> ${originalName}</p>
        <p><strong>サイズ:</strong> ${sizeKB} KB</p>
        <p>下の処理ボタンから画像処理を選択してください</p>
    `;
    fileInfo.style.display = 'block';
    fileInfo.classList.add('fade-in');
}

// 処理ボタンの状態更新
function updateProcessButtons() {
    processButtons.forEach(btn => {
        btn.disabled = !uploadedFileName;
    });
}

// 画像処理実行
function processImage(type, filename) {
    showLoading();
    
    fetch(`/process/${type}/${filename}`)
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.success) {
            showResults(filename, data.result_image, data.message, data.timestamp, type);
        } else {
            showError(data.error);
        }
    })
    .catch(error => {
        hideLoading();
        showError('処理エラー: ' + error.message);
    });
}

// 結果表示
function showResults(originalFilename, resultFilename, message, timestamp, processType) {
    const originalImage = document.getElementById('originalImage');
    const resultImage = document.getElementById('resultImage');
    const resultMessage = document.getElementById('resultMessage');
    const resultTimestamp = document.getElementById('resultTimestamp');
    
    originalImage.src = `/uploads/${originalFilename}`;
    resultImage.src = `/results/${resultFilename}`;
    resultMessage.textContent = message;
    resultTimestamp.textContent = `処理完了時刻: ${timestamp}`;
    
    resultsSection.style.display = 'block';
    resultsSection.classList.add('fade-in');
    
    // 結果セクションまでスクロール
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// デモ画像作成
function createDemoImage(demoType) {
    showLoading();
    
    fetch(`/create_demo_image/${demoType}`)
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.success) {
            uploadedFileName = data.filename;
            showFileInfo(`デモ画像 (${demoType})`, 0);
            updateProcessButtons();
            
            // デモ画像に適した処理を自動実行
            setTimeout(() => {
                if (demoType === 'face') {
                    processImage('face', data.filename);
                } else if (demoType === 'shapes') {
                    processImage('object', data.filename);
                }
            }, 500);
        } else {
            showError(data.error);
        }
    })
    .catch(error => {
        hideLoading();
        showError('デモ画像作成エラー: ' + error.message);
    });
}

// ローディング表示
function showLoading() {
    loading.style.display = 'flex';
}

// ローディング非表示
function hideLoading() {
    loading.style.display = 'none';
}

// エラー表示
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    
    // 5秒後に自動で非表示
    setTimeout(() => {
        errorMessage.style.display = 'none';
    }, 5000);
}

// ページ読み込み時のアニメーション
document.addEventListener('DOMContentLoaded', function() {
    const sections = document.querySelectorAll('section');
    sections.forEach((section, index) => {
        setTimeout(() => {
            section.classList.add('fade-in');
        }, index * 200);
    });
});

// 画像プレビュー機能
function previewImage(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const preview = document.createElement('img');
        preview.src = e.target.result;
        preview.style.maxWidth = '200px';
        preview.style.marginTop = '10px';
        preview.style.borderRadius = '10px';
        
        // 既存のプレビューを削除
        const existingPreview = fileInfo.querySelector('img');
        if (existingPreview) {
            existingPreview.remove();
        }
        
        fileInfo.appendChild(preview);
    };
    reader.readAsDataURL(file);
}

// エラーハンドリング改善
window.addEventListener('error', function(e) {
    console.error('JavaScript Error:', e.error);
    showError('予期しないエラーが発生しました');
});

// ネットワークエラーハンドリング
window.addEventListener('online', function() {
    console.log('Network connection restored');
});

window.addEventListener('offline', function() {
    showError('ネットワーク接続が失われました');
});