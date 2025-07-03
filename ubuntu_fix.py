#!/usr/bin/env python3
"""
Ubuntu環境での問題を解決するスクリプト
"""
import os
import sys
import subprocess
import platform

def check_ubuntu_environment():
    """Ubuntu環境の確認"""
    print("=== Ubuntu環境チェック ===")
    
    # OS情報
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"アーキテクチャ: {platform.machine()}")
    
    # Pythonバージョン
    print(f"Pythonバージョン: {sys.version}")
    
    # 仮想環境の確認
    if 'VIRTUAL_ENV' in os.environ:
        print(f"仮想環境: {os.environ['VIRTUAL_ENV']}")
    else:
        print("仮想環境: 未アクティベート")

def install_system_dependencies():
    """必要なシステムパッケージをインストール"""
    print("\n=== システム依存関係のインストール ===")
    
    # 必要なパッケージリスト
    packages = [
        'python3-dev',
        'python3-pip',
        'python3-venv',
        'build-essential',
        'cmake',
        'pkg-config',
        'libgtk-3-dev',
        'libavcodec-dev',
        'libavformat-dev',
        'libswscale-dev',
        'libv4l-dev',
        'libxvidcore-dev',
        'libx264-dev',
        'libjpeg-dev',
        'libpng-dev',
        'libtiff-dev',
        'gfortran',
        'openexr',
        'libatlas-base-dev',
        'python3-numpy'
    ]
    
    print("必要なシステムパッケージ:")
    for pkg in packages:
        print(f"  - {pkg}")
    
    print("\n以下のコマンドを実行してください:")
    print("sudo apt update")
    print("sudo apt install -y " + " ".join(packages))

def fix_opencv_installation():
    """OpenCVインストールの修正"""
    print("\n=== OpenCVインストール修正 ===")
    
    # 仮想環境のパス
    current_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(current_dir, 'venv', 'bin', 'python')
    venv_pip = os.path.join(current_dir, 'venv', 'bin', 'pip')
    
    if os.path.exists(venv_python):
        print("仮想環境でOpenCVを再インストール中...")
        
        commands = [
            [venv_pip, 'uninstall', 'opencv-python', '-y'],
            [venv_pip, 'install', '--upgrade', 'pip'],
            [venv_pip, 'install', 'opencv-python-headless'],
            [venv_pip, 'install', 'opencv-contrib-python-headless']
        ]
        
        for cmd in commands:
            print(f"実行中: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"エラー: {result.stderr}")
                else:
                    print("成功")
            except Exception as e:
                print(f"例外: {e}")
    else:
        print("仮想環境が見つかりません")

def test_imports():
    """インポートテスト"""
    print("\n=== インポートテスト ===")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(current_dir, 'venv', 'bin', 'python')
    
    test_imports = [
        "import cv2; print('OpenCV version:', cv2.__version__)",
        "import numpy as np; print('NumPy version:', np.__version__)",
        "import matplotlib; print('Matplotlib version:', matplotlib.__version__)",
        "from sklearn import __version__; print('scikit-learn version:', __version__)"
    ]
    
    for test in test_imports:
        try:
            result = subprocess.run([venv_python, '-c', test], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ {result.stdout.strip()}")
            else:
                print(f"✗ エラー: {result.stderr.strip()}")
        except Exception as e:
            print(f"✗ 例外: {e}")

def create_alternative_runner():
    """代替実行スクリプトを作成"""
    print("\n=== 代替実行スクリプト作成 ===")
    
    script_content = '''#!/bin/bash
# Ubuntu環境用実行スクリプト

echo "Ubuntu環境でOpenCVプロジェクトを実行"
echo "========================================"

# 現在のディレクトリに移動
cd "$(dirname "$0")"

# 仮想環境をアクティベート
if [ -d "venv" ]; then
    echo "仮想環境をアクティベート中..."
    source venv/bin/activate
    
    echo "利用可能なスクリプト:"
    echo "1. demo_test.py (推奨)"
    echo "2. object_detection_demo.py"
    echo "3. face_detection.py"
    echo "4. image_classification.py"
    echo ""
    
    echo "例: python demo_test.py"
    echo ""
    
    # シェルを起動して仮想環境を維持
    exec bash
else
    echo "エラー: 仮想環境が見つかりません"
    echo "以下を実行してセットアップしてください:"
    echo "python3 -m venv venv"
    echo "source venv/bin/activate"
    echo "pip install -r requirements.txt"
fi
'''
    
    with open('ubuntu_run.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('ubuntu_run.sh', 0o755)
    print("ubuntu_run.sh を作成しました")
    print("実行方法: ./ubuntu_run.sh")

def main():
    print("Ubuntu OpenCV環境修正ツール")
    print("=" * 40)
    
    check_ubuntu_environment()
    install_system_dependencies()
    fix_opencv_installation()
    test_imports()
    create_alternative_runner()
    
    print("\n" + "=" * 40)
    print("修正完了")
    print("\n推奨実行方法:")
    print("1. ./ubuntu_run.sh")
    print("2. source venv/bin/activate && python demo_test.py")

if __name__ == "__main__":
    main()