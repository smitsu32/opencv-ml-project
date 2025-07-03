#!/usr/bin/env python3
"""
インストール状況確認スクリプト
"""
import os
import sys
import subprocess

def check_system_python():
    """システムのPythonを確認"""
    print("=== システムPython確認 ===")
    print(f"Python実行パス: {sys.executable}")
    print(f"Pythonバージョン: {sys.version}")
    
    # システムパッケージの確認
    try:
        import cv2
        print(f"✓ システムにOpenCVがインストール済み: {cv2.__version__}")
    except ImportError:
        print("✗ システムにOpenCVがインストールされていません")
    
    try:
        import numpy as np
        print(f"✓ システムにNumPyがインストール済み: {np.__version__}")
    except ImportError:
        print("✗ システムにNumPyがインストールされていません")

def check_venv():
    """仮想環境を確認"""
    print("\n=== 仮想環境確認 ===")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    venv_path = os.path.join(current_dir, 'venv')
    venv_python = os.path.join(venv_path, 'bin', 'python')
    
    if os.path.exists(venv_path):
        print(f"✓ 仮想環境ディレクトリ存在: {venv_path}")
        
        if os.path.exists(venv_python):
            print(f"✓ 仮想環境Python存在: {venv_python}")
            
            # 仮想環境内のパッケージを確認
            try:
                result = subprocess.run([venv_python, '-c', 'import cv2; print(cv2.__version__)'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✓ 仮想環境内OpenCV: {result.stdout.strip()}")
                else:
                    print(f"✗ 仮想環境内OpenCVエラー: {result.stderr.strip()}")
            except Exception as e:
                print(f"✗ 仮想環境確認エラー: {e}")
        else:
            print(f"✗ 仮想環境Python不在: {venv_python}")
    else:
        print(f"✗ 仮想環境ディレクトリ不在: {venv_path}")

def check_requirements():
    """requirements.txtを確認"""
    print("\n=== requirements.txt確認 ===")
    
    req_file = 'requirements.txt'
    if os.path.exists(req_file):
        print(f"✓ requirements.txt存在")
        with open(req_file, 'r') as f:
            content = f.read()
            print("内容:")
            for line in content.strip().split('\n'):
                print(f"  - {line}")
    else:
        print(f"✗ requirements.txt不在")

def reinstall_packages():
    """パッケージの再インストール"""
    print("\n=== パッケージ再インストール ===")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(current_dir, 'venv', 'bin', 'python')
    venv_pip = os.path.join(current_dir, 'venv', 'bin', 'pip')
    
    if os.path.exists(venv_python):
        print("仮想環境内でパッケージを再インストール中...")
        
        # pipのアップグレード
        subprocess.run([venv_pip, 'install', '--upgrade', 'pip'], check=False)
        
        # requirements.txtから再インストール
        if os.path.exists('requirements.txt'):
            result = subprocess.run([venv_pip, 'install', '-r', 'requirements.txt'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ パッケージ再インストール成功")
            else:
                print(f"✗ パッケージ再インストールエラー: {result.stderr}")
        else:
            # 個別パッケージインストール
            packages = ['opencv-python', 'numpy', 'matplotlib', 'scikit-learn', 'Pillow', 'imutils']
            for pkg in packages:
                print(f"  インストール中: {pkg}")
                subprocess.run([venv_pip, 'install', pkg], check=False)
    else:
        print("✗ 仮想環境が見つかりません")

def main():
    print("OpenCV機械学習プロジェクト インストール確認")
    print("=" * 50)
    
    check_system_python()
    check_venv()
    check_requirements()
    
    print("\n" + "=" * 50)
    print("推奨実行方法:")
    print("1. 仮想環境使用:")
    print("   python3 run_with_venv.py demo_test.py")
    print("2. 手動実行:")
    print("   source venv/bin/activate && python demo_test.py")
    
    # 再インストールの提案
    user_input = input("\nパッケージを再インストールしますか? (y/N): ")
    if user_input.lower() == 'y':
        reinstall_packages()
        print("\n再インストール完了。再度テストしてください。")

if __name__ == "__main__":
    main()