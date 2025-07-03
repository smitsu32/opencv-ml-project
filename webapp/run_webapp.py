#!/usr/bin/env python3
"""
OpenCV Webアプリケーション起動スクリプト
仮想環境を自動で使用してFlaskアプリを起動します
"""

import os
import sys
import subprocess
from pathlib import Path

def check_environment():
    """環境チェック"""
    print("=== 環境チェック ===")
    
    # 現在のディレクトリ
    current_dir = Path(__file__).parent.absolute()
    print(f"現在のディレクトリ: {current_dir}")
    
    # 仮想環境の確認
    venv_path = current_dir.parent / 'venv'
    venv_python = venv_path / 'bin' / 'python'
    
    if venv_path.exists():
        print(f"✓ 仮想環境ディレクトリ: {venv_path}")
        if venv_python.exists():
            print(f"✓ 仮想環境Python: {venv_python}")
            return str(venv_python)
        else:
            print(f"✗ 仮想環境Python不在: {venv_python}")
    else:
        print(f"✗ 仮想環境ディレクトリ不在: {venv_path}")
    
    return None

def check_dependencies(python_path):
    """依存関係チェック"""
    print("\n=== 依存関係チェック ===")
    
    required_modules = ['flask', 'cv2', 'numpy', 'PIL']
    
    for module in required_modules:
        try:
            result = subprocess.run([python_path, '-c', f'import {module}; print("✓ {module}")'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout.strip())
            else:
                print(f"✗ {module}: インポートエラー")
                return False
        except Exception as e:
            print(f"✗ {module}: {e}")
            return False
    
    return True

def create_directories():
    """必要なディレクトリを作成"""
    print("\n=== ディレクトリ作成 ===")
    
    directories = [
        'static/uploads',
        'static/results',
        'static/css',
        'static/js',
        'templates'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ {dir_path}")

def start_webapp(python_path):
    """Webアプリを起動"""
    print("\n=== Webアプリ起動 ===")
    print("Flask開発サーバーを起動しています...")
    print("ブラウザで http://localhost:5000 にアクセスしてください")
    print("停止するには Ctrl+C を押してください")
    print("-" * 50)
    
    try:
        # Flaskアプリを実行
        subprocess.run([python_path, 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n\nWebアプリを停止しました")
    except subprocess.CalledProcessError as e:
        print(f"\nエラーが発生しました: {e}")
        return False
    except Exception as e:
        print(f"\n予期しないエラー: {e}")
        return False
    
    return True

def install_missing_dependencies(python_path):
    """不足している依存関係をインストール"""
    print("\n=== 依存関係インストール ===")
    
    pip_path = str(Path(python_path).parent / 'pip')
    
    packages = ['Flask', 'Werkzeug']
    
    for package in packages:
        print(f"インストール中: {package}")
        try:
            subprocess.run([pip_path, 'install', package], check=True)
            print(f"✓ {package} インストール完了")
        except subprocess.CalledProcessError:
            print(f"✗ {package} インストール失敗")
            return False
    
    return True

def main():
    print("OpenCV Webアプリケーション起動ツール")
    print("=" * 40)
    
    # 環境チェック
    python_path = check_environment()
    if not python_path:
        print("\n❌ 仮想環境が見つかりません")
        print("\n解決方法:")
        print("1. プロジェクトルートディレクトリに移動")
        print("2. python3 -m venv venv")
        print("3. source venv/bin/activate")
        print("4. pip install -r requirements.txt")
        return False
    
    # 必要なディレクトリを作成
    create_directories()
    
    # 依存関係チェック
    if not check_dependencies(python_path):
        print("\n⚠️  依存関係に問題があります")
        user_input = input("自動でインストールしますか? (y/N): ")
        if user_input.lower() == 'y':
            if not install_missing_dependencies(python_path):
                print("❌ 依存関係のインストールに失敗しました")
                return False
        else:
            print("手動で以下を実行してください:")
            print("source venv/bin/activate && pip install Flask Werkzeug")
            return False
    
    # Webアプリ起動
    return start_webapp(python_path)

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)