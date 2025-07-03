#!/usr/bin/env python3
"""
仮想環境内でスクリプトを実行するためのラッパー
"""
import os
import sys
import subprocess

def run_script_in_venv(script_name):
    """仮想環境内でスクリプトを実行"""
    # 現在のディレクトリを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 仮想環境のPythonパスを構築
    venv_python = os.path.join(current_dir, 'venv', 'bin', 'python')
    
    # スクリプトのパスを構築
    script_path = os.path.join(current_dir, script_name)
    
    # 仮想環境のPythonでスクリプトを実行
    if os.path.exists(venv_python) and os.path.exists(script_path):
        print(f"仮想環境内で {script_name} を実行中...")
        try:
            result = subprocess.run([venv_python, script_path], 
                                  capture_output=False, 
                                  text=True)
            return result.returncode == 0
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            return False
    else:
        print(f"ファイルが見つかりません:")
        print(f"  仮想環境Python: {venv_python} ({'存在' if os.path.exists(venv_python) else '不在'})")
        print(f"  スクリプト: {script_path} ({'存在' if os.path.exists(script_path) else '不在'})")
        return False

def main():
    print("OpenCV機械学習プロジェクト実行ツール")
    print("=" * 40)
    
    scripts = {
        '1': ('demo_test.py', 'デモテスト（推奨）'),
        '2': ('face_detection.py', '顔検出システム'),
        '3': ('object_detection.py', '物体検出システム'),
        '4': ('image_classification.py', '画像分類システム'),
        '5': ('run_example.py', 'セットアップテスト')
    }
    
    print("\n実行可能なスクリプト:")
    for key, (script, description) in scripts.items():
        print(f"{key}. {description}")
    
    print("\n直接実行する場合:")
    print("python run_with_venv.py <スクリプト名>")
    print("例: python run_with_venv.py demo_test.py")
    
    # コマンドライン引数がある場合
    if len(sys.argv) > 1:
        script_name = sys.argv[1]
        run_script_in_venv(script_name)
        return
    
    # 対話式実行
    try:
        choice = input("\n実行したいスクリプトを選択してください (1-5): ")
        if choice in scripts:
            script_name, description = scripts[choice]
            print(f"\n{description}を実行します...")
            run_script_in_venv(script_name)
        else:
            print("無効な選択です。")
    except KeyboardInterrupt:
        print("\n終了します。")
    except EOFError:
        print("\n自動的にデモテストを実行します...")
        run_script_in_venv('demo_test.py')

if __name__ == "__main__":
    main()