#!/usr/bin/env python3
"""
すべてのデモを順番に実行するスクリプト
"""
import subprocess
import sys
import os

def run_demo(script_name, description):
    """デモスクリプトを実行"""
    print(f"\n{'='*50}")
    print(f"実行中: {description}")
    print(f"スクリプト: {script_name}")
    print(f"{'='*50}")
    
    # 仮想環境のPythonパスを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(current_dir, 'venv', 'bin', 'python')
    
    if os.path.exists(venv_python):
        try:
            result = subprocess.run([venv_python, script_name], 
                                  capture_output=False, text=True)
            if result.returncode == 0:
                print(f"✓ {description} 完了")
                return True
            else:
                print(f"✗ {description} でエラーが発生")
                return False
        except Exception as e:
            print(f"✗ {description} 実行中にエラー: {e}")
            return False
    else:
        print(f"✗ 仮想環境が見つかりません: {venv_python}")
        return False

def main():
    print("OpenCV機械学習プロジェクト 全デモ実行")
    print("=" * 50)
    print("このスクリプトは仮想環境を自動で使用してすべてのデモを実行します。")
    
    demos = [
        ('demo_test.py', 'システム全体テスト'),
        ('object_detection_demo.py', '物体検出デモ'),
        ('run_example.py', 'セットアップ確認')
    ]
    
    successful = 0
    failed = 0
    
    for script, description in demos:
        if os.path.exists(script):
            if run_demo(script, description):
                successful += 1
            else:
                failed += 1
        else:
            print(f"✗ スクリプトが見つかりません: {script}")
            failed += 1
    
    print(f"\n{'='*50}")
    print("実行結果:")
    print(f"成功: {successful}件")
    print(f"失敗: {failed}件")
    
    if failed == 0:
        print("\n✓ すべてのデモが正常に実行されました！")
        print("\n次に実際のスクリプトを試すには:")
        print("1. source venv/bin/activate")
        print("2. python face_detection.py")
        print("3. python object_detection.py")
        print("4. python image_classification.py")
    else:
        print("\n✗ いくつかのデモで問題が発生しました。")
        print("エラーメッセージを確認してください。")
    
    print(f"\n{'='*50}")

if __name__ == "__main__":
    main()