# -*- coding: utf-8 -*-
"""鍵盤 hook 診斷:按任何鍵都會印出來。按 Esc 結束。"""
import keyboard

print("keyboard 版本:", getattr(keyboard, "__version__", "未知"))
print("hook 已安裝。請隨便按幾個鍵(包含右 Shift),看有沒有印出來。按 Esc 結束。\n")

def on_event(e):
    print(f"事件: name={e.name!r}  scan_code={e.scan_code}  type={e.event_type}")

keyboard.hook(on_event)
keyboard.wait("esc")
print("結束。")
