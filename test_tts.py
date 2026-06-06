# -*- coding: utf-8 -*-
"""CosyVoice raw-socket + 正確解析 HTTP chunked encoding。
測「首段 PCM 抵達時間」(client side)+「server 開始送 audio 的時間」(server side)。"""
import json
import socket
import time

import numpy as np
import sounddevice as sd

HOST = "127.0.0.1"
PORT = 8765
RATE = 24000

SAMPLES = [
    "你好。",
    "我幫你把剛剛複製的內容整理成三個重點。",
    "好,讓我仔細幫你分析一下這個情況看起來是網路設定的問題,你可以先檢查防火牆。",
]


class ChunkedReader:
    """讀 HTTP/1.1 chunked transfer encoding,每次 yield 真實 payload bytes(不含 chunk-size 標頭)。"""
    def __init__(self, sock, initial_buf=b""):
        self.sock = sock
        self.buf  = initial_buf

    def _read_until(self, sep):
        while sep not in self.buf:
            chunk = self.sock.recv(8192)
            if not chunk: return None
            self.buf += chunk
        idx = self.buf.index(sep)
        out = self.buf[:idx]
        self.buf = self.buf[idx + len(sep):]
        return out

    def _read_exact(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(8192)
            if not chunk: return None
            self.buf += chunk
        out = self.buf[:n]
        self.buf = self.buf[n:]
        return out

    def __iter__(self):
        while True:
            size_line = self._read_until(b"\r\n")
            if size_line is None: return
            size = int(size_line.split(b";", 1)[0], 16)
            if size == 0: return
            data = self._read_exact(size)
            if data is None: return
            self._read_exact(2)   # 吃掉 chunk 尾部的 \r\n
            yield data


def stream_and_play(text: str):
    body = json.dumps({"text": text}).encode("utf-8")
    req  = (
        f"POST /tts HTTP/1.1\r\nHost: {HOST}:{PORT}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("ascii") + body

    t0 = time.time()
    s  = socket.socket(); s.connect((HOST, PORT)); s.sendall(req)

    # 讀 header
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = s.recv(4096)
        if not chunk: print("  ✗ 中斷"); return
        buf += chunk
    header_t = time.time() - t0
    print(f"  HTTP header 抵達: {header_t*1000:.0f}ms")

    hdr_end  = buf.index(b"\r\n\r\n") + 4
    headers  = buf[:hdr_end].decode("latin1")
    body_buf = buf[hdr_end:]
    is_chunked = "transfer-encoding: chunked" in headers.lower()
    print(f"  chunked={is_chunked}, status={headers.split(chr(13)+chr(10),1)[0]}")

    stream = sd.OutputStream(samplerate=RATE, channels=1, dtype="int16")
    stream.start()

    first_pcm_t  = None
    total_bytes  = 0
    leftover     = b""

    if is_chunked:
        reader = ChunkedReader(s, body_buf)
        for data in reader:
            if first_pcm_t is None:
                first_pcm_t = time.time() - t0
                print(f"  首段 PCM(server 真開始送 audio):{first_pcm_t*1000:.0f}ms")
            blob = leftover + data
            even = len(blob) - (len(blob) % 2)
            if even:
                stream.write(np.frombuffer(blob[:even], dtype=np.int16))
                total_bytes += even
            leftover = blob[even:]
    else:
        # 非 chunked:讀到 EOF
        pending = body_buf
        while True:
            if not pending:
                pending = s.recv(8192)
                if not pending: break
            if first_pcm_t is None:
                first_pcm_t = time.time() - t0
                print(f"  首段 PCM: {first_pcm_t*1000:.0f}ms")
            blob = leftover + pending; pending = b""
            even = len(blob) - (len(blob) % 2)
            if even:
                stream.write(np.frombuffer(blob[:even], dtype=np.int16))
                total_bytes += even
            leftover = blob[even:]

    s.close()
    time.sleep(0.3)
    stream.stop(); stream.close()
    total_t = time.time() - t0
    audio_s = total_bytes / 2 / RATE
    print(f"  總 {total_t:.2f}s, audio {audio_s:.1f}s, ratio {total_t/audio_s:.2f}x")


if __name__ == "__main__":
    for i, text in enumerate(SAMPLES, 1):
        print(f"\n[{i}] {text[:40]}{'…' if len(text) > 40 else ''}")
        try:
            stream_and_play(text)
        except Exception as e:
            print(f"  ✗ {type(e).__name__}: {e}")
