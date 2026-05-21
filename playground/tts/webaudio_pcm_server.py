# SPDX-License-Identifier: Apache-2.0
"""Small WebAudio playground for raw PCM speech streaming."""

from __future__ import annotations

import argparse

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Higgs PCM Streaming Playground</title>
<style>
:root { color-scheme: light dark; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body { margin: 0; background: #111; color: #f4f4f4; }
main { max-width: 980px; margin: 0 auto; padding: 28px; }
h1 { font-size: 24px; font-weight: 650; margin: 0 0 18px; }
textarea { width: 100%; min-height: 150px; box-sizing: border-box; resize: vertical; font: inherit; padding: 12px; border-radius: 6px; border: 1px solid #444; background: #181818; color: #fff; }
.controls { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 12px; margin: 14px 0; }
label { display: grid; gap: 6px; font-size: 13px; color: #cfcfcf; }
input { font: inherit; padding: 8px; border-radius: 6px; border: 1px solid #444; background: #181818; color: #fff; }
button { font: inherit; padding: 10px 14px; border: 0; border-radius: 6px; color: #101010; background: #77d48b; cursor: pointer; }
button:disabled { opacity: .55; cursor: wait; }
.row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
#status { color: #cfcfcf; white-space: pre-wrap; }
a { color: #8dc7ff; }
a[hidden], audio[hidden] { display: none; }
audio { width: 100%; margin-top: 16px; }
</style>
</head>
<body>
<main>
<h1>Higgs PCM Streaming Playground</h1>
<textarea id="text">Today we are testing low latency speech streaming with raw PCM playback, steady pacing, and enough duration to reveal whether chunk boundaries sound clean.</textarea>
<div class="controls">
<label>Temperature<input id="temperature" type="number" value="0.3" min="0" max="2" step="0.05"></label>
<label>Top P<input id="top_p" type="number" value="0.95" min="0" max="1" step="0.05"></label>
<label>Max Tokens<input id="max_new_tokens" type="number" value="512" min="32" max="4096" step="32"></label>
<label>Queue Lead (ms)<input id="lead_ms" type="number" value="120" min="20" max="1000" step="10"></label>
</div>
<div class="row"><button id="start">Start PCM Stream</button><button id="stop" disabled>Stop</button><span id="status">Idle</span></div>
<a id="download" hidden download="higgs_stream_pcm.wav">Download final WAV</a>
<audio id="final" controls hidden></audio>
</main>
<script>
const startButton = document.getElementById('start');
const stopButton = document.getElementById('stop');
const statusEl = document.getElementById('status');
const finalAudio = document.getElementById('final');
const downloadEl = document.getElementById('download');
let controller = null;
let audioContext = null;
let scheduledTime = 0;
let firstAudioAt = null;
let startedAt = 0;
let chunkCount = 0;
let collected = [];
let sampleRate = 24000;

function setStatus(text) { statusEl.textContent = text; }
function bytesFromBase64(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}
function pcm16ToFloat32(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(bytes.byteLength / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.max(-1, view.getInt16(i * 2, true) / 32768);
  }
  return out;
}
function playPcm(bytes, sr) {
  if (!audioContext) audioContext = new AudioContext({ sampleRate: sr });
  if (audioContext.state === 'suspended') audioContext.resume();
  const floats = pcm16ToFloat32(bytes);
  const buffer = audioContext.createBuffer(1, floats.length, sr);
  buffer.copyToChannel(floats, 0);
  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(audioContext.destination);
  const lead = Number(document.getElementById('lead_ms').value || 120) / 1000;
  if (scheduledTime <= audioContext.currentTime) {
    scheduledTime = audioContext.currentTime + lead;
  }
  source.start(scheduledTime);
  scheduledTime += buffer.duration;
}
function makeWavBlob(chunks, sr) {
  const total = chunks.reduce((n, c) => n + c.byteLength, 0);
  const out = new ArrayBuffer(44 + total);
  const v = new DataView(out);
  const write = (off, s) => {
    for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i));
  };
  write(0, 'RIFF'); v.setUint32(4, 36 + total, true); write(8, 'WAVE');
  write(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, 1, true); v.setUint32(24, sr, true);
  v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true);
  v.setUint16(34, 16, true); write(36, 'data'); v.setUint32(40, total, true);
  let off = 44;
  for (const c of chunks) {
    new Uint8Array(out, off, c.byteLength).set(c);
    off += c.byteLength;
  }
  return new Blob([out], { type: 'audio/wav' });
}
async function handleEvent(data) {
  if (data === '[DONE]') return;
  const event = JSON.parse(data);
  if (event.audio && event.audio.data) {
    const bytes = bytesFromBase64(event.audio.data);
    sampleRate = event.audio.sample_rate || sampleRate;
    if (event.audio.format !== 'pcm') {
      throw new Error('Expected PCM, got ' + event.audio.format);
    }
    if (firstAudioAt === null) firstAudioAt = performance.now();
    chunkCount += 1;
    collected.push(bytes.slice().buffer);
    playPcm(bytes, sampleRate);
    const ttfa = ((firstAudioAt - startedAt) / 1000).toFixed(2);
    const queued = audioContext
      ? Math.max(0, scheduledTime - audioContext.currentTime).toFixed(2)
      : '0.00';
    setStatus('Streaming PCM | chunks ' + chunkCount + ' | TTFA ' + ttfa + 's | queued ' + queued + 's');
  } else if (event.finish_reason) {
    setStatus('Finished | chunks ' + chunkCount + ' | reason ' + event.finish_reason);
  }
}
startButton.onclick = async () => {
  startButton.disabled = true;
  stopButton.disabled = false;
  finalAudio.hidden = true;
  downloadEl.hidden = true;
  if (audioContext) await audioContext.close();
  audioContext = null;
  scheduledTime = 0;
  firstAudioAt = null;
  chunkCount = 0;
  collected = [];
  controller = new AbortController();
  startedAt = performance.now();
  setStatus('Connecting...');
  try {
    const payload = {
      input: document.getElementById('text').value,
      voice: 'default',
      response_format: 'pcm',
      stream: true,
      temperature: Number(document.getElementById('temperature').value),
      top_p: Number(document.getElementById('top_p').value),
      max_new_tokens: Number(document.getElementById('max_new_tokens').value)
    };
    const response = await fetch('/v1/audio/speech', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal
    });
    if (!response.ok) throw new Error(await response.text());
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const result = await reader.read();
      if (result.done) break;
      buffer += decoder.decode(result.value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop();
      for (const part of parts) {
        for (const line of part.split('\n')) {
          if (line.startsWith('data: ')) await handleEvent(line.slice(6).trim());
        }
      }
    }
    if (collected.length) {
      const blob = makeWavBlob(collected.map(x => new Uint8Array(x)), sampleRate);
      const url = URL.createObjectURL(blob);
      finalAudio.src = url;
      finalAudio.hidden = false;
      downloadEl.href = url;
      downloadEl.hidden = false;
    }
  } catch (err) {
    if (err.name !== 'AbortError') setStatus('Error: ' + err.message);
  } finally {
    startButton.disabled = false;
    stopButton.disabled = true;
    controller = null;
  }
};
stopButton.onclick = () => {
  if (controller) controller.abort();
  setStatus('Stopped');
};
</script>
</body>
</html>"""


def create_app(api_base: str) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    async def index():
        return HTMLResponse(HTML)

    @app.post("/v1/audio/speech")
    async def speech_proxy(request: Request):
        body = await request.body()
        headers = {
            "content-type": request.headers.get("content-type", "application/json")
        }

        async def stream():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{api_base.rstrip('/')}/v1/audio/speech",
                    content=body,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        yield chunk

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="WebAudio PCM TTS playground")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7864)
    args = parser.parse_args()

    uvicorn.run(
        create_app(args.api_base),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
