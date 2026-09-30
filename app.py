"""
Live browser demo for the Flappy Bird DQN agent.

Runs the trained policy headlessly (no real display needed) and streams
the gameplay to any connected browser as an MJPEG feed, with a small
JSON endpoint for the live episode/score counters. One shared game runs
in the background; every visitor watches the same live session.

Local run:      uvicorn app:app --reload
Render start:   uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import os

# Must happen before pygame / flappy_bird_gymnasium are imported anywhere,
# since there is no real display on a Render instance.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import asyncio
import io
import threading
import time

import gymnasium as gym
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from PIL import Image

import flappy_bird_gymnasium  # noqa: F401  (registers the FlappyBird-v0 env)
from dqn import DQN

MODEL_FILE = os.environ.get("MODEL_FILE", "model/flappybirdv0.pt")
TARGET_FPS = 30
JPEG_QUALITY = 80


class GameStream:
    """Runs the trained agent in a background thread and holds the latest frame."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._stats = {"episode": 0, "score": 0, "best_score": 0, "status": "starting"}
        self._started = False

    def start(self):
        if self._started:
            return
        self._started = True
        threading.Thread(target=self._run, daemon=True).start()

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def _run(self):
        device = "cpu"  # tiny 2-layer MLP; a GPU would only add overhead here
        env = gym.make("FlappyBird-v0", render_mode="rgb_array")
        policy = DQN(env.observation_space.shape[0], env.action_space.n).to(device)

        loaded = os.path.exists(MODEL_FILE)
        if loaded:
            try:
                policy.load_state_dict(torch.load(MODEL_FILE, map_location=device))
            except Exception as exc:  # pragma: no cover - defensive only
                print(f"[app] could not load {MODEL_FILE}: {exc}")
                loaded = False
        else:
            print(f"[app] no checkpoint at {MODEL_FILE}; playing with random weights")
        policy.eval()

        episode = 0
        best_score = 0
        frame_interval = 1.0 / TARGET_FPS

        while True:
            episode += 1
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)
            terminated = False
            score = 0

            with self._lock:
                self._stats.update(
                    episode=episode,
                    score=0,
                    best_score=best_score,
                    status="playing" if loaded else "playing (untrained checkpoint)",
                )

            while not terminated:
                tick_start = time.time()

                with torch.no_grad():
                    action = policy(state.unsqueeze(0)).squeeze().argmax().item()

                next_state, reward, terminated, _, _ = env.step(action)
                if reward is not None and reward >= 1.0:
                    score += 1

                frame = env.render()
                jpeg = self._encode_jpeg(frame)

                with self._lock:
                    self._latest_jpeg = jpeg
                    self._stats["score"] = score
                    if score > best_score:
                        best_score = score
                        self._stats["best_score"] = best_score

                state = torch.tensor(next_state, dtype=torch.float32, device=device)

                elapsed = time.time() - tick_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

            with self._lock:
                self._stats["status"] = "restarting"
            time.sleep(0.8)  # let viewers register the crash before the next round starts

    @staticmethod
    def _encode_jpeg(frame) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=JPEG_QUALITY)
        return buf.getvalue()


game = GameStream()
app = FastAPI(title="Flappy Bird DQN — live demo")


@app.on_event("startup")
async def _startup():
    game.start()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return JSONResponse(game.get_stats())


async def _mjpeg_generator():
    boundary = b"--frame"
    delay = 1.0 / TARGET_FPS
    while True:
        frame = game.get_frame()
        if frame is not None:
            yield (
                boundary
                + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(frame)).encode()
                + b"\r\n\r\n"
                + frame
                + b"\r\n"
            )
        await asyncio.sleep(delay)


@app.get("/stream")
def stream():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, private", "Pragma": "no-cache"},
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Flappy Bird DQN — live demo</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🐦</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root{
    --bg: #12141a;
    --surface: #1c1f29;
    --border: #2a2e3b;
    --text: #e8e6e0;
    --text-muted: #8b8e9c;
    --accent: #e3a857;
    --accent-soft: rgba(227,168,87,0.14);
    --score: #7fb88f;
  }
  *{ box-sizing: border-box; }
  body{
    margin:0;
    background:var(--bg);
    color:var(--text);
    font-family:'IBM Plex Mono', ui-monospace, monospace;
    display:flex;
    justify-content:center;
    padding:56px 20px 40px;
    line-height:1.6;
  }
  .page{ width:100%; max-width:420px; }
  h1{
    font-family:'Fraunces', Georgia, serif;
    font-weight:600;
    font-size:clamp(28px, 5vw, 36px);
    line-height:1.15;
    margin:0 0 10px;
    letter-spacing:-0.01em;
  }
  .subhead{
    color:var(--text-muted);
    font-size:14px;
    margin:0 0 36px;
    max-width:38ch;
  }
  .viewport-wrap{ position:relative; }
  .viewport{
    position:relative;
    border-radius:14px;
    overflow:hidden;
    border:1px solid var(--border);
    box-shadow: 0 0 0 1px var(--accent-soft), 0 20px 60px -20px rgba(0,0,0,0.6);
    background:var(--surface);
    animation: glow 3.2s ease-in-out infinite;
  }
  @keyframes glow{
    0%, 100% { box-shadow: 0 0 0 1px var(--accent-soft), 0 20px 60px -20px rgba(0,0,0,0.6); }
    50%      { box-shadow: 0 0 0 1px rgba(227,168,87,0.32), 0 20px 60px -20px rgba(0,0,0,0.6); }
  }
  .viewport img{ display:block; width:100%; height:auto; }
  .live-badge{
    position:absolute; top:12px; left:12px;
    display:flex; align-items:center; gap:6px;
    background:rgba(18,20,26,0.72);
    border:1px solid var(--border);
    border-radius:20px;
    padding:4px 10px 4px 8px;
    font-size:11px;
    color:var(--text);
    backdrop-filter: blur(4px);
  }
  .live-badge .dot{
    width:6px; height:6px; border-radius:50%;
    background:#e0685c;
    animation: pulse 1.6s ease-in-out infinite;
  }
  @keyframes pulse{
    0%, 100% { opacity:1; }
    50%      { opacity:0.35; }
  }
  .telemetry{
    display:flex;
    justify-content:space-between;
    margin-top:18px;
    padding:0 4px;
  }
  .stat{ text-align:left; }
  .stat-value{
    display:block;
    font-size:20px;
    font-weight:500;
    color:var(--text);
  }
  .stat[data-kind="score"] .stat-value{ color:var(--score); }
  .stat[data-kind="best"] .stat-value{ color:var(--accent); }
  .stat-label{
    display:block;
    font-size:11px;
    color:var(--text-muted);
    margin-top:2px;
  }
  .about{
    margin-top:40px;
    padding-top:28px;
    border-top:1px solid var(--border);
    color:var(--text-muted);
    font-size:13px;
  }
  .about p{ margin:0 0 12px; }
  .about strong{ color:var(--text); font-weight:500; }
  footer{
    margin-top:32px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    flex-wrap:wrap;
    gap:12px;
    font-size:12px;
    color:var(--text-muted);
  }
  footer .links{ display:flex; gap:10px; }
  footer a{
    color:var(--text);
    text-decoration:none;
    border:1px solid var(--border);
    border-radius:8px;
    padding:6px 12px;
    transition: border-color 0.15s ease, color 0.15s ease;
  }
  footer a:hover{ border-color:var(--accent); color:var(--accent); }
</style>
</head>
<body>
  <div class="page">
    <h1>Flappy Bird, played by a neural network</h1>
    <p class="subhead">A Deep Q-Network trained from raw LIDAR readings — no rules coded in, just reward and error.</p>

    <div class="viewport-wrap">
      <div class="viewport">
        <img id="stream" src="/stream" alt="Live Flappy Bird DQN gameplay" onerror="setTimeout(()=>{this.src='/stream?'+Date.now()}, 2000)">
        <div class="live-badge"><span class="dot"></span>live</div>
      </div>
      <div class="telemetry">
        <div class="stat" data-kind="episode"><span class="stat-value" id="episode">–</span><span class="stat-label">episode</span></div>
        <div class="stat" data-kind="score"><span class="stat-value" id="score">–</span><span class="stat-label">pipes, this run</span></div>
        <div class="stat" data-kind="best"><span class="stat-value" id="best">–</span><span class="stat-label">best so far</span></div>
      </div>
    </div>

    <div class="about">
      <p><strong>What it's looking at:</strong> 180 simulated LIDAR rays plus the bird's own position and velocity — no pixels, just distances.</p>
      <p><strong>How it decided to flap:</strong> a small feed-forward network trained with experience replay and a target network, the way DQN was originally set up for Atari.</p>
      <p>It's still mid-training, so expect real pipes cleared and the occasional faceplant into the first one — both are the same learning process.</p>
    </div>

    <footer>
      <span>Built by Abhishek Grover</span>
      <span class="links">
        <a href="https://github.com/AbhishekGrover1" target="_blank" rel="noopener">GitHub</a>
        <a href="https://www.linkedin.com/in/abhishek-grover07/" target="_blank" rel="noopener">LinkedIn</a>
      </span>
    </footer>
  </div>

<script>
  async function pollStats(){
    try{
      const res = await fetch('/stats', {cache:'no-store'});
      const data = await res.json();
      document.getElementById('episode').textContent = data.episode;
      document.getElementById('score').textContent = data.score;
      document.getElementById('best').textContent = data.best_score;
    }catch(e){ /* transient network hiccup, next poll will retry */ }
  }
  pollStats();
  setInterval(pollStats, 1000);
</script>
</body>
</html>
"""
