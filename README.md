<h1 align="center">Yuna</h1>

<p align="center">
A voice AI companion with a Live2D avatar that answers fast and can also do things for you.<br>
Built on <a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber">Open-LLM-VTuber</a>.
</p>

Yuna starts talking with a fast model while a router decides, at the same moment, whether you are chatting or asking her to do something. Jobs go to a [Hermes agent](https://github.com/NousResearch/hermes-agent) in the background, and she reads out the result once the conversation goes quiet. You can keep talking to her while tasks run, check on them, change them, cancel them, or approve a command out loud.

## What's different from Open-LLM-VTuber

- **Real-time agent** (`realtime_agent`): a fast talker model always answers, and Jev, OpenRouter's decision model, routes each turn to chat or to a background task.
- **Background tasks**: Hermes runs start, steer, stop and ask for permission by voice, and finished results are announced when you stop talking.
- **Low first-audio latency**: pre-rendered acknowledgements and openers ("Hmph,", "Oh?", "Well,"), a backup provider if the talker is slow to start, and a shorter first sentence.
- **Filler voices**: optional short "hmm" or "okay" clips play the moment you stop speaking, while the reply is still being generated.
- **Azure TTS over REST** with an edge-tts fallback voice when Azure fails or is rate-limited, plus pronunciation fixes.
- **Fixes**: spaces kept in saved replies, TTS kept off the event loop, and websocket sends that are safe to run concurrently.

Everything else from Open-LLM-VTuber still works: other agents and LLMs, ASR and TTS backends, the web and desktop clients, Live2D expressions and chat history.

## How one turn works

Your words go to the talker and to Jev at the same time. The talker's reply streams into a holding buffer. When Jev says "chat", the buffer is released and spoken. When Jev says "task", the buffer is thrown away, a Hermes run starts, and Yuna plays a short acknowledgement that was rendered in advance.

![One turn: speech goes to the talker and the Jev router at once; the held reply is spoken on chat or dropped for a task, which starts a Hermes run](assets/yuna/one-turn.svg)

The talker is never blocked on routing. It has been streaming for the whole time Jev was deciding, so a chat turn loses nothing by waiting for the route.

### What each route does

Unsure, change, cancel and approval turns get a short talker reply (at most 80 tokens) steered by a system instruction placed after your message.

| Route | Held reply | What Yuna says | What happens to work |
| --- | --- | --- | --- |
| chat | spoken | The talker's reply as it streamed | Nothing |
| follow-up · status | spoken | The talker's reply; task state is already in its prompt | Result marked as told |
| new task | dropped | A pre-rendered acknowledgement | `POST /v1/runs` to Hermes |
| follow-up · change | dropped | Short reply confirming the change | Steer the run; if Hermes refuses, stop it and start again with the update |
| follow-up · cancel | dropped | Short reply confirming the stop | `POST /v1/runs/{id}/stop` |
| unsure | dropped | Asks whether you want it done | Your request is saved as a pending offer |
| accept offer | dropped | A pre-rendered acknowledgement | Starts the saved request plus what you just said |
| approval | dropped | Short reply confirming yes or no | `POST /v1/runs/{id}/approval` with `once` or `deny`. A yes needs Jev at 0.8 or higher; a no counts at any confidence |

### Background tasks

Each Hermes run has a watcher that follows its event stream and falls back to polling. Running and finished tasks become one-line entries in every talker prompt and in Jev's view of active tasks. When a run finishes, the task announcer waits until no turn is running and nothing has happened for a second, then starts a turn that reads the result aloud. A run that pauses for permission goes through the same announcer, and Yuna asks you out loud; your next turn answers it.

![Background tasks: the task manager drives Hermes runs, the announcer waits for quiet, and the talker reads the result](assets/yuna/background-tasks.svg)

Results that finish while no client is connected are announced when one connects.

### Why routing costs nothing on chat turns

Jev's typical answer arrives before the talker's first speakable clause.

![Timing of a chat turn: Jev routes at 0.37 s median, the talker's first clause arrives at 0.62 s median, and the opener clip starts between 0.40 and 0.62 s](assets/yuna/chat-timing.svg)

Medians from a latency benchmark, with the opener range from live turns. End to end, first audio was about 2.0 s median with edge-tts, which took 0.7–1.2 s of that.

## Quick start

You need [uv](https://docs.astral.sh/uv/), an [OpenRouter](https://openrouter.ai) API key, and a running [hermes-agent](https://github.com/NousResearch/hermes-agent) API server for background tasks.

```bash
git clone --recursive https://github.com/Sero01/Yuna.git
cd Yuna
uv sync
cp config_templates/conf.default.yaml conf.yaml
```

In `conf.yaml`:

1. Set `conversation_agent_choice: 'realtime_agent'`.
2. Provide your keys, either in `agent_settings.realtime_agent` or as the `OPENROUTER_API_KEY` and `HERMES_API_KEY` environment variables. `HERMES_API_KEY` is Hermes' `API_SERVER_KEY`.
3. Optional: point `soul_path` at a persona file (for example Hermes' `SOUL.md`) and `user_profile_path` at facts about you (for example `memories/USER.md`).
4. Optional: switch `tts_model` to `azure_tts` and set `edge_fallback_voice`, add `openers`, or turn on `filler_config.enabled`.

Then start the server and open http://localhost:12393:

```bash
uv run run_server.py
```

For ASR, TTS, Live2D models and the desktop client, the [Open-LLM-VTuber docs](https://open-llm-vtuber.github.io/docs/quick-start) still apply.

## Settings

Everything lives under `character_config.agent_config.agent_settings.realtime_agent` in `conf.yaml`. Set `conversation_agent_choice` back to `basic_memory_agent` to use a single LLM instead.

| Setting | Default | Effect |
| --- | --- | --- |
| `talker_model` | `deepseek/deepseek-v4.1-flash` | The model that always answers. Temperature 0.8, 200 tokens max |
| `talker_provider` | `''` | Pin one OpenRouter provider (prompt caching is per provider); empty means fastest |
| `hedge_after_s` | `1.5` | Starts a second request on a different provider if no token has arrived |
| `jev_model` | `typesafe/jev-1.13` | Called through OpenRouter's alpha decisions endpoint |
| `jev_fallback_after_s` | `0.6` | Races the Groq fallback router; the first usable answer wins, Jev on a tie |
| `jev_timeout_s` | `1.5` | Jev is abandoned after this; with no answer at all, the turn is treated as chat |
| `unsure_low` / `unsure_high` | `0.35` / `0.65` | Task probability band where Yuna asks instead of acting |
| `max_active_tasks` | `3` | Concurrent Hermes runs; `task_timeout_s` is 600 |
| `max_turns` | `8` | Exchanges of history in the talker prompt |
| `quiet_gap_s` | `1.0` | Silence required before a result is announced |
| `openers` | `[]` | Interjections like `'Hmph,'`, `'Oh?'`, `'Well,'`, each cut from a carrier sentence and cached as a clip |
| `prerender_acks` | `True` | Acknowledgements are generated and synthesized before they are needed |

## Where the code lives

Paths are under `src/open_llm_vtuber/`.

| File | What it does |
| --- | --- |
| `agent/agents/realtime/realtime_agent.py` | Turn orchestration, held reply, opener split, speech pipeline |
| `agent/agents/realtime/router.py` | Jev router and the Groq fallback race |
| `agent/agents/realtime/talker.py` | Streaming OpenRouter client with provider hedging |
| `agent/agents/realtime/context.py` | Memory window and prompt assembly |
| `agent/agents/realtime/tasks.py` | Hermes Runs API client, approvals and task status lines |
| `agent/agents/realtime/ack_pool.py` | Pre-generated, pre-rendered acknowledgements |
| `agent/agents/realtime/openers.py` | Opener clips cut from a carrier sentence |
| `agent/agents/realtime/prompts.py` | Instructions for acks, unsure, change, cancel, approvals and results |
| `conversations/task_announcer.py` | Waits for quiet, then starts the task-result turn |
| `conversations/filler.py` | Filler voice clips |

## Tests

```bash
uv run python tests/run_all.py
```

## Credits and license

Yuna is a fork of [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber) by Yi-Ting Chiu and contributors, released under the [MIT license](LICENSE). The web frontend is the upstream [Open-LLM-VTuber-Web](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber-Web) submodule.

### Live2D sample models

This project includes Live2D sample models provided by Live2D Inc. These assets are licensed separately under the Live2D Free Material License Agreement and the Terms of Use for Live2D Cubism Sample Data. They are not covered by the MIT license of this project.

This content uses sample data owned and copyrighted by Live2D Inc. The sample data are utilized in accordance with the terms and conditions set by Live2D Inc. (See [Live2D Free Material License Agreement](https://www.live2d.jp/en/terms/live2d-free-material-license-agreement/) and [Terms of Use](https://www.live2d.com/eula/live2d-sample-model-terms_en.html)).

Note: For commercial use, especially by medium or large-scale enterprises, the use of these Live2D sample models may be subject to additional licensing requirements. If you plan to use this project commercially, please ensure that you have the appropriate permissions from Live2D Inc., or use versions of the project without these models.
