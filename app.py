# Dream Diffusion Model GGUF Inference for RTX 3050 8GB
# Improved Version: system prompt, stop button, logging, chat template, stats

import os
import sys
import subprocess
import time
import threading
import queue as queue_module
import re
import logging
import copy
import gradio as gr

# ==================== LOGGING ====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dream")

# ==================== CONFIGURATION ====================

LLAMA_DIFFUSION_CLI_PATH = r"D:\llama.cpp\build\bin\Release\llama-diffusion-cli.exe"
MODEL_PATH = r"D:\models\bartowski\Dream-org_Dream-v0-Instruct-7B-GGUF\Dream-org_Dream-v0-Instruct-7B-Q4_K_M.gguf"

def _check_paths():
    """Validate required files exist. Returns (ok, error_message)."""
    if not os.path.exists(MODEL_PATH):
        return False, f"Model not found at:\n  {MODEL_PATH}"
    if not os.path.exists(LLAMA_DIFFUSION_CLI_PATH):
        return False, f"llama-diffusion-cli.exe not found at:\n  {LLAMA_DIFFUSION_CLI_PATH}"
    return True, ""


def _get_gpu_name() -> str:
    """Return the name of the first CUDA GPU, or 'CPU' if none available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    # Fallback: wmic on Windows
    try:
        out = subprocess.check_output(
            ["wmic", "path", "win32_VideoController", "get", "Name", "/value"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        for line in out.splitlines():
            if line.startswith("Name=") and line.strip() != "Name=":
                return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return "Unknown GPU"


# Derive display strings at module load (fast — no file I/O)
MODEL_NAME = os.path.splitext(os.path.basename(MODEL_PATH))[0]  # e.g. Dream-org_Dream-v0-...-Q4_K_M
GPU_NAME   = _get_gpu_name()                                      # e.g. NVIDIA GeForce RTX 3050
log.info("Model : %s", MODEL_NAME)
log.info("GPU   : %s", GPU_NAME)

# ==================== ANSI / OUTPUT PARSING ====================

ANSI_ESCAPE = re.compile(r'\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def strip_ansi(text):
    """Remove ANSI escape codes from text."""
    return ANSI_ESCAPE.sub('', text)


def parse_diffusion_line_to_vis(line):
    """
    Convert a partial diffusion step line into (token, color) tuples.
    Revealed tokens are green; gaps between double-spaces become [MASK].
    """
    tokens = []
    parts = line.split(' ')
    for part in parts:
        stripped = part.strip()
        if stripped:
            tokens.append((stripped, "#66CC66"))    # green = revealed
        else:
            tokens.append(("[MASK]", "#555555"))    # gray  = masked
    # Remove leading/trailing pure-mask artifacts
    while tokens and tokens[0][0] == "[MASK]":
        tokens.pop(0)
    while tokens and tokens[-1][0] == "[MASK]":
        tokens.pop()
    return tokens if tokens else [("[MASK]...", "#555555")]

# ==================== PROMPT BUILDER ====================

def build_prompt_from_messages(messages):
    """
    Build prompt in the plain User:/Assistant: format that Dream-v0-Instruct-GGUF expects.
    (ChatML <|im_start|> tokens are NOT used by this GGUF variant.)
    """
    parts = []
    for msg in messages:
        role    = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    parts.append("Assistant:")
    return "\n".join(parts)

# ==================== HISTORY HELPERS ====================

def format_gradio_history_to_messages(history, system_prompt=""):
    """Convert Gradio [user, assistant] pairs to OpenAI-style message list."""
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    for pair in history:
        user_msg, assistant_msg = pair
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg is not None and assistant_msg != "":
            messages.append({"role": "assistant", "content": str(assistant_msg)})
    return messages

def add_user_message_to_gradio_history(history, message):
    if not history:
        history = []
    return history + [[message, None]]

# ==================== GENERATION ====================

def run_diffusion_generation(prompt, max_tokens, steps, temperature, top_p, top_k,
                             alg, alg_temp, step_callback=None, stop_event=None):
    """
    Run diffusion generation with real-time step callback.
    step_callback(step, total, partial_text) is called for every diffusion step.
    stop_event: threading.Event — set it to abort generation early.
    Returns (success, final_text, error_msg, elapsed_seconds).
    """
    alg_map = {"origin": 0, "entropy": 1, "margin": 2, "random": 3, "low_confidence": 4}

    prompt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_prompt.txt")
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    tk_arg = str(top_k) if isinstance(top_k, int) and top_k > 0 else "40"

    cmd = [
        LLAMA_DIFFUSION_CLI_PATH,
        "-m", MODEL_PATH,
        "-f", prompt_file,
        "-n", str(max_tokens),
        "--ctx-size", "4096",
        "--diffusion-steps", str(steps),
        "--temp", str(temperature),
        "--top-p", str(top_p),
        "--top-k", tk_arg,
        "--diffusion-eps", str(alg_temp),
        "--diffusion-alg-temp", str(alg_temp),
        "--diffusion-algorithm", str(alg_map.get(alg, 4)),
        "-ngl", "99",
        "-fa", "on",
        "--diffusion-visual",
    ]

    log.info("CMD: %s", " ".join(cmd))

    stderr_queue = queue_module.Queue()
    stdout_lines: list[str] = []

    def read_stderr(stream):
        try:
            for raw in iter(stream.readline, ''):
                stderr_queue.put(raw)
        except Exception:
            pass
        finally:
            stderr_queue.put(None)  # sentinel
            stream.close()

    def read_stdout(stream):
        """Collect all stdout lines — the CLI writes the final answer here."""
        try:
            for raw in iter(stream.readline, ''):
                line = strip_ansi(raw).rstrip()
                if line:
                    stdout_lines.append(line)
        except Exception:
            pass
        finally:
            stream.close()

    start_time = time.time()
    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,   # final answer arrives here
            stderr=subprocess.PIPE,   # progress / visualization on stderr
            text=True, encoding='utf-8', errors='ignore',
        )
        t_out = threading.Thread(target=read_stdout, args=(process.stdout,), daemon=True)
        t_err = threading.Thread(target=read_stderr, args=(process.stderr,), daemon=True)
        t_out.start()
        t_err.start()

        # ── Parse stderr in real-time ─────────────────────────────────
        step_re  = re.compile(r'diffusion step:\s*(\d+)/(\d+)')
        total_re = re.compile(r'total time:')

        current_step       = 0
        # Accumulate ALL lines between step headers (long outputs are multi-line)
        current_step_lines: list[str] = []
        last_text          = ""
        final_text         = ""
        in_step_body       = False
        generation_done    = False

        while True:
            # Abort if stop requested
            if stop_event and stop_event.is_set():
                log.info("Stop requested — killing process.")
                process.kill()
                return False, "", "Generation stopped by user.", time.time() - start_time

            try:
                raw = stderr_queue.get(timeout=0.1)
            except queue_module.Empty:
                if process.poll() is not None:
                    break
                if time.time() - start_time > 300:
                    process.kill()
                    return False, "", "Timeout (300 s).", time.time() - start_time
                continue

            if raw is None:
                break

            line = strip_ansi(raw).rstrip()
            if not line:
                continue

            # "diffusion step: N/M" — flush previous step's lines, start new
            m = step_re.search(line)
            if m:
                if current_step_lines:
                    last_text = " ".join(current_step_lines)
                current_step       = int(m.group(1))
                current_step_lines = []
                in_step_body       = True
                continue

            # "total time:" — flush final step, switch to final-answer mode
            if total_re.search(line):
                if current_step_lines:
                    last_text = " ".join(current_step_lines)
                generation_done = True
                in_step_body    = False
                continue

            # Skip loader lines ONLY before any diffusion step has started
            if not in_step_body and not generation_done and any(
                line.startswith(p) for p in (
                    "ggml_", "llama_", "load:", "load_tensors", "print_info",
                    "build:", "diffusion_params", "sched_", "llama_context",
                    "llama_model", "Device", "." * 5,
                )
            ):
                continue

            # Lines after "total time:" — may be the clean final answer on stderr
            if generation_done:
                final_text = (final_text + "\n" + line) if final_text else line
                continue

            # Lines within a diffusion step body (can be many for large outputs)
            if in_step_body:
                current_step_lines.append(line)
                if len(current_step_lines) == 1 and step_callback:
                    # Fire UI callback on first line so visualization updates live
                    step_callback(current_step, steps, line)
                continue

        process.wait(timeout=10)
        t_out.join(timeout=5)
        t_err.join(timeout=5)

    except Exception as e:
        import traceback; traceback.print_exc()
        return False, "", f"Exception: {e}", time.time() - start_time
    finally:
        if os.path.exists(prompt_file):
            try:
                os.remove(prompt_file)
            except OSError:
                pass

    elapsed = time.time() - start_time

    if process and process.returncode not in (0, None, -9):  # -9 = killed (stop)
        return False, "", f"CLI exited with code {process.returncode}", elapsed

    # Prefer stdout (full answer) over stderr visualization snippet (last_text)
    # or post-"total time:" stderr lines (final_text).
    stdout_text = "\n".join(stdout_lines).strip()
    result = stdout_text or last_text.strip() or final_text.strip()
    log.info("Final result (%d chars, %.1fs) [src=%s]: %s",
             len(result), elapsed,
             "stdout" if stdout_text else ("last_text" if last_text else "final_text"),
             repr(result[:120]))
    return True, result, "", elapsed

# ==================== GENERATION WITH VISUALIZATION ====================

def dream_generate_with_visualization(history, system_prompt, max_new_tokens, steps,
                                      temperature, top_p, top_k, delay, alg, alg_temp,
                                      stop_event):
    log.info("Parameters: tokens=%d steps=%d temp=%.2f", max_new_tokens, steps, temperature)

    messages = format_gradio_history_to_messages(history, system_prompt)
    prompt   = build_prompt_from_messages(messages)
    log.debug("Prompt:\n%s", prompt)

    vis_state = {
        "step": 0, "total": steps, "text": "", "vis": [],
        "done": False, "error": None, "result": "", "elapsed": 0.0,
    }
    vis_lock = threading.Lock()

    def step_callback(step, total, partial_text):
        vis_tokens = parse_diffusion_line_to_vis(partial_text)
        with vis_lock:
            vis_state["step"]  = step
            vis_state["total"] = total
            vis_state["text"]  = partial_text
            vis_state["vis"]   = vis_tokens

    def generation_thread():
        try:
            success, result, error, elapsed = run_diffusion_generation(
                prompt, max_new_tokens, steps, temperature,
                top_p, top_k, alg, alg_temp,
                step_callback=step_callback,
                stop_event=stop_event,
            )
            with vis_lock:
                vis_state["elapsed"] = elapsed
                if success:
                    vis_state["result"] = result
                else:
                    vis_state["error"] = error
        except Exception as e:
            with vis_lock:
                vis_state["error"] = str(e)
        finally:
            with vis_lock:
                vis_state["done"] = True

    gen_thread = threading.Thread(target=generation_thread)
    gen_thread.start()

    # ── Yield loop ────────────────────────────────────────────────────
    last_yielded_step    = -1
    intermediate_history = copy.deepcopy(history)

    while True:
        with vis_lock:
            done   = vis_state["done"]
            step   = vis_state["step"]
            total  = vis_state["total"]
            vis    = vis_state["vis"]
            error  = vis_state["error"]
            result = vis_state["result"]

        if step != last_yielded_step and vis:
            last_yielded_step = step
            pct = int(100 * step / max(total, 1))
            intermediate_history[-1][1] = f"⏳ Step {step}/{total} ({pct}%)"
            yield (
                format_gradio_history_to_messages(intermediate_history, system_prompt),
                vis,
                history,
                "",           # stats placeholder while running
            )

        if done:
            break
        time.sleep(delay)

    gen_thread.join()

    elapsed = vis_state["elapsed"]

    if vis_state["error"]:
        err = vis_state["error"]
        history[-1][1] = f"❌ {err[:300]}"
        stats_md = f"❌ **Failed** after `{elapsed:.1f}s`"
        yield (
            format_gradio_history_to_messages(history, system_prompt),
            [(err[:80], "#CC6666")],
            history,
            stats_md,
        )
        return

    final_text = vis_state["result"] or "(no output)"
    history[-1][1] = final_text

    # Final visualization: first 5 green (new), rest blue (context)
    final_vis  = []
    all_words  = final_text.split()
    for i, word in enumerate(all_words[:60]):
        final_vis.append((word, "#66CC66" if i < 5 else "#6699CC"))
    if len(all_words) > 60:
        final_vis.append(("...", "#888888"))

    tps = steps / elapsed if elapsed > 0 else 0
    stats_md = (
        f"✅ **Done** &nbsp;|&nbsp; "
        f"⏱ `{elapsed:.1f}s` &nbsp;|&nbsp; "
        f"🔢 `{steps}` steps &nbsp;|&nbsp; "
        f"📝 `{len(all_words)}` words"
    )

    yield (
        format_gradio_history_to_messages(history, system_prompt),
        final_vis,
        history,
        stats_md,
    )
    log.info("Done. (%.1fs)", elapsed)


# Module-level stop event — safe because concurrency_limit=1 (one generation at a time).
# threading.Event cannot be serialized through gr.State in Gradio 6.x.
_active_stop_event: threading.Event | None = None


def bot_response_generator(history, system_prompt, max_new_tokens, steps,
                           temperature, top_p, top_k, delay, alg, alg_temp):
    global _active_stop_event
    if not history or history[-1][1] is not None:
        log.debug("Skipping: no pending message.")
        yield format_gradio_history_to_messages(history, system_prompt), [], history, ""
        return

    # Create a fresh stop event for this run
    stop_event = threading.Event()
    _active_stop_event = stop_event

    yield from dream_generate_with_visualization(
        history, system_prompt, max_new_tokens, steps,
        temperature, top_p, top_k, delay, alg, alg_temp,
        stop_event,
    )

    _active_stop_event = None


def user_message_submitted(message, history):
    if not message or not message.strip():
        return history, gr.update(), ""
    new_history       = add_user_message_to_gradio_history(history, message.strip())
    messages_for_ui   = format_gradio_history_to_messages(new_history)
    return new_history, messages_for_ui, ""


def retry_last_message(history):
    """Remove the last assistant reply so it gets regenerated."""
    if not history:
        return history, format_gradio_history_to_messages(history)
    # If the last pair already has a reply, clear it; otherwise leave as-is
    if history[-1][1] is not None:
        history = copy.deepcopy(history)
        history[-1][1] = None
    return history, format_gradio_history_to_messages(history)


# ==================== GRADIO UI ====================

css = """
/* ── Global ──────────────────────────────── */
body, .gradio-container {
    background: #0f1117 !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ── Header gradient ─────────────────────── */
#dream-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 8px;
    border: 1px solid #2a2a4a;
}
#dream-header h1 { margin: 0; font-size: 1.6rem; }
#dream-header p  { margin: 4px 0 0; color: #9aa5c0; font-size: 0.9rem; }

/* ── Chatbot ─────────────────────────────── */
#chatbot-display {
    border: 1px solid #2a2a4a !important;
    border-radius: 10px !important;
    background: #0d1117 !important;
}

/* ── Visualization panel ─────────────────── */
#vis-panel {
    font-family: 'Fira Code', 'Consolas', monospace !important;
    background: #0d1117 !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 10px !important;
}

/* ── Stats bar ───────────────────────────── */
#stats-display {
    background: #12181f !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 8px !important;
    padding: 6px 12px !important;
    font-size: 0.85rem !important;
    color: #8abeff !important;
    min-height: 36px;
}

/* ── Buttons ─────────────────────────────── */
#send-btn    { background: linear-gradient(135deg, #0f3460, #533483) !important; }
#stop-btn    { background: #5c1a1a !important; color: #ff8080 !important; }
#retry-btn   { background: #1a3a1a !important; color: #88ff88 !important; }
#clear-btn   { background: #1a1a2e !important; color: #9aa5c0 !important; }

/* ── Input textbox ───────────────────────── */
#user-input textarea {
    background: #0d1117 !important;
    border: 1px solid #2a3a5a !important;
    color: #e0e0e0 !important;
    border-radius: 8px !important;
}

/* ── System prompt ───────────────────────── */
#system-prompt textarea {
    background: #0d1117 !important;
    border: 1px solid #2a3a3a !important;
    color: #c0d8c0 !important;
    font-size: 0.85rem !important;
}

/* ── Accordion ───────────────────────────── */
.accordion { background: #0f1117 !important; border: 1px solid #2a2a4a !important; }

/* ── Slider labels ───────────────────────── */
.gradio-slider label { color: #9aa5c0 !important; }

/* ── Selection highlight ─────────────────── */
::selection, ::-moz-selection { background-color: #3a4a7a; }
"""

with gr.Blocks() as demo:

    # ── Header ──────────────────────────────────────────────────────
    gr.HTML(f"""
    <div id="dream-header">
        <h1>🧠 Dream Diffusion — GGUF Inference</h1>
        <p><code>{MODEL_NAME}</code> &nbsp;·&nbsp; {GPU_NAME}
           &nbsp;·&nbsp; Real-time diffusion visualization</p>
    </div>
    """)

    # ── State ────────────────────────────────────────────────────────
    chat_history_state  = gr.State([])

    # ── Main layout ──────────────────────────────────────────────────
    with gr.Row():
        # Left — chat
        with gr.Column(scale=3):
            chatbot_display = gr.Chatbot(
                elem_id="chatbot-display",
                label="Conversation",
                height=520,
            )

            with gr.Group():
                with gr.Row():
                    user_input_textbox = gr.Textbox(
                        elem_id="user-input",
                        placeholder="Type your message and press Enter…",
                        show_label=False,
                        container=False,
                        scale=5,
                        lines=1,
                        max_lines=4,
                    )
                    send_button = gr.Button("Send ▶", elem_id="send-btn",
                                           scale=1, variant="primary")

            with gr.Row():
                stop_button  = gr.Button("⏹ Stop",  elem_id="stop-btn",  scale=1)
                retry_button = gr.Button("🔁 Retry", elem_id="retry-btn", scale=1)
                clear_button = gr.Button("🗑 Clear",  elem_id="clear-btn", scale=1)

        # Right — visualization + stats
        with gr.Column(scale=2):
            vis_output_display = gr.HighlightedText(
                elem_id="vis-panel",
                label="Diffusion Process Visualization",
                show_legend=True,
                combine_adjacent=False,
                color_map={
                    "[MASK]": "#555555",
                    "token":  "#66CC66",
                    "word":   "#6699CC",
                },
            )
            stats_display = gr.Markdown(
                elem_id="stats-display",
                value="*Waiting for generation…*",
            )

    # ── System Prompt ────────────────────────────────────────────────
    with gr.Accordion("🧩 System Prompt", open=False):
        system_prompt_input = gr.Textbox(
            elem_id="system-prompt",
            label="System Prompt",
            placeholder="e.g. You are a concise and helpful assistant.",
            lines=3,
            value="",
        )

    # ── Generation Parameters ────────────────────────────────────────
    with gr.Accordion("⚙️ Generation Parameters", open=True):
        gr.Markdown("**Optimised defaults for RTX 3050 8 GB VRAM**")

        with gr.Row():
            max_new_tokens_slider = gr.Slider(
                16, 1024, value=128, step=16, label="Max New Tokens")
            steps_slider = gr.Slider(
                8, 128, value=16, step=8, label="Diffusion Steps")

        with gr.Row():
            temperature_slider = gr.Slider(
                0.0, 2.0, value=0.05, step=0.05, label="Temperature")
            top_p_slider = gr.Slider(
                0.0, 1.0, value=0.95, step=0.05, label="Top-p")

        with gr.Row():
            top_k_slider = gr.Slider(
                0, 100, value=0, step=1, label="Top-k (0 = auto)")
            delay_slider = gr.Slider(
                0.0, 0.5, value=0.05, step=0.01, label="Visualization Delay (s)")

        with gr.Row():
            alg_dropdown = gr.Dropdown(
                choices=["origin", "entropy", "margin", "random", "low_confidence"],
                value="low_confidence",
                label="Diffusion Algorithm",
            )
            alg_temp_slider = gr.Slider(
                0.0, 1.0, value=0.1, step=0.01, label="Algorithm Temperature")

    # ── Callbacks ────────────────────────────────────────────────────

    generation_params = [
        system_prompt_input,
        max_new_tokens_slider, steps_slider, temperature_slider,
        top_p_slider, top_k_slider, delay_slider, alg_dropdown, alg_temp_slider,
    ]

    def clear_conversation():
        return [], [], "", [], "*Waiting for generation…*"

    clear_button.click(
        fn=clear_conversation,
        inputs=[],
        outputs=[
            chat_history_state, chatbot_display,
            user_input_textbox, vis_output_display, stats_display,
        ],
        queue=False,
    )

    def stop_generation():
        ev = _active_stop_event
        if ev is not None:
            ev.set()
            log.info("Stop event set.")
        return "*⏹ Stopping…*"

    stop_button.click(
        fn=stop_generation,
        inputs=[],
        outputs=[stats_display],
        queue=False,
    )

    # Retry: clear last assistant reply, then re-trigger bot
    retry_button.click(
        fn=retry_last_message,
        inputs=[chat_history_state],
        outputs=[chat_history_state, chatbot_display],
        queue=False,
    ).then(
        fn=bot_response_generator,
        inputs=[chat_history_state] + generation_params,
        outputs=[chatbot_display, vis_output_display, chat_history_state, stats_display],
    )

    submit_event_args = dict(
        fn=user_message_submitted,
        inputs=[user_input_textbox, chat_history_state],
        outputs=[chat_history_state, chatbot_display, user_input_textbox],
    )

    bot_response_event_args = dict(
        fn=bot_response_generator,
        inputs=[chat_history_state] + generation_params,
        outputs=[chatbot_display, vis_output_display, chat_history_state, stats_display],
    )

    clear_vis_fn = lambda: ([], "*Generating…*")

    # Enter key
    submit_action = user_input_textbox.submit(**submit_event_args)
    submit_action.then(clear_vis_fn, inputs=None,
                       outputs=[vis_output_display, stats_display])
    submit_action.then(**bot_response_event_args)

    # Send button
    send_action = send_button.click(**submit_event_args)
    send_action.then(clear_vis_fn, inputs=None,
                     outputs=[vis_output_display, stats_display])
    send_action.then(**bot_response_event_args)


# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    print("=" * 70)
    print("  Dream Diffusion Model — GGUF Inference (RTX 3050 8GB)")
    print("=" * 70)
    print(f"  Model : {MODEL_PATH}")
    print(f"  CLI   : {LLAMA_DIFFUSION_CLI_PATH}")
    print("-" * 70)

    ok, err = _check_paths()
    if not ok:
        print(f"\n❌ ERROR: {err}")
        print("\nPlease update LLAMA_DIFFUSION_CLI_PATH / MODEL_PATH at the top of app.py.")
        sys.exit(1)

    print("\n✓ All checks passed!")
    print("  Starting Gradio server → http://localhost:7860")
    print("=" * 70)

    demo.queue(max_size=10, default_concurrency_limit=1).launch(
        share=False,
        debug=False,
        show_error=True,
        theme=gr.themes.Base(),
        css=css,
    )
