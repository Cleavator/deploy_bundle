import gradio as gr
import numpy as np
import os
import random
import cv2
import sys
import time
from crc_qa_v2 import answer_crc_question
from drop.drop_inference import full_droplet_analysis

print(f"[BOOT] pid={os.getpid()}")
print(f"[BOOT] python={sys.executable}")
print(f"[BOOT] cwd={os.getcwd()}")
print(f"[BOOT] gradio={gr.__version__}")
print(f"[BOOT] time={time.strftime('%Y-%m-%d %H:%M:%S')}")

try:
    import torch
    import cellpose
    print(f"[BOOT] torch version: {torch.__version__}")
    print(f"[BOOT] torch.backends.mps.is_available(): {torch.backends.mps.is_available()}")
    print(f"[BOOT] torch.cuda.is_available(): {torch.cuda.is_available()}")
except ImportError:
    print("[BOOT] torch or cellpose not found/importable at boot time")

# --- Question Banks ---
HEALTHY_QUESTIONS = [
    "What are effective dietary habits to prevent colorectal cancer?",
    "How does physical exercise reduce the risk of colorectal cancer?",
    "What are the recommended screening guidelines for colorectal cancer in healthy individuals?",
    "What lifestyle changes can lower the risk of developing colorectal cancer?"
]

DISEASE_QUESTIONS = [
    "What are the standard clinical treatments for colorectal cancer?",
    "Please explain the difference between chemotherapy and targeted therapy for CRC.",
    "What are the common side effects of colorectal cancer treatments?",
    "What follow-up care is required after colorectal cancer surgery?"
]

def add_user_message(message, history):
    """
    Append user message to history and clear the textbox.
    """
    if history is None:
        history = []
    if message is None or message.strip() == "":
        return "", history
    return "", history + [[message, None]]

def normalize_chat_history(history):
    """Coerce chat history into a list of [user, bot] pairs across Gradio formats."""
    if not history:
        return []

    # Gradio may pass tuples, lists, or message dicts depending on version
    if isinstance(history, tuple):
        history = list(history)

    normalized = []
    pending_user = None

    for item in history:
        if isinstance(item, dict) and "role" in item:
            role = item.get("role")
            content = item.get("content")

            # Newer Gradio may wrap content as a list of parts; flatten to text
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )

            text = content if isinstance(content, str) else str(content)
            if role == "user":
                pending_user = text
            elif role == "assistant":
                normalized.append([pending_user, text])
                pending_user = None
            continue

        if isinstance(item, (list, tuple)):
            if len(item) >= 2:
                normalized.append([item[0], item[1]])
            continue

        if isinstance(item, str):
            normalized.append([item, None])

    if pending_user is not None:
        normalized.append([pending_user, None])

    return normalized

def generate_response(history, api_key):
    """
    Debug version (Non-generator) to test Gradio wiring.
    """
    print(f"[DEBUG] generate_response called. History type: {type(history)}")
    print(f"[DEBUG] API Key received: {'Yes' if api_key else 'No'}")
    
    if history is None:
        print("[DEBUG] History is None")
        return []
        
    if not history:
        print("[DEBUG] History is empty")
        return history

    user_message = history[-1][0]
    print(f"[DEBUG] User message: {user_message}")
    
    # Simple echo response
    history[-1][1] = f"RAG test successful (non-streaming): You asked -> {user_message}"
    return history

import traceback

def crc_chatbot_rag(history, message, mode_ui="vanilla (fast)"):
    """
    Stream response using yield for better UX.
    """
    print("========== [DEBUG] crc_chatbot_rag called ==========")
    print("[DEBUG] history type:", type(history), "value:", history)
    print("[DEBUG] message:", message)
    print("[DEBUG] mode:", mode_ui)

    # API Key from environment variable
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    # Map UI mode to internal mode
    mode_map = {
        "vanilla (fast)": "vanilla",
        "agentic (deep)": "agentic"
    }
    mode_internal = mode_map.get(mode_ui, "vanilla")
    
    history = normalize_chat_history(history)

    # 1. Basic input validation
    if not message or message.strip() == "":
        bot_reply = "Please enter a question before submitting."
        history.append([None, bot_reply])
        safe_history = [(u, v) for u, v in history]
        yield safe_history, "", None
        return

    try:
        # 2. Call RAG backend
        print(f"[DEBUG] calling RAG backend in {mode_internal} mode ...")
        
        # Initialize user message in history with empty bot response
        # history format: [[user_msg, bot_msg], ...]
        history.append([message, ""])
        
        stats = None
        
        # Iterate over the generator from answer_crc_question
        # It yields accumulated text chunks
        for chunk in answer_crc_question(message, api_key=api_key, mode=mode_internal):
             if isinstance(chunk, dict):
                 stats = chunk
                 continue 
             
             # Update the last bot message with the current accumulated text
             history[-1][1] = chunk
             
             # Yield the updated history to update the UI in real-time
             safe_history = [(u, v) for u, v in history]
             yield safe_history, "", stats
             
        print("[DEBUG] RAG streaming complete.")
        safe_history = [(u, v) for u, v in history]
        yield safe_history, "", stats

    except Exception as e:
        print("!!!!!! [ERROR] Exception in crc_chatbot_rag !!!!!!")
        print("[ERROR]", repr(e))
        traceback.print_exc()

        bot_reply = f"Internal error in RAG logic: {e}\nPlease screenshot the terminal log for the developer to troubleshoot."
        history[-1][1] = bot_reply
        safe_history = [(u, v) for u, v in history]
        yield safe_history, "", None

def format_concentration(conc_M):
    """
    Format concentration with adaptive units (fM, pM, nM, uM, mM).
    Input: concentration in Molar (M).
    Output: formatted string (e.g., "12.0 pM").
    """
    if conc_M is None:
        return "N/A"
    try:
        if np.isnan(conc_M):
            return "N/A"
    except:
        pass
    
    # Handle zero or negative
    if conc_M <= 0:
        return "0 M"

    if conc_M < 1e-12:
        # fM range (or even smaller, but treating as fM)
        val = conc_M * 1e15
        unit = "fM"
    elif conc_M < 1e-9:
        # pM range: [1e-12, 1e-9)
        val = conc_M * 1e12
        unit = "pM"
    elif conc_M < 1e-6:
        # nM range: [1e-9, 1e-6)
        val = conc_M * 1e9
        unit = "nM"
    elif conc_M < 1e-3:
        # uM range: [1e-6, 1e-3)
        val = conc_M * 1e6
        unit = "µM"
    else:
        # mM range: >= 1e-3
        val = conc_M * 1e3
        unit = "mM"
        
    # Format to 3 significant figures
    return f"{val:.3g} {unit}"

def run_droplet_gradio(img: np.ndarray, target: str, ambiguous_policy: str = "error", max_side: int = 1024, save_masks: bool = False, save_overlays: bool = False) -> str:
    if img is None:
        print("[DEBUG] No image provided.")
        return "Please upload a droplet image first."
    
    print(f"[DEBUG] Processing image for target: {target}, policy: {ambiguous_policy}, shape: {img.shape}, dtype: {img.dtype}")
    print(f"[DEBUG] Image min: {img.min()}, max: {img.max()}")
    
    # Try to handle possible non-standard image formats (although Gradio should have converted to numpy)
    # If it is TIFF, Gradio may have read it in, as long as it is a numpy array it is usually fine
    # But if it is float type (0-1), it may need to be converted to uint8 (0-255)
    if img.dtype == np.float32 or img.dtype == np.float64:
        print("[DEBUG] Image is float, converting to uint8")
        try:
            # Check if it's 0-1 normalized or just float container for 0-255
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            print(f"[DEBUG] Converted image shape: {img.shape}, dtype: {img.dtype}")
        except Exception as e:
             print(f"[ERROR] Failed to convert image: {e}")
             return f"Image format conversion failed: {str(e)}"

    try:
        print("[DEBUG] Calling full_droplet_analysis...")
        result = full_droplet_analysis(img, target, ambiguous_policy=ambiguous_policy, max_side=int(max_side))
        print(f"[DEBUG] full_droplet_analysis returned: {result}")
        
        if result is None:
             print("[ERROR] full_droplet_analysis returned None")
             return "Analysis failed: Internal error (result is None)"

        lines = []
        lines.append(f"### Target: {result['target']}")
        
        # Simplified output as requested
        # Removed target_source and ambiguous_reason
             
        lines.append(f"- Disease classification: **{result['label']}**")
        
        if result.get("confidence") is not None:
            lines.append(f"- Prediction confidence: {result['confidence']:.3f}")
        
        # Adaptive Concentration Formatting
        conc_M = result.get("concentration_M")
        
        # Fallback if concentration_M is missing but fM is there
        if conc_M is None and result.get("concentration_fM") is not None:
             conc_M = result.get("concentration_fM") * 1e-15
        # Fallback if only log10_concentration is there
        elif conc_M is None and result.get("log10_concentration") is not None:
             conc_M = 10 ** result.get("log10_concentration")
             
        formatted_conc = format_concentration(conc_M)
        
        # Optional: Add scientific notation if needed, but user emphasized the adaptive unit
        # Adding simple scientific notation for reference if valid
        extra_info = ""
        if conc_M is not None and conc_M > 0:
             extra_info = f" (≈ 10^{np.log10(conc_M):.2f} M)"
             
        lines.append(f"- Estimated concentration: **{formatted_conc}**{extra_info}")
             
        lines.append(f"- Number of detected droplets: {result.get('n_droplets', 'N/A')}")
        
        return "\n".join(lines)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] Exception in run_droplet_gradio: {e}")
        return f"Error during analysis: {str(e)}"

def analyze_and_suggest(file_obj, target):
    """
    Generator that first yields the image analysis result,
    then automatically triggers RAG to generate suggestions based on the result.
    """
    
    # 0. Read Image from File Path
    if file_obj is None:
        yield "Please upload an image first.", ""
        return

    # Extract file path from file object (if type="file") or string (if type="filepath")
    if isinstance(file_obj, str):
        file_path = file_obj
    elif hasattr(file_obj, "name"):
        file_path = file_obj.name
    else:
        # Fallback
        file_path = str(file_obj)

    try:
        # cv2.imread handles tiff well
        print(f"[DEBUG] Reading image from: {file_path}")
        img = cv2.imread(file_path)
        
        if img is None:
             yield "Failed to read image. Please ensure it is a valid image file.", ""
             return
             
        # Convert BGR (OpenCV default) to RGB (Model expectation)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        print(f"[ERROR] Image read failed: {e}")
        yield f"Error reading image: {str(e)}", ""
        return

    # 1. Run Image Analysis
    # Hardcoded parameters as requested
    max_side = 768
    save_masks = False
    save_overlays = False
    
    analysis_result = run_droplet_gradio(img, target, ambiguous_policy="error", max_side=max_side, save_masks=save_masks, save_overlays=save_overlays)
    
    # Determine Title based on result for the placeholder
    is_crc_positive_temp = False
    if "**Disease**" in analysis_result or "**Cancer**" in analysis_result:
        is_crc_positive_temp = True
    elif "Disease" in analysis_result and "Healthy" not in analysis_result:
        is_crc_positive_temp = True
        
    if is_crc_positive_temp:
        temp_title = "### 💊 Clinical Management Insights\n\n"
    else:
        temp_title = "### 🌿 Health & Prevention Tips\n\n"
        
    # Yield analysis result immediately so user doesn't wait
    # The suggestion part shows a waiting message
    # [MODIFIED] Use a specific status instead of generic "Generating..."
    placeholder_text = temp_title + "🔍 *Retrieving evidence from clinical literature...*"
    yield analysis_result, placeholder_text
    
    # 2. Determine Prompt based on Analysis Result
    
    
    # We look for keywords "Disease" or "Cancer" in the output text to identify CRC positive cases.
    # Note: run_droplet_gradio outputs markdown like: "**Disease**" or "**Cancer**" or "**Healthy**"
    
    is_crc_positive = False
    if "**Disease**" in analysis_result or "**Cancer**" in analysis_result:
        is_crc_positive = True
    elif "Disease" in analysis_result and "Healthy" not in analysis_result:
        # Fallback check
        is_crc_positive = True
        
    prompt = ""
    title_md = ""
    
    if is_crc_positive:
        # Scenario B: CRC / Disease
        prompt = random.choice(DISEASE_QUESTIONS)
        title_md = "### 💊 Clinical Management Insights\n\n> **Note:** The following insights are based on retrieved clinical literature.\n\n"
    else:
        # Scenario A: Healthy / Non-CRC
        prompt = random.choice(HEALTHY_QUESTIONS)
        title_md = "### 🌿 Health & Prevention Tips\n\n> **Tip:** Early prevention and lifestyle changes are key.\n\n"
        
    # Call RAG with generator
    # Ensure API key is available
    if not api_key:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
    
    if not api_key:
         yield "Error: DEEPSEEK_API_KEY not found in environment variables. Please check .env file."
         return


    print(f"[UI] entering run callback, time={time.time()}, pid={os.getpid()}, python={sys.executable}")
    print(f"[UI] prompt (first 120 chars): {prompt[:120] if prompt else 'None'}")
    should_call_rag = bool(prompt and prompt.strip())
    print(f"[UI] will call RAG: {should_call_rag}")

    if not should_call_rag:
        yield analysis_result, title_md + "\n(No prompt generated, RAG skipped)"
        return

    try:
        accumulated_text = ""
        # Yield a loading state
        # yield analysis_result, title_md + "_Generating suggestions..._"
        
        print("[UI] RAG start ...")
        t0 = time.time()
        
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        
        # Wrapper to consume the generator with timeout
        # We collect all chunks to avoid complexity of streaming across threads with timeout
        # OR we just wait for the first chunk to ensure it doesn't hang on retrieval.
        
        def start_rag_stream():
            # This function runs in a thread. 
            # It creates the generator and fetches the first chunk.
            # If successful, it returns the generator and the first chunk.
            g = answer_crc_question(prompt, api_key=api_key)
            try:
                first_chunk = next(g)
                return g, first_chunk
            except StopIteration:
                return g, None
                
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(start_rag_stream)
            try:
                # Wait up to 10s for the first chunk (which includes retrieval)
                gen, first_chunk = future.result(timeout=10)
                
                # If we got here, retrieval is done and we have the first chunk.
                if first_chunk is not None:
                    if isinstance(first_chunk, str):
                        accumulated_text += first_chunk
                        yield analysis_result, title_md + accumulated_text
                    
                    # Continue consuming the rest of the generator directly (in main thread)
                    for chunk in gen:
                        if isinstance(chunk, dict):
                            continue
                        accumulated_text = chunk
                        yield analysis_result, title_md + chunk
                else:
                    yield analysis_result, title_md + "\nNo suggestions available."
                    
            except TimeoutError:
                print(f"[RAG] TIMEOUT after 10s")
                err_msg = "RAG timeout after 10s (retrieval/generation may be blocked). Check vector DB path and LLM/embedding connectivity."
                yield analysis_result, title_md + "\n" + err_msg
                return

        
        t1 = time.time()
        print(f"[UI] RAG returned in {t1-t0:.4f} s")
            
    except Exception as e:
        print(f"[ERROR] RAG Suggestion failed: {e}")
        import traceback
        traceback.print_exc()
        yield analysis_result, title_md + f"\n(RAG failed: {type(e).__name__}: {str(e)})"


# --- Gradio Interface Construction ---
css = """
.gradio-container { max-width: 1200px !important; }
.group-box {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    background-color: white;
}
.progress-text{display:none !important;}
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"]
)

with gr.Blocks(theme=theme, title="iDAPP-agent AI", css=css, analytics_enabled=False) as demo:
    # Header
    gr.Markdown(
        """
        # 🔬 iDAPP-agent: Intelligent Colorectal Cancer Analysis Platform
        ### Dual-channel Digital Ago Analysis & RAG-based Clinical Assistant
        """,
        elem_classes="text-center"
    )
    
    with gr.Tabs():
        with gr.Tab("iDAPP-agent (RAG)"):
            gr.Markdown("### iDAPP-agent (RAG)\nThis tool does not provide medical advice. All outputs are for research and educational use only and must not be used for diagnosis or treatment.")

            crc_chatbot = gr.Chatbot(label="CRC Assistant")
            
            with gr.Row():
                with gr.Column(scale=4):
                    msg = gr.Textbox(label="Question", placeholder="Type your question about CRC here...", lines=3)
                with gr.Column(scale=1):
                    mode_radio = gr.Radio(
                        choices=["vanilla (fast)", "agentic (deep)"],
                        value="vanilla (fast)",
                        label="Answer mode"
                    )
            
            submit_btn = gr.Button("Submit", variant="primary")
            
            with gr.Accordion("Run Statistics", open=False):
                stats_box = gr.JSON(label="Execution Details")

            submit_btn.click(
                fn=crc_chatbot_rag,
                inputs=[crc_chatbot, msg, mode_radio],
                outputs=[crc_chatbot, msg, stats_box],
            )

            msg.submit(
                fn=crc_chatbot_rag,
                inputs=[crc_chatbot, msg, mode_radio],
                outputs=[crc_chatbot, msg, stats_box],
            )

        with gr.Tab("miR DropNet"):
            gr.Markdown("### Droplet / miRNA Analysis")
            with gr.Row():
                with gr.Column():
                    with gr.Group(elem_classes="group-box"):
                        # Changed from gr.Image to gr.File to support TIFF without preview issues
                        droplet_img_input = gr.File(
                            label="Upload Image (TIFF/PNG)",
                            # file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                            type="file"
                        )
                        
                        target_radio = gr.Radio(
                            choices=["miR-21", "miR-92a"],
                            value="miR-92a",
                            label="Target",
                        )
                        
                        run_btn = gr.Button("🚀 Run Analysis", variant="primary")
                
                with gr.Column():
                    with gr.Group(elem_classes="group-box"):
                        result_md = gr.Markdown(label="Analysis Summary")
                        
                        # New Component for Intelligent Suggestions
                        with gr.Accordion("Intelligent Suggestions", open=True):
                            suggestion_md = gr.Markdown(label="Intelligent Suggestions")
            
            # Updated click event to use the new generator function
            # Disable button -> Run Analysis -> Enable button
            run_btn.click(
                fn=lambda: gr.update(interactive=False),
                outputs=run_btn,
                queue=False
            ).then(
                fn=analyze_and_suggest,
                inputs=[droplet_img_input, target_radio],
                outputs=[result_md, suggestion_md],
                queue=True,
                show_progress="hidden",
            ).then(
                fn=lambda: gr.update(interactive=True),
                outputs=run_btn,
                queue=False
            )
            
            # Inject JavaScript to replace Chinese text with English
            gr.HTML("""
            <script>
            (function() {
              const REPLACE_MAP = {
                "拖放图片至此处": "Drag & drop image here",
                "点击上传": "Click to upload",
                "-或-": "- or -"
              };

              function replaceAllText() {
                const walker = document.createTreeWalker(
                  document.body,
                  NodeFilter.SHOW_TEXT,
                  null,
                  false
                );
                let node;
                while ((node = walker.nextNode())) {
                  const trimmed = node.nodeValue.trim();
                  if (REPLACE_MAP[trimmed]) {
                    node.nodeValue = REPLACE_MAP[trimmed];
                  }
                }
              }

              # Initial replacement after delay
              setTimeout(replaceAllText, 1000);

              # Listen for DOM changes
              const observer = new MutationObserver((mutations) => {
                let shouldRun = false;
                for (const m of mutations) {
                  if (m.addedNodes && m.addedNodes.length > 0) {
                    shouldRun = true;
                    break;
                  }
                }
                if (shouldRun) {
                  replaceAllText();
                }
              });

              if (document.body) {
                observer.observe(document.body, { childList: true, subtree: true });
              } else {
                document.addEventListener("DOMContentLoaded", () => {
                  observer.observe(document.body, { childList: true, subtree: true });
                  replaceAllText();
                });
              }
            })();
            </script>
            """)

if __name__ == "__main__":
    # --- Start Application ---
    import sys
    print("[DEBUG] sys.argv =", sys.argv)
    print("[DEBUG] sys.executable =", sys.executable)
    print("[BOOT] Force launching on port 7860...")

    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860, 
        share=False,       # Disable share for debugging
        debug=True,
        show_error=True
    )
