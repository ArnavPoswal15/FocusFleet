"""Gradio Blocks layout for FocusFleet.

This module only wires UI components to the callbacks defined in
`focusfleet.database` and `focusfleet.detection`. No detection math or
DB queries live here — that separation is what lets the UI be redesigned
independently from the underlying logic.
"""

import gradio as gr

from .. import config
from ..database import login_driver, register_driver
from ..detection import classify_image, process_frame
from ..state import STATE
from .styles import CUSTOM_CSS, HEADER_HTML


def poll_status():
    """Called by gr.Timer at 1 Hz — returns the current status banner HTML.

    Runs independently of the video stream's frame rate, so banner updates
    are infrequent and deliberate (no flicker on stable states).
    """
    return config.STATUS_HTML.get(STATE.current_status_key, config.STATUS_HTML[config.STATUS_NEUTRAL])


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="FocusFleet — Driver Drowsiness Detection",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(primary_hue="teal", secondary_hue="slate", neutral_hue="slate"),
    ) as demo:
        driver_state = gr.State(value=None)

        gr.HTML(HEADER_HTML)

        with gr.Tab("Login / Register", id="auth"):
            gr.Markdown("Sign in or create an account to start monitoring.")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<p class="auth-section-label">Login</p>')
                    login_name = gr.Textbox(label="Name", placeholder="Your name")
                    login_password = gr.Textbox(label="Password", type="password", placeholder="••••••••")
                    login_button = gr.Button("Login", variant="primary")
                    login_status = gr.Textbox(label="Status", interactive=False)
                with gr.Column(scale=1):
                    gr.HTML('<p class="auth-section-label">Register</p>')
                    reg_name = gr.Textbox(label="Name", placeholder="Your name")
                    reg_password = gr.Textbox(label="Password", type="password", placeholder="••••••••")
                    reg_contact = gr.Textbox(label="Contact Info", placeholder="Email or phone (optional)")
                    reg_license = gr.Textbox(label="License Number", placeholder="Optional")
                    register_button = gr.Button("Register", variant="secondary")
                    reg_status = gr.Textbox(label="Status", interactive=False)

            login_button.click(
                fn=login_driver,
                inputs=[login_name, login_password],
                outputs=[driver_state, login_status],
            )
            register_button.click(
                fn=register_driver,
                inputs=[reg_name, reg_password, reg_contact, reg_license],
                outputs=reg_status,
            )

        with gr.Tab("Live Monitoring", id="monitor"):
            welcome_msg = gr.Markdown(
                value="Please log in to access live drowsiness detection.",
                elem_id="welcome-msg",
            )
            ear_threshold_slider = gr.Slider(
                minimum=config.EAR_THRESHOLD_MIN,
                maximum=config.EAR_THRESHOLD_MAX,
                value=config.DEFAULT_EAR_THRESHOLD,
                step=config.EAR_THRESHOLD_STEP,
                label="EAR threshold (lower = more sensitive)",
                elem_id="ear-slider",
            )
            with gr.Row(elem_id="monitor-row"):
                webcam_input = gr.Image(
                    sources=["webcam"], type="numpy", streaming=True,
                    label="Webcam", height=380, show_label=True,
                )
                detection_output = gr.Image(
                    label="Detection", interactive=False, height=380, show_label=True,
                )
            status_banner = gr.HTML(
                value=config.STATUS_HTML[config.STATUS_NEUTRAL],
                elem_id="status-html",
            )

            # Fast loop: runs every frame, updates the detection view only.
            webcam_input.stream(
                fn=process_frame,
                inputs=[webcam_input, ear_threshold_slider],
                outputs=[detection_output],
            )

            # Slow loop: 1 Hz banner refresh, decoupled from frame rate.
            status_timer = gr.Timer(value=1)
            status_timer.tick(fn=poll_status, outputs=[status_banner])

        with gr.Tab("Image Check", id="upload"):
            gr.Markdown("Upload a single image to classify drowsiness (no login required).")
            upload_input = gr.Image(type="numpy", label="Upload Image", height=340)
            upload_prediction = gr.Textbox(label="Result", interactive=False, elem_id="upload-result")
            upload_input.change(fn=classify_image, inputs=upload_input, outputs=upload_prediction)

        def update_welcome(driver_info):
            if driver_info is not None:
                return f"**Welcome, {driver_info['Name']}!** Use the webcam below to monitor your alertness."
            return "Please log in to access live drowsiness detection."

        driver_state.change(fn=update_welcome, inputs=driver_state, outputs=welcome_msg)

    return demo
