import gradio as gr
from main import process_video_stream

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 视频实时情感识别（无音频）")
    gr.Markdown("上传视频，逐帧实时分析并显示情感标签（无音频输出）")
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="上传视频", interactive=True)
            btn = gr.Button("开始实时分析", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="实时画面", type="numpy", height=400)
    btn.click(fn=process_video_stream, inputs=input_video, outputs=output_image)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)