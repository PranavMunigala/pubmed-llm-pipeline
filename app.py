import gradio as gr

from pipeline import run_pipeline


def search_pubmed(query: str, n: int):
    return run_pipeline(query=query, n=int(n), output=None)


with gr.Blocks(title="PubMed LLM Pipeline") as demo:
    gr.Markdown("# PubMed LLM Pipeline")
    query = gr.Textbox(label="Search query", placeholder="e.g. protein folding neural network")
    n = gr.Slider(5, 100, value=50, step=1, label="N")
    submit = gr.Button("Run")
    output = gr.JSON(label="Structured output")
    submit.click(search_pubmed, inputs=[query, n], outputs=output)


if __name__ == "__main__":
    demo.launch()
