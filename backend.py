import transformers
from flask import Flask, request, jsonify
import transformers
import torch

app = Flask(__name__)

# Load QA model
qa_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
qa_tokenizer = transformers.AutoTokenizer.from_pretrained(qa_model_name)
qa_model = transformers.AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

# Load BERT model for attention visualization
attention_model_name = "bert-base-uncased"
attention_tokenizer = transformers.BertTokenizer.from_pretrained(attention_model_name)  # Removed '47'
attention_model = transformers.BertModel.from_pretrained(attention_model_name, output_attentions=True)

@app.route("/qa", methods=["POST"])
def answer_question():
    """Answer a question based on a given context."""
    data = request.json
    question = data.get("question")
    context = data.get("context")

    if not question or not context:
        return jsonify({"error": "Both question and context are required!"}), 400

    inputs = qa_tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = qa_model(**inputs)

    start_scores = torch.nn.functional.softmax(outputs.start_logits, dim=-1)
    end_scores = torch.nn.functional.softmax(outputs.end_logits, dim=-1)

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1

    confidence = (start_scores[0, start_index] * end_scores[0, end_index - 1]).item()
    answer = qa_tokenizer.decode(inputs.input_ids[0][start_index:end_index], skip_special_tokens=True)

    return jsonify({"answer": answer, "confidence": round(confidence * 100, 2)})


@app.route("/attention", methods=["POST"])
def attention_visualization():
    """Generate BERT attention visualization data."""
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "Text input is required!"}), 400

    inputs = attention_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = attention_model(**inputs)

    attentions = outputs.attentions
    num_layers = len(attentions)
    attention_shape = attentions[0].shape

    return jsonify({
        "message": "Attention visualization ready",
        "num_layers": num_layers,
        "attention_shape": str(attention_shape)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
