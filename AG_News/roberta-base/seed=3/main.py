import os
import torch
from pathlib import Path
from google.cloud import storage
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

#----------------------------------------------------------------------

dataset = 'AG_News'
model_name = 'roberta-base'
seed = 3
num_labels = 4

BUCKET_NAME = os.getenv("MODELS_BUCKET", "models-paper-bert-family")
MODEL_PREFIX = os.getenv("MODEL_PREFIX", f"{dataset}/{model_name}/seed={seed}/epoch_1")

LOCAL_MODEL_DIR = Path(os.getenv("LOCAL_MODEL_DIR", "./model"))

#----------------------------------------------------------------------

def gcs_sync_prefix(bucket_name: str, prefix: str, local_dir: Path) -> None:
    """
    Downloads all objects under gs://bucket/prefix/ into local_dir,
    preserving the subdirectory structure.
    """
    # Initialize GCS client
    client = storage.Client()

    # Ensure the local target directory exists
    local_dir.mkdir(parents=True, exist_ok=True)

    found_any = False

    # List all blobs under the given prefix
    for blob in client.list_blobs(bucket_name, prefix=prefix.rstrip("/") + "/"):
        # Skip directory placeholders
        if blob.name.endswith("/"):
            continue

        found_any = True

        # Compute relative path with respect to the prefix
        rel_path = blob.name[len(prefix):].lstrip("/")

        # Build destination path on local filesystem
        dest_path = local_dir / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Download blob to local file
        blob.download_to_filename(dest_path)

    # Fail fast if the prefix contained no files
    if not found_any:
        raise RuntimeError(f"No objects found at gs://{bucket_name}/{prefix}")

#----------------------------------------------------------------------

def ensure_model_local() -> None:
    """
    Ensures the model is available locally.
    Downloads it from GCS only if required files are missing.

    This function supports common Hugging Face layouts, such as:
    - pytorch_model.bin
    - model.safetensors
    - sharded weights (model-*.safetensors / pytorch_model-*.bin)
    """
    # Check for mandatory configuration file
    has_config = (LOCAL_MODEL_DIR / "config.json").exists()

    # Check for any supported model weight format
    has_weights = (
        (LOCAL_MODEL_DIR / "model.safetensors").exists()
        or (LOCAL_MODEL_DIR / "pytorch_model.bin").exists()
        or any(LOCAL_MODEL_DIR.glob("model-*.safetensors"))
        or any(LOCAL_MODEL_DIR.glob("pytorch_model-*.bin"))
    )

    # Download from GCS only if something critical is missing
    if not (has_config and has_weights):
        gcs_sync_prefix(BUCKET_NAME, MODEL_PREFIX, LOCAL_MODEL_DIR)

#----------------------------------------------------------------------

print("✔ Flask app is loading...")
ensure_model_local()

tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(
    str(LOCAL_MODEL_DIR),
    num_labels=num_labels,
)
model.eval()

print(f"✔ Model loaded from {LOCAL_MODEL_DIR}")

#----------------------------------------------------------------------

@app.route("/", methods=["POST"])
def predict_labels():
  
  data = request.get_json(silent=True) or {}
  text = data.get("text", "")
  
  encoded_input = tokenizer(text, return_tensors='pt', truncation = True)
  output = model(**encoded_input)
  logits_array = output.logits.to('cpu').detach().numpy()[0]
  pred = int(logits_array.argmax())

  return jsonify({"prediction": pred}), 200

#----------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

#----------------------------------------------------------------------