import os
import sys
import json
import torch
import torchaudio
from transformers.models.wav2vec2 import Wav2Vec2ForCTC, Wav2Vec2Processor
from huggingface_hub import snapshot_download


def load_model(model_dir):
    model_file = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(model_file):
        print(
            json.dumps(
                {"status": "downloading", "message": "Starting model download..."}
            ),
        )
        try:
            # Download the model files
            snapshot_download(
                repo_id="vitouphy/wav2vec2-xls-r-300m-timit-phoneme",
                local_dir=model_dir,
                ignore_patterns=["*.bin"],
            )
            print(
                json.dumps(
                    {"status": "downloading", "message": "Model download complete."}
                ),
            )
        except Exception as e:
            print(
                json.dumps(
                    {"status": "error", "message": f"Download failed: {str(e)}"}
                ),
            )
            sys.exit(1)
    else:
        print(
            json.dumps(
                {"status": "found", "message": "Model loaded from local directory."}
            ),
        )

    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    return processor, model


def process_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample to 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )
        waveform = resampler(waveform)

    # Convert to mono
    waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def main():
    if len(sys.argv) < 3:
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": "Usage: script.py <model_dir> <audio_path>",
                }
            )
        )
        sys.exit(1)

    model_dir = sys.argv[1]
    audio_path = sys.argv[2]

    try:
        processor, model = load_model(model_dir)
        waveform = process_audio(audio_path)

        input_values = processor(
            waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_values

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)

        result = transcription[0].strip()
        if result == "":
            print(
                json.dumps(
                    {
                        "status": "success",
                        "phonemes": None,
                        "message": "No phonemes detected.",
                    }
                )
            )
        else:
            print(json.dumps({"status": "success", "phonemes": result}))
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
