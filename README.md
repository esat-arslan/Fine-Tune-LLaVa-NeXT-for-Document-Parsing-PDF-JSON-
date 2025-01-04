# Fine-Tune LLaVa-NeXT for Document Parsing (PDF → JSON)

## Overview
This project fine-tunes **LLaVa-NeXT (v1.6 Mistral 7B)** for document parsing tasks, specifically transforming **PDF or image-based documents into structured JSON format**. The dataset used for fine-tuning is **CORD-v2** (Receipt Dataset).

## Key Features
- **Fine-tuning LLaVa-NeXT:** Optimize model for document parsing tasks.
- **Support for QLoRA:** Efficient fine-tuning with 4-bit quantization.
- **Structured JSON Output:** Extract structured data from document images.
- **Validation Metrics:** Edit distance-based evaluation for accuracy.
- **Integration with Hugging Face Hub:** Automatic model versioning and uploads.
- **WandB Integration:** Real-time experiment tracking.

## Dataset
- **Dataset Name:** [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2)
- **Dataset Description:** Collection of receipt images annotated with JSON metadata.

## Installation
```bash
pip install transformers datasets peft nltk bitsandbytes lightning wandb
```

## Configuration
Update the following variables in your script:
```python
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
REPO_ID = "YOUR-HUB-REPO-TO-PUSH"
WANDB_PROJECT = "LLaVaNeXT"
WANDB_NAME = "llava-next-demo-cord"
```

## Training
Run the training script:
```python
trainer.fit(model_module)
```

## Evaluation
Evaluate the model using the validation dataset:
```python
eval_results = trainer.validate()
```

## Testing
Test the model on a sample document image:
```python
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts)
```

## Model Push to Hub
The model and processor are automatically pushed to the Hugging Face Hub after each epoch.

## JSON Output Example
```json
{
  "store_name": "Sample Store",
  "total_amount": "15.99",
  "items": [
    {"name": "Item 1", "price": "5.00"},
    {"name": "Item 2", "price": "10.99"}
  ]
}
```

## Logging with WandB
All training and validation logs are sent to your WandB dashboard.

## Hardware Requirements
- GPU with CUDA support
- At least 24GB GPU memory (Recommended)

## Acknowledgments
- [LLaVa-NeXT](https://github.com/llava-hf/llava-next)
- [CORD Dataset](https://huggingface.co/datasets/naver-clova-ix/cord-v2)
- Hugging Face

## License
This project is licensed under the **Apache 2.0 License**.

## Contributing
Contributions are welcome! Open an issue or submit a pull request to improve the project.


