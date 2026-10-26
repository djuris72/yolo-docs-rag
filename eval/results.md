Eksperiment	hit@1	hit@5	MRR
Baseline (bge-small, bez filtera)	0.35	0.87	0.578
Bez yolov5/ i platform/	0.52	0.94	0.690


(venv) D:\code\yolo-docs-rag>python -m eval.run_eval
exclude: none
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 2552.61it/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
n=31  hit@1=0.35  hit@5=0.87  MRR=0.578
MISS ('q18', 'Which setting reduces memory use by changing how many images are processed at once?', 6)
MISS ('q19', 'How do I make predictions only keep detections above a certain certainty?', 14)
MISS ('q28', 'How do I export a model and then check the accuracy of the exported version?', 8)
MISS ('q29', 'Which export formats are best for CPU inference and how do I use them?', 7)

(venv) D:\code\yolo-docs-rag>python -m eval.run_eval --exclude yolov5 platform
exclude: ['yolov5', 'platform']
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 2758.37it/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
n=31  hit@1=0.52  hit@5=0.94  MRR=0.690
MISS ('q19', 'How do I make predictions only keep detections above a certain certainty?', 12)
MISS ('q28', 'How do I export a model and then check the accuracy of the exported version?', 7)

(venv) D:\code\yolo-docs-rag>set LLM_MODEL=qwen2.5:3b

(venv) D:\code\yolo-docs-rag>python -m src.generate How do I resume an interrupted training run?
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 2622.26it/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
You can resume an interrupted training run by setting the `resume` argument to `True` and specifying the path to the last saved checkpoint. Here are examples for both Python and CLI:

### Python
```python
from ultralytics import YOLO

# Load the partially trained model
model = YOLO("path/to/last.pt")

# Resume training
results = model.train(resume=True)
```

### CLI
```bash
yolo train resume model=path/to/last.pt
```

For more information, you can refer to the FAQ section on Resuming Interrupted Trainings. 

Sources:
[1] https://docs.ultralytics.com/platform/train/cloud-training/
[2] https://docs.ultralytics.com/yolov5/tutorials/comet-logging-integration/
[3] https://docs.ultralytics.com/modes/train/
[4] https://docs.ultralytics.com/modes/train/

(venv) D:\code\yolo-docs-rag>python -m src.generate How do I train a model on audio data?
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 1944.83it/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
I could not find this in the Ultralytics documentation. 

Sources:
[1] https://docs.ultralytics.com/guides/model-training-tips/
[2] https://docs.ultralytics.com/platform/train/models/
[3] https://docs.ultralytics.com/guides/model-training-tips/
[4] https://docs.ultralytics.com/guides/steps-of-a-cv-project/

(venv) D:\code\yolo-docs-rag>

Prva komanda je filter plus reranker, a druga samo reranker, bez filtera. Drugo merenje je zapravo najzanimljivije, jer pokazuje može li reranker da zameni grubo filtriranje, a ne mora da ga dopunjava.

(venv) D:\code\yolo-docs-rag>python -m eval.run_eval --exclude yolov5 platform --rerank
exclude: ['yolov5', 'platform'] True
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 2215.06it/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 794/794 [00:00<00:00, 2.61MB/s]
D:\code\yolo-docs-rag\venv\Lib\site-packages\huggingface_hub\file_download.py:149: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in D:\hf_cache\hub\models--cross-encoder--ms-marco-MiniLM-L-6-v2. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
D:\code\yolo-docs-rag\venv\Lib\site-packages\huggingface_hub\file_download.py:149: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in D:\hf_cache\hub. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
model.safetensors: downloading bytes: ████████████████████████████████████████████████████████████████████████████████████████████████████| 86.0MB, 6.92MB/s  
model.safetensors: reconstructing file: 100%|████████████████████████████████████████████████████████████████████████████████████| 90.9MB / 90.9MB, 7.93MB/s  
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 105/105 [00:00<00:00, 2132.29it/s]
tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1.33k/1.33k [00:00<00:00, 4.17MB/s]
vocab.txt: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 8.30MB/s]
tokenizer.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 711k/711k [00:00<00:00, 23.6MB/s]
special_tokens_map.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 132/132 [00:00<00:00, 604kB/s]
n=31  hit@1=0.71  hit@5=0.94  MRR=0.802
MISS ('q19', 'How do I make predictions only keep detections above a certain certainty?', 12)
MISS ('q28', 'How do I export a model and then check the accuracy of the exported version?', 10)

(venv) D:\code\yolo-docs-rag>python -m eval.run_eval --rerank
exclude: [] True
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 2693.07it/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 105/105 [00:00<00:00, 2527.34it/s]
n=31  hit@1=0.58  hit@5=0.94  MRR=0.735
MISS ('q19', 'How do I make predictions only keep detections above a certain certainty?', 14)
MISS ('q28', 'How do I export a model and then check the accuracy of the exported version?', 11)

(venv) D:\code\yolo-docs-rag>