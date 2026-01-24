# CLIP-FAISS-Multimodal-Retrieval-Engine

Multimodal Image Search Using CLIP and FAISS

A vision–language image retrieval system that supports **text-to-image** and **image-to-image** search using OpenAI’s CLIP model and FAISS for fast similarity search.

---

## Features

- Text → Image semantic search  
- Image → Image similarity search  
- CLIP (ViT-B/32) embeddings  
- FAISS indexing for fast retrieval  
- Works on Kaggle / Colab  
- Visualization of top-k results  

---

## Tech Stack

- Python  
- PyTorch  
- OpenAI CLIP  
- FAISS  
- NumPy  
- Pillow  
- Matplotlib  

---

## Dataset

This project uses the **Flickr8k** dataset.

Download via Kaggle:

```python
import kagglehub
kagglehub.dataset_download("adityajn105/flickr8k")
