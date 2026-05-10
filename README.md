# 🔥 PyTorch Mastery — Zero to Hero

<div align="center">

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Course](https://img.shields.io/badge/Course-ZTM%20PyTorch-blueviolet?style=for-the-badge)](https://www.udemy.com/course/pytorch-for-deep-learning/)
[![Status](https://img.shields.io/badge/Status-In%20Progress-brightgreen?style=for-the-badge)]()
[![GitHub](https://img.shields.io/badge/GitHub-yokeswarans-181717?style=for-the-badge&logo=github)](https://github.com/yokeswarans)

</div>

---

> *"The expert in anything was once a beginner."*

This isn't just a course repo. It's a record of a real learning journey — late nights debugging tensor shapes, "aha" moments when the math finally clicked, and every experiment that failed before one finally worked.

If you're thinking about learning PyTorch — **this repo is proof that you can do it too.**

---

## 🙋 Why I Started This

Deep learning felt intimidating. Papers I couldn't understand. Code that didn't make sense. A gap between "I know Python" and "I can build neural networks."

The [Zero to Mastery PyTorch course](https://www.udemy.com/course/pytorch-for-deep-learning/) changed that. It doesn't just teach you the API — it teaches you how to *think* like a deep learning engineer. Every section builds on the last, every exercise forces you to actually use what you've learned.

So I documented everything. Every notebook here is my own code, my own experiments, my own notes. Not copy-pasted — actually worked through.

---

## 📍 My Progress

| # | Section | Core Skill Built | My Notebook | Status |
|:-:|---------|-----------------|-------------|:------:|
| `00` | **PyTorch Fundamentals** | Tensors, shapes, GPU operations — the atomic unit of all deep learning | [📓 View](https://github.com/yokeswarans/pytorch-mastery-zero-to-hero/blob/main/00_pytorch_fundamentals.ipynb) | ✅ Done |
| `01` | **PyTorch Workflow** | The end-to-end loop: data → model → loss → train → evaluate → save | [📓 View](https://github.com/yokeswarans/pytorch-mastery-zero-to-hero/blob/main/01_Pytorch_workflow.ipynb) | ✅ Done |
| `02` | **Neural Network Classification** | Binary & multi-class classification, activation functions, decision boundaries | [📓 View](https://github.com/yokeswarans/pytorch-mastery-zero-to-hero/blob/main/02_Pytorch_classification.ipynb) | ✅ Done |
| `03` | **Computer Vision** | CNNs, convolutional layers, pooling, feature maps, FashionMNIST experiments | [📓 View](https://github.com/yokeswarans/pytorch-mastery-zero-to-hero/blob/main/03_Pytorch_computer_vision.ipynb) | ✅ Done |
| `04` | **Custom Datasets** | Loading real-world image data, transforms, `DataLoader`, train/val/test splits | [📓 View](https://github.com/yokeswarans/pytorch-mastery-zero-to-hero/blob/main/04_Pytorch_custom_dataset.ipynb) | ✅ Done |
| `05` | **Going Modular** | Refactoring messy notebooks into clean, reusable Python scripts — real-world engineering | [📓 View](https://github.com/yokeswarans/pytorch-mastery-zero-to-hero/blob/main/05_Pytorch_modular_cellmode.ipynb) | ✅ Done |
| `06` | **Transfer Learning** | Standing on the shoulders of giants — fine-tuning EfficientNet on custom data | [📓 View](https://github.com/yokeswarans/pytorch-mastery-zero-to-hero/blob/main/06_Pytorch_transfer_learning.ipynb) | ✅ Done |
| `07` | **🏆 Milestone 1 — Experiment Tracking** | Tracking, comparing, and managing model runs like a professional ML engineer | [📓 View](https://github.com/yokeswarans/pytorch-mastery-zero-to-hero/blob/main/07_Pytorch_experiment_tracking.ipynb) | ✅ Done |
| `08` | **🏆 Milestone 2 — Model Deployment** | Shipping a real trained model to the internet with Gradio + HuggingFace Spaces | [📓 View](https://github.com/yokeswarans/pytorch-mastery-zero-to-hero/blob/main/08_Pytorch_model_deployment.ipynb) | ✅ Done |
| `09` | **🏆 Milestone 3 — Paper Replicating** | Reproducing the Vision Transformer (ViT) research paper from scratch in PyTorch | Coming soon | 🔄 Next |
| `10` | **PyTorch 2.0** | `torch.compile`, new performance features, and what changed in PyTorch 2.0 | Coming soon | 🔄 Next |

---

## 💡 What I Actually Learned (Beyond the Code)

These are the real lessons this course taught me — the ones that don't show up in a syllabus:

- **Start simple, then add complexity.** Every great model starts as a basic linear layer. Resist the urge to jump straight to transformers.
- **The training loop is everything.** Once you truly understand forward pass → loss → backward → update, nothing in deep learning feels magical anymore.
- **Visualize before you optimize.** Plot your data. Plot your predictions. Plot your loss curves. You'll catch 80% of bugs just by looking.
- **Transfer learning is a superpower.** Fine-tuning a pretrained model beats training from scratch almost every time — use it.
- **If in doubt, run the code.** The best way to understand anything in PyTorch is to experiment. Break things on purpose. Then fix them.

---


## 🗂️ Repo Structure

```
pytorch-mastery-zero-to-hero/
│
├── 📁 Excercises/                        ← Section exercises (some pending upload ⏳)
├── 📁 demos/
│   └── foodvision_mini/                  ← HuggingFace deployment demo app
├── 📁 going_modular/                     ← Modular Python scripts (prediction.py etc.)
├── 📁 models/                            ← Saved model files for deployment
│
├── 00_pytorch_fundamentals.ipynb         
├── 01_Pytorch_workflow.ipynb             
├── 02_Pytorch_classification.ipynb       
├── 03_Pytorch_computer_vision.ipynb      
├── 04_Pytorch_custom_dataset.ipynb       
├── 05_Pytorch_modular_cellmode.ipynb     
├── 06_Pytorch_transfer_learning.ipynb    
├── 07_Pytorch_experiment_tracking.ipynb  
├── 08_Pytorch_model_deployment.ipynb     
│
├── helper_functions.py                   ← Shared utilities across notebooks
└── .gitignore                            
```

---

## 🚀 Run It Yourself

```bash
git clone https://github.com/yokeswarans/pytorch-mastery-zero-to-hero.git
cd pytorch-mastery-zero-to-hero
pip install torch torchvision torchaudio
pip install matplotlib pandas tqdm
jupyter notebook
```

> 💡 **Tip:** Open any `.ipynb` file, read the section's objectives first, then run each cell one by one — don't just scroll through. The understanding comes from *doing* it.

---

## 🎯 Who Should Take This Course

You're ready for this if you have:

- 3–6 months of Python experience (loops, functions, classes)
- A basic idea of what machine learning is — no depth needed
- Access to Google Colab or a local GPU — free tier is enough to start
- The patience to sit with confusion until it turns into clarity

You **don't** need a math PhD. You **don't** need to have built a model before. You just need to show up consistently.

> **Great starting point:** [ZTM Data Science & ML Bootcamp](https://dbourke.link/ZTMMLcourse) gives you the Python + ML foundations to hit the ground running.

---

## 📚 Resources That Made This Possible

| Resource | Why It's Valuable |
|----------|------------------|
| [🎓 Udemy Course](https://www.udemy.com/course/pytorch-for-deep-learning/) | Full structured course with videos, exercises & projects |
| [📘 learnpytorch.io](https://learnpytorch.io) | Free companion book — searchable, detailed, excellent |
| [🎥 YouTube (25+ hrs free)](https://youtu.be/Z_ikDlimN6A) | Watch the entire course for free on YouTube |
| [📄 PyTorch Docs](https://pytorch.org/docs/stable/index.html) | Official documentation — your best friend after the course |

---

## 🤝 Let's Connect

If this repo helped you get started, or if you're working through the same course — I'd love to connect.

Drop a ⭐ if this was useful, open an issue if you spot something wrong, or just say hi on GitHub.

> *Every expert was once exactly where you are right now. The only difference is they kept going.*

<div align="center">
  <br>
  <strong>My notes, my code, my progress - by <a href="https://github.com/yokeswarans">yokeswaran</a></strong>
  <br><br>
  <em>Learning in public · Failing forward · Getting better every day</em>
</div>
