# 🛋️ FriendsGPT

**FriendsGPT** is a small, character-level GPT language model trained on **3.9MB** of *Friends* TV show transcripts. It generates authentic-sounding sitcom dialogue between characters like **Ross**, **Rachel**, **Joey**, and more — live, word by word.

The model achieved a validation loss of **1.06**, and now supports **streaming output** through a smooth, interactive **Gradio** interface.

![Gradio Screenshot](https://github.com/user-attachments/assets/81439ff3-483f-42da-95d9-543232f86579) <!-- Optional: Replace or remove this if no image -->

---

## 🚀 Features

-  Trained on ~3.9MB of *Friends* transcripts  
-  Validation loss: **1.06**  
-  Real-time **streaming generation**  
-  Lightweight GPT-style architecture  
-  Built-in **Gradio web interface**  
-  Easy to run locally  

---

## 🧠 Model Summary

| Hyperparameter | Value       |
|----------------|-------------|
| Block size     | 256         |
| Embedding size | 384         |
| Layers         | 6           |
| Heads          | 6           |
| Dropout        | 0.2         |
| Max iters      | ~14,000     |

---

## 🧪 Example

### 🗣️ Prompt:
#### Input: 
```
ROSS: Rachel, I love you.

RACHEL:
```
#### Output:
```
ROSS: Rachel, I love you.

RACHEL: Ohh, thank you.

MONICA: Alright. 

ROSS: What's wrong with this?

MONICA: Nothing. 
```
--- 

### 📹 Demo
![Gradio Demo](https://github.com/user-attachments/assets/429d3a17-0ab4-4f8f-a4f6-a8788091601b)

