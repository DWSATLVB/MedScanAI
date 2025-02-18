# MedScanAI
An AI model to analyze your scans :)

## Description
Our AI-powered medical imaging model leverages the Claude API and GPT-Neo to analyze X-ray, CT, and MRI scans with precision. Designed to assist radiologists, it enhances diagnostic accuracy by identifying patterns, anomalies, and potential conditions in medical images. With cutting-edge machine learning, it streamlines workflow and supports informed decision-making in healthcare.

## Dependencies and APIs
#### **1. Claude API**
#### **2. PyTorch**
#### **3. numpy**
#### **4. requests**
#### **5. cv2**
#### **6. PIL (Python Image Library)**
#### **7. Django**
#### **8. Transformers (from hugging face)**
#### **9. os**
#### **10. Custom Functions: analyze_image_claude and MedicalImage model.**

## Installation Guide 

### **1. Create a Virtual Environment (PLEASE)**
```bash
python -m venv medical_ai_env
source medical_ai_env/bin/activate  # macOS/Linux
medical_ai_env\Scripts\activate   # Windows   
```
### **2. Install Dependencies**
```bash 
pip install torch torchvision torchaudio pillow django transformers numpy openai
```
### **3. Install PyTorch**
#### For CPU
```bash 
pip install torch torchvision torchaudio
```
#### For GPU (CUDA 11.8+)
```bash 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
### **4. Install Additional Dependencies**
```bash
pip install django-core-files-storage python-dotenv
```

## Contributor
### **1. Miraj Adhikari, Github: https://github.com/MirajAdhikari**
