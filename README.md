# cv_project - Learning OpenCV & Computer Vision

A learning project focused on **Computer Vision with OpenCV**, without model training components. I write this for my teammate 

## Learning Objectives
- Learn how to use **OpenCV** and image processing libraries
- Understand fundamental and advanced image processing algorithms
- Practice using **Python virtual environment (venv)**
- Solve real-world computer vision problems

## Requirements
- Windows 10/11
- Python **3.11, 3.12, or 3.13** (Python 3.14 not supported - OpenCV not yet compatible)
- Recommended text editor (optional): VSCode + Extensions: **Python** (Microsoft), **Pylance**

---

## 1) Clone / Download the project
Open CMD and navigate to your desired project directory, for example:
```bat
cd C:\dev
git clone https://github.com/dquang05/computerVision.git
cd computerVision
```
## 2) Create and activate virtual environment (venv)
In the project directory (`computerVision`), run:

```bat
Check version: python --version
Create virtual environment: python -m venv .venv
Activate virtual environment: .venv\Scripts\activate
When activated, (.venv) should appear at the beginning of the command line
```
## 3) Install required libraries
```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
## 4) Verify Installation
```bat  
python -c "import cv2; import numpy as np; print('OpenCV:', cv2.__version__); print('Numpy:', np.__version__)"
```
## 5) Deactivate virtual environment
```bat
deactivate
```

---

## Project Structure
```
src/
├── Ex1/          # Exercise 1 - Image Processing Basics
│   ├── Question1/
│   ├── Question2/
│   ├── Question3/
│   ├── Question4/
│   └── Question5/
└── Ex2/          # Exercise 2 - Advanced Topics
    ├── Question1/
    ├── Question2/
    └── Question3/
```

## Usage Guide
1. Select an exercise from the `src/` directory
2. Open the corresponding Python file
3. Make sure venv is activated (should show `(.venv)` at the command line prompt)
4. Run the file: `python src/ExX/QuestionY/file.py`

## Main Libraries
- **OpenCV (cv2)** - Image processing and computer vision
- **NumPy** - Numerical computing
- **Other libraries** - See `requirements.txt`
