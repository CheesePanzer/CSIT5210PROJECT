# Kernel SVM Module
This directory contains the kernel-SVM reproduction pipeline for the OVO/OVR imbalanced multi-class classification project.
---
## Python Version
3.0+
---
## Training Setup for Linux/MacOS
1. Create NEW Python Virtual Environment at /src:
```
python3 -m venv .venv
```
2. Get into the Python Virtual Environment:
```
source .venv/bin/activate
```
3. Install required modules:
```
pip install -r requirements.txt 
```
4. Train OVO and OVR at /src:
```
python3 -m kernel-svm.svm
```
5. Generate results at /src:
```
python3 -m kernel-svm.generate_result
```