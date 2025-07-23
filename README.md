# create_full_keras

Sentiment classification pipeline using TF-IDF, Keras, and ONNX Runtime.  
Test vectors are generated in C for inference inside a native Visual Studio project.

## Components
- Python scripts for training, ONNX conversion, and vector generation
- C++ test harness and header integration via ONNX Runtime
- Visual Studio project configured with DLL applocal and lib paths

## To regenerate models
Run Python scripts in this directory in sequence:
1. `train_tfidf_model.py`
2. `convert_to_onnx.py`
3. `generate_test_vectors.py`

## Notes
- All artifacts (`/models`, `/output`, etc.) are auto-created
- Built and tested with ONNX Runtime on Windows
- No makefile provided—Linux support TBD
