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

-Note on Model Output and Sentiment Interpretation
While tokenization and class probabilities align with training labels, some semantic mismatches occur due to limited contextual understanding. For example, positive phrases like “A masterpiece” may be classified as Extreme Negative because the model lacks deeper context training.

This highlights a common limitation in supervised models trained on user-labeled data without extensive semantic context. Interpret outputs carefully and consider expanding training data or applying context-aware techniques for improved accuracy. Based on manual inspection of each phrase in sentiment_test_vectors.c against the numeric outputs, the model achieves approximately 70% accuracy, if you fudge the one-offs, you get up to 92%, but if you anylize each record, to see of the model understood the sentiment string, it gave the one-off fudge the nod, its about 85%. While the tokenization and vectorization correctly represent the input text, the model lacks sufficient contextual understanding to consistently align sentiment classifications with the nuanced meaning of each phrase.
