#include "sentiment_test_vectors.h"
#include <iostream>
#include <iomanip>
#include <onnxruntime_cxx_api.h>

#define MODEL_PATH L"full_keras_model.onnx"

int get_max_index(const float *buffer, int length);
void print_vector(const float *vec, int length);

const char *sentiment_labels[] = {"Extreme Negative", "Strong Negative",
                                  "Moderate Negative", "Positive",
                                  "Strong Positive"};

int main() {
  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    Ort::Session session(env, MODEL_PATH, session_options);
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    std::vector<const char *> input_names{input_name.get()};
    std::vector<const char *> output_names{output_name.get()};

    std::vector<int64_t> input_shape{1, NUM_INPUTS};

    for (int i = 0; i < NUM_SAMPLES; i++) {
      const float *input_data = &test_vectors[i].input[0];

      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          memory_info, const_cast<float *>(input_data), NUM_INPUTS,
          input_shape.data(), input_shape.size());

      auto output_tensors =
          session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                      &input_tensor, 1, output_names.data(), 1);

      float *output_data = output_tensors[0].GetTensorMutableData<float>();

      int predicted_class = get_max_index(output_data, 5);
      int expected_class = test_vectors[i].expected;

      std::cout << "Test case " << i << ":\n";
      std::cout << "Text: " << test_vectors[i].text << "\n";
      std::cout << "Expected Classification: " << expected_class << ", "
                << sentiment_labels[expected_class] << "\n";
      std::cout << "Predicted Classification: " << predicted_class << ", "
                << sentiment_labels[predicted_class] << "\n";
      std::cout << "Predicted Confidence: " << std::fixed
                << std::setprecision(6) << output_data[predicted_class] << "\n";

      std::cout << "Predicted Output: ";
      print_vector(output_data, 5);
      std::cout << "\n\n";
    }

  } catch (const Ort::Exception &e) {
    std::cerr << "ONNX Runtime exception: " << e.what() << "\n";
    return -1;
  } catch (const std::exception &e) {
    std::cerr << "Standard exception: " << e.what() << "\n";
    return -1;
  }

  return 0;
}

int get_max_index(const float *buffer, int length) {
  int max_index = 0;
  float max_value = buffer[0];
  for (int i = 1; i < length; ++i) {
    if (buffer[i] > max_value) {
      max_value = buffer[i];
      max_index = i;
    }
  }
  return max_index;
}

void print_vector(const float *vec, int length) {
  for (int i = 0; i < length; ++i) {
    std::cout << std::fixed << std::setprecision(6) << vec[i] << " ";
  }
  std::cout << "\n";
}