#include <thrust/host_vector.h>

#include <cmath>
#include <iostream>

void compare_host_vectors(const thrust::host_vector<float>& vec1,
                          const thrust::host_vector<float>& vec2,
                          float tolerance = 10e-6) {
  static int count = 1;
  std::cout << "======== Comparing version " << count << " ========\n";

  if (vec1.size() != vec2.size()) {
    std::cerr << "Vectors have different sizes." << std::endl;
  }

  float maxDiff = 0.0f;
  for (size_t i = 0; i < vec1.size(); ++i) {
    float diff = std::abs(vec1[i] - vec2[i]);
    if (diff > maxDiff) {
      maxDiff = diff;
    }
  }

  std::cout << "The maximum absolute difference between the two vectors is: "
            << maxDiff << std::endl;

  bool res = maxDiff <= tolerance;
  if (res) {
    std::cout << "The two vectors are equal within the tolerance of "
              << tolerance << std::endl;
  } else {
    std::cout << "The two vectors are not equal within the tolerance of "
              << tolerance << std::endl;
  }

  std::cout << "\n\n";
  count++;
}