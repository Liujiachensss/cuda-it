#pragma once

#include <thrust/host_vector.h>

#include <cmath>
#include <iostream>

namespace utils {

void compare(const thrust::host_vector<float> &vec1,
             const thrust::host_vector<float> &vec2, float tolerance = 10e-6) {
  static int count = 1;
  std::cout << "======== Comparing version " << count << " ========\n";

  if (vec1.size() != vec2.size()) {
    std::cerr << "Vectors have different sizes." << std::endl;
  }

  float maxDiff = 0.0f; 
  float maxVal1 = vec1[0];
  float minVal1 = vec1[0];
  float maxVal2 = vec2[0];
  float minVal2 = vec2[0];
  for (size_t i = 0; i < vec1.size(); ++i) {
    if (vec1[i] > maxVal1) {
      maxVal1 = vec1[i];
    }
    if (vec1[i] < minVal1) {
      minVal1 = vec1[i];
    }
    if (vec2[i] > maxVal2) {
      maxVal2 = vec2[i];
    }
    if (vec2[i] < minVal2) {
      minVal2 = vec2[i];
    }

    float diff = std::abs(vec1[i] - vec2[i]);
    if (diff > maxDiff) {
      maxDiff = diff;
    }
  }

  printf("The value range of the first vector is: [%f, %f]\n", minVal1, maxVal1);
  printf("The value range of the second vector is: [%f, %f]\n", minVal2, maxVal2);

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

} // namespace utils