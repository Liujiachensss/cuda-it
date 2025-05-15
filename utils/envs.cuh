#pragma once

#include <cstdio>

void envs_set_maxconnections(int maxcon) {
  char set_maxconnect[256];
  sprintf(set_maxconnect, "CUDA_DEVICE_MAX_CONNECTIONS=%d", maxcon);
  putenv(set_maxconnect);
}