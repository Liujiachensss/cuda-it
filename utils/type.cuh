#pragma once

template <typename T>
using ccr_ptr = const T *const __restrict__;

template <typename T>
using cc_ptr = const T *const;

template <typename T>
using r_ptr = T *__restrict__;

template <typename T>
using _ptr = T *;

// template <typename T>
// using cr_ptr = const T * __restrict__;

template <typename T>
using rd_cr_ptr = const T *__restrict__;  // read only
