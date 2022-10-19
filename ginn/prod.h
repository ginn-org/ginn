// Copyright 2022 Bloomberg Finance L.P.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef GINN_PROD_H
#define GINN_PROD_H

#include <ginn/dev.h>
#include <ginn/except.h>
#include <ginn/tensor.h>

// GPU Prod routines

namespace ginn {
namespace internal {

#ifdef GINN_ENABLE_GPU

// TODO: Maybe make these static functions of Tensor class, and get rid of
//   data() method of Tensor? This file is the only thing that uses that method.

template <typename Scalar>
inline void gpu_gemm(cublasHandle_t& handle,
                     const Scalar* A,
                     const Scalar* B,
                     Scalar* C,
                     const int lda,
                     const int ldb,
                     const int ldc,
                     const int m,
                     const int k,
                     const int n,
                     const Scalar bet = 0,
                     const cublasOperation_t& op1 = CUBLAS_OP_N,
                     const cublasOperation_t& op2 = CUBLAS_OP_N) {
  // C = AB
  // A is mxk, B is kxn, C is mxn
  const Scalar alf(1);
  const Scalar* alpha = &alf;
  const Scalar* beta = &bet;

  if constexpr (std::is_same_v<Scalar, float>) {
    GINN_CUBLAS_CALL(cublasSgemm(
        handle, op1, op2, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
  } else if constexpr (std::is_same_v<Scalar, Half>) {
    auto alpha_ = reinterpret_cast<const __half*>(alpha);
    auto beta_ = reinterpret_cast<const __half*>(beta);
    auto A_ = reinterpret_cast<const __half*>(A);
    auto B_ = reinterpret_cast<const __half*>(B);
    auto C_ = reinterpret_cast<__half*>(C);
    GINN_CUBLAS_CALL(cublasHgemm(
        handle, op1, op2, m, n, k, alpha_, A_, lda, B_, ldb, beta_, C_, ldc));
  } else {
    GINN_THROW("Unexpected Scalar type in gpu_gemm!");
  }
}

template <typename Scalar>
inline void gpu_batched_gemm(cublasHandle_t& handle,
                             const Scalar* A,
                             const Scalar* B,
                             Scalar* C,
                             const int lda,
                             const int ldb,
                             const int ldc,
                             const int sA,
                             const int sB,
                             const int sC,
                             const int m,
                             const int k,
                             const int n,
                             const int batches,
                             const Scalar bet = Scalar(0),
                             const cublasOperation_t& op1 = CUBLAS_OP_N,
                             const cublasOperation_t& op2 = CUBLAS_OP_N) {
  const Scalar alf(1);
  const Scalar* alpha = &alf;
  const Scalar* beta = &bet;
  /*
  <T>gemmStridedBatched(cublasHandle_t handle,
                      cublasOperation_t transA, cublasOperation_t transB,
                      int M, int N, int K,
                      const T* alpha,
                      const T* A, int ldA, int strideA,
                      const T* B, int ldB, int strideB,
                      const T* beta,
                      T* C, int ldC, int strideC,
                      int batchCount)
  */

  if constexpr (std::is_same_v<Scalar, float>) {
    // clang-format off
    GINN_CUBLAS_CALL(cublasSgemmStridedBatched(
          handle,
          op1, op2,
          m, n, k,
          alpha,
          A, lda, sA,
          B, ldb, sB,
          beta,
          C, ldc, sC,
          batches));
    // clang-format on
  } else if constexpr (std::is_same_v<Scalar, Half>) {
    auto alpha_ = reinterpret_cast<const __half*>(alpha);
    auto beta_ = reinterpret_cast<const __half*>(beta);
    auto A_ = reinterpret_cast<const __half*>(A);
    auto B_ = reinterpret_cast<const __half*>(B);
    auto C_ = reinterpret_cast<__half*>(C);
    // clang-format off
    GINN_CUBLAS_CALL(cublasHgemmStridedBatched(
          handle,
          op1, op2,
          m, n, k,
          alpha_,
          A_, lda, sA,
          B_, ldb, sB,
          beta_,
          C_, ldc, sC,
          batches));
    // clang-format on
  } else {
    GINN_THROW("Unexpected Scalar type in gpu_batched_gemm!");
  }
}

enum class ProdResult { Assign, Add };
enum class ProdTranspose { None, First, Second, Both };

// C = A * B
template <typename Scalar>
inline void gpu_prod(Tensor<Scalar>& c,
                     Tensor<Scalar>& a,
                     Tensor<Scalar>& b,
                     ProdResult res = ProdResult::Assign,
                     ProdTranspose tra = ProdTranspose::None) {
  auto op1 = tra == ProdTranspose::First or tra == ProdTranspose::Both
                 ? CUBLAS_OP_T
                 : CUBLAS_OP_N;
  auto op2 = tra == ProdTranspose::Second or tra == ProdTranspose::Both
                 ? CUBLAS_OP_T
                 : CUBLAS_OP_N;

  Size a_outer, a_inner, b_outer, b_inner;
  if (op1 == CUBLAS_OP_T) {
    a_outer = a.cols(), a_inner = a.rows();
  } else {
    a_outer = a.rows(), a_inner = a.cols();
  }
  if (op2 == CUBLAS_OP_T) {
    b_outer = b.rows(), b_inner = b.cols();
  } else {
    b_outer = b.cols(), b_inner = b.rows();
  }

  GINN_ASSERT(a_inner == b_inner);
  GINN_ASSERT(a_outer == c.rows());
  GINN_ASSERT(b_outer == c.cols());

  gpu_gemm<Scalar>(cublas_handle(c.dev()->id().idx),
                   a.data(),
                   b.data(),
                   c.data(),
                   /* lda */ a.rows(),
                   /* ldb */ b.rows(),
                   /* ldc */ c.rows(),
                   /* m */ a_outer,
                   /* k */ a_inner,
                   /* n */ b_outer,
                   res == ProdResult::Add ? Scalar(1.) : Scalar(0.),
                   op1,
                   op2);
}

// C[:,:,i] = A[:,:,i] * B[:,:,i] ∀i
template <typename Scalar>
inline void gpu_batched_prod(Tensor<Scalar>& c,
                             Tensor<Scalar>& a,
                             Tensor<Scalar>& b,
                             ProdResult res = ProdResult::Assign,
                             ProdTranspose tra = ProdTranspose::None) {
  auto shape3 = [](const Tensor<Scalar>& t) {
    return Tensor<Scalar>::reduce(t.shape(), 3);
  };
  Shape a_shape = shape3(a), b_shape = shape3(b), c_shape = shape3(c);
  Size a_rows = a_shape[0], a_cols = a_shape[1], batches = a_shape[2];
  Size b_rows = b_shape[0], b_cols = b_shape[1];
  Size c_rows = c_shape[0], c_cols = c_shape[1];

  auto op1 = tra == ProdTranspose::First or tra == ProdTranspose::Both
                 ? CUBLAS_OP_T
                 : CUBLAS_OP_N;
  auto op2 = tra == ProdTranspose::Second or tra == ProdTranspose::Both
                 ? CUBLAS_OP_T
                 : CUBLAS_OP_N;

  Size a_outer, a_inner, b_outer, b_inner;
  if (op1 == CUBLAS_OP_T) {
    a_outer = a_cols, a_inner = a_rows;
  } else {
    a_outer = a_rows, a_inner = a_cols;
  }
  if (op2 == CUBLAS_OP_T) {
    b_outer = b_rows, b_inner = b_cols;
  } else {
    b_outer = b_cols, b_inner = b_rows;
  }

  GINN_ASSERT(a_inner == b_inner);
  GINN_ASSERT(a_outer == c_rows);
  GINN_ASSERT(b_outer == c_cols);

  gpu_batched_gemm<Scalar>(cublas_handle(c.dev()->id().idx),
                           a.data(),
                           b.data(),
                           c.data(),
                           /* lda */ a_rows,
                           /* ldb */ b_rows,
                           /* ldc */ c_rows,
                           /* sA */ a_rows * a_cols,
                           /* sB */ b_rows * b_cols,
                           /* sC */ a_outer * b_outer,
                           /* m */ a_outer,
                           /* k */ a_inner,
                           /* n */ b_outer,
                           batches,
                           res == ProdResult::Add ? Scalar(1.) : Scalar(0.),
                           op1,
                           op2);
}

#endif

} // namespace internal
} // namespace ginn

#endif
