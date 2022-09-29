#include "ceres/mkl_wrapper.h"

#ifdef CERES_WITH_MKL

#include <emmintrin.h>

namespace ceres::internal::mkl_wrapper {
sparse_status_t create_csr_impl(sparse_matrix_t* A,
                                const MKL_INT rows,
                                const MKL_INT cols,
                                MKL_INT* rows_start,
                                MKL_INT* rows_end,
                                MKL_INT* col_indx,
                                float* values) {
  return mkl_sparse_s_create_csr(A,
                                 SPARSE_INDEX_BASE_ZERO,
                                 rows,
                                 cols,
                                 rows_start,
                                 rows_end,
                                 col_indx,
                                 values);
}
sparse_status_t create_csr_impl(sparse_matrix_t* A,
                                const MKL_INT rows,
                                const MKL_INT cols,
                                MKL_INT* rows_start,
                                MKL_INT* rows_end,
                                MKL_INT* col_indx,
                                double* values) {
  return mkl_sparse_d_create_csr(A,
                                 SPARSE_INDEX_BASE_ZERO,
                                 rows,
                                 cols,
                                 rows_start,
                                 rows_end,
                                 col_indx,
                                 values);
}

sparse_status_t qr_factorize_impl(sparse_matrix_t A, float* alt_values) {
  return mkl_sparse_s_qr_factorize(A, alt_values);
}

sparse_status_t qr_factorize_impl(sparse_matrix_t A, double* alt_values) {
  return mkl_sparse_d_qr_factorize(A, alt_values);
}

sparse_status_t qr_solve_impl(sparse_matrix_t A,
                              int cols, int rows,
                              const float* b,
                              float* x) {
  return mkl_sparse_s_qr_solve(SPARSE_OPERATION_NON_TRANSPOSE,
                               A,
                               nullptr,
                               SPARSE_LAYOUT_COLUMN_MAJOR,
                               1,
                               x,
                               cols,
                               b,
                               rows);
}

sparse_status_t qr_solve_impl(sparse_matrix_t A,
                              int cols, int rows,
                              const double* b,
                              double* x) {
  return mkl_sparse_d_qr_solve(SPARSE_OPERATION_NON_TRANSPOSE,
                               A,
                               nullptr,
                               SPARSE_LAYOUT_COLUMN_MAJOR,
                               1,
                               x,
                               cols,
                               b,
                               rows);
}

void cvtpd_ps(const double* src, float* dst, size_t length) {
  size_t accurate_length = length - length % 4;
  for (size_t i = 0; i < accurate_length; i += 4) {
    __m128d s1 = _mm_load_pd((src + i));
    __m128d s2 = _mm_load_pd((src + i + 2));

    __m128 d = _mm_movelh_ps(_mm_cvtpd_ps(s1), _mm_cvtpd_ps(s2));

    _mm_store_ps((dst + i), d);
  }

  for (size_t j = accurate_length; j < length; ++j) {
    dst[j] = static_cast<float>(src[j]);
  }
}

void cvtps_pd(const float* src, double* dst, size_t length) {
  size_t accurate_length = length - length % 4;
  for (size_t i = 0; i < accurate_length; i += 4) {
    __m128 s1 = _mm_load_ps((src + i));

    _mm_store_pd(dst + i, _mm_cvtps_pd(s1));
    _mm_store_pd(dst + i + 2, _mm_cvtps_pd(_mm_movehl_ps(s1, s1)));
  }

  for (size_t j = accurate_length; j < length; ++j) {
    dst[j] = static_cast<double>(src[j]);
  }
}
}  // namespace ceres::internal::mkl_wrapper

#endif