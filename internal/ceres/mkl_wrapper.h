//
// Created by victor on 23.09.22.
//

#ifndef CERES_MKL_WRAPPER_H
#define CERES_MKL_WRAPPER_H

#include "ceres/internal/config.h"

#ifdef CERES_WITH_MKL
#include <mkl_sparse_qr.h>

#include <stddef.h>

namespace ceres::internal::mkl_wrapper {
sparse_status_t create_csr_impl(sparse_matrix_t* A,
                                const MKL_INT rows,
                                const MKL_INT cols,
                                MKL_INT* rows_start,
                                MKL_INT* rows_end,
                                MKL_INT* col_indx,
                                float* values);
sparse_status_t create_csr_impl(sparse_matrix_t* A,
                                const MKL_INT rows,
                                const MKL_INT cols,
                                MKL_INT* rows_start,
                                MKL_INT* rows_end,
                                MKL_INT* col_indx,
                                double* values);

sparse_status_t qr_factorize_impl(sparse_matrix_t A, float* alt_values);
sparse_status_t qr_factorize_impl(sparse_matrix_t A, double* alt_values);

sparse_status_t qr_solve_impl(sparse_matrix_t A,
                              int cols, int rows,
                              const float* b,
                              float* x);
sparse_status_t qr_solve_impl(sparse_matrix_t A,
                              int cols, int rows,
                              const double* b,
                              double* x);

void cvtpd_ps(const double* src, float* dst, size_t length);

void cvtps_pd(const float* src, double* dst, size_t length);
}

#endif

#endif  // CERES_MKL_WRAPPER_H
