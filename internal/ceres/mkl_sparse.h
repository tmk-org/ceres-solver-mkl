//
// Created by victor on 21.09.22.
//

#ifndef CERES_INTERNAL_MKL_SPARSE_H_
#define CERES_INTERNAL_MKL_SPARSE_H_

#include "ceres/internal/config.h"


#include "ceres/block_structure.h"
#include "ceres/internal/disable_warnings.h"
#include "ceres/linear_solver.h"
#include "ceres/sparse_cholesky.h"

#include "glog/logging.h"

namespace ceres::internal {

class CompressedRowSparseMatrix;
class TripletSparseMatrix;

class CERES_NO_EXPORT MklSparseCholesky final : public SparseCholesky {
 public:
  static std::unique_ptr<SparseCholesky> Create(OrderingType ordering_type);

  // SparseCholesky interface.
  ~MklSparseCholesky() override;
  CompressedRowSparseMatrix::StorageType StorageType() const final;
  LinearSolverTerminationType Factorize(CompressedRowSparseMatrix* lhs,
                                        std::string* message) final;
  LinearSolverTerminationType Solve(const double* rhs,
                                    double* solution,
                                    std::string* message) final;

 private:
  explicit MklSparseCholesky();
};

class CERES_NO_EXPORT FloatMklSparseCholesky final : public SparseCholesky {
 public:
  static std::unique_ptr<SparseCholesky> Create(OrderingType ordering_type);

  // SparseCholesky interface.
  ~FloatMklSparseCholesky() override;
  CompressedRowSparseMatrix::StorageType StorageType() const final;
  LinearSolverTerminationType Factorize(CompressedRowSparseMatrix* lhs,
                                        std::string* message) final;
  LinearSolverTerminationType Solve(const double* rhs,
                                    double* solution,
                                    std::string* message) final;

 private:
  explicit FloatMklSparseCholesky();
};

}

#endif  // CERES_INTERNAL_MKL_SPARSE_H_
