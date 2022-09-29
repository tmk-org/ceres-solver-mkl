//
// Created by victor on 21.09.22.
//

#include "ceres/sparse_cholesky.h"
#include "mkl_sparse.h"

#ifdef CERES_WITH_MKL
#include "mkl_wrapper.h"

//#include "Eigen/SparseCore"
namespace {
template<typename RealType>
std::unique_ptr<ceres::internal::SparseCholesky> CreateImpl(ceres::internal::OrderingType orderingType);
}

namespace ceres::internal {

template <typename RealType, OrderingType _ordering_type>
class CERES_NO_EXPORT MklSparseCholeskyBase final : public SparseCholesky {
 public:
  explicit MklSparseCholeskyBase() = default;
  // SparseCholesky interface.
  ~MklSparseCholeskyBase() override {
    if (_analysed) {
      mkl_sparse_destroy(_csrA);
    }
  }

  CompressedRowSparseMatrix::StorageType StorageType() const final {
    return CompressedRowSparseMatrix::StorageType::UNSYMMETRIC;
  }

  LinearSolverTerminationType Factorize(CompressedRowSparseMatrix* lhs,
                                        std::string* message) final {
    RealType* values = nullptr;
    
    if (_cols && _rows && (_capacity != lhs->num_nonzeros() ||
                      _rows != lhs->num_rows() ||
                      _cols != lhs->num_cols())) {
      LOG(WARNING) << "Matrix with different structure passed into initialised "
                      "MKL Sparse QR based solver. Reinitialise the solver...";
      _cols = 0;
      _rows = 0;
      mkl_sparse_destroy(_csrA);
    }
    
    if constexpr (std::is_same_v<RealType, double>) {
      values = lhs->mutable_values();
    } 
    else {
      if (std::is_same_v<RealType, float>) {
        if (_values.size() < lhs->num_nonzeros()) {
          _values.clear();
          _values.resize(lhs->num_nonzeros());
        }
        mkl_wrapper::cvtpd_ps(lhs->values(), values, lhs->num_nonzeros());
        values = _values.data();
      } else {
        LOG(FATAL)
            << "MKL QR Sparse solver supports only float and double types.";
        return LinearSolverTerminationType::FATAL_ERROR;
      }
    }
    if(lhs->num_rows() == 1 && lhs->num_cols() == 1) {
      LOG(WARNING) << "Try to factorize 1x1 matrix (scalar number)";

      if constexpr (std::is_same_v<RealType, double>) {
        _scalar_lhs = *lhs->values();
      }
      else {
        if(std::is_same_v<RealType, float>) {
          _scalar_lhs = static_cast<float>(*lhs->values());
        }
        else {
          LOG(FATAL) << "MKL QR Sparse solver supports only float and double types.";
        }
      }
      _cols = 1;
      _rows = 1;
      return LinearSolverTerminationType::SUCCESS;
    }

    if (!_cols) {
      if (!create_csr(&_csrA,
                      lhs->num_rows(),
                      lhs->num_cols(),
                      lhs->mutable_rows(),
                      lhs->mutable_rows() + 1,
                      lhs->mutable_cols(),
                      values)) {
        sparseStatusToString(message);
        return LinearSolverTerminationType::FATAL_ERROR;
      }
      _result = mkl_sparse_qr_reorder(_csrA, {
                                                 .type = SPARSE_MATRIX_TYPE_GENERAL,
                                                 .mode = SPARSE_FILL_MODE_UPPER});
      if(_result != SPARSE_STATUS_SUCCESS) {
        sparseStatusToString(message);
        mkl_sparse_destroy(_csrA);
        return LinearSolverTerminationType::FATAL_ERROR;
      }
      _cols = lhs->num_cols();
      _rows = lhs->num_rows();
      _capacity = lhs->num_nonzeros();
      if (std::is_same_v<RealType, float>) {
        _rhs_values.resize(_rows);
        _sol_values.resize(_cols);
      }
    }

    if (!qr_factorize(values)) {
      sparseStatusToString(message);
      return LinearSolverTerminationType::FATAL_ERROR;
    }
    return LinearSolverTerminationType::SUCCESS;
  }

  LinearSolverTerminationType Solve(const double* rhs,
                                    double* solution,
                                    std::string* message) final {
    CHECK(_cols) << "Solve called without a call to Factorize first.";
    if(_cols == 1 && _rows == 1) {
      CHECK(_scalar_lhs) << "Try to solve 0*x=b";
      if constexpr (std::is_same_v<RealType, double>) {
        *solution = *rhs/_scalar_lhs;
      }
      else {
        if(std::is_same_v<RealType, float>) {
          *solution = static_cast<float>(*rhs)/_scalar_lhs;
        }
        else {
          LOG(FATAL) << "MKL QR Sparse solver supports only float and double types.";
        }
      }
      return LinearSolverTerminationType::SUCCESS;
    }
    if constexpr (std::is_same_v<RealType, double>) {
      if(!qr_solve(rhs, solution)) {
        sparseStatusToString(message);
        return LinearSolverTerminationType::FATAL_ERROR;
      }
    }
    else {
      if(std::is_same_v<RealType, float>) {
        mkl_wrapper::cvtpd_ps(rhs, _rhs_values.data(), _rows);
        if(!qr_solve(_rhs_values.data(), _sol_values.data())) {
          sparseStatusToString(message);
          return LinearSolverTerminationType::FATAL_ERROR;
        }
        mkl_wrapper::cvtps_pd(_sol_values.data(), solution, _cols);
      }
      else {
        LOG(FATAL) << "MKL QR Sparse solver supports only float and double types.";
      }
    }
    return LinearSolverTerminationType::SUCCESS;
  }

 private:
  void sparseStatusToString(std::string *message) {
    if (_result == SPARSE_STATUS_SUCCESS || !message) {
      return;
    }
    std::string reason;
    switch (_result) {
      case SPARSE_STATUS_NOT_INITIALIZED:
        reason = " empty handle or matrix arrays.";
        break;
      case SPARSE_STATUS_ALLOC_FAILED:
        reason = " internal error: memory allocation failed.";
        break;
      case SPARSE_STATUS_INVALID_VALUE:
        reason = " invalid input value";
        break;
      case SPARSE_STATUS_EXECUTION_FAILED:
        reason =
            " execution failed: e.g. 0-diagonal element for triangular solver, etc.";
        break;
      case SPARSE_STATUS_INTERNAL_ERROR:
        reason = " internal error.";
      case SPARSE_STATUS_NOT_SUPPORTED:
        reason =
            " unsupported operation e.g. operation for double precision doesn't support other types.";
        break;
      default:
        reason = " unidentified. Check MKL and Ceres versions";
    }
    *message = "MKL Sparse QR solver error:" + reason;
    LOG(INFO) << *message;
  }

  // mkl_sparse_?_ routines
  bool create_csr(sparse_matrix_t* A,
                  const MKL_INT rows,
                  const MKL_INT cols,
                  MKL_INT* rows_start,
                  MKL_INT* rows_end,
                  MKL_INT* col_indx,
                  RealType* values) {
    _result =
        mkl_wrapper::create_csr_impl(A, rows, cols, rows_start, rows_end, col_indx, values);
    return _result == SPARSE_STATUS_SUCCESS;
  }

  bool qr_factorize(RealType *alt_values) {
    _result = mkl_wrapper::qr_factorize_impl(_csrA, alt_values);
    return _result == SPARSE_STATUS_SUCCESS;
  }

  bool qr_solve(const RealType *b, RealType *x) {
    _result = mkl_wrapper::qr_solve_impl(_csrA, _cols, _rows, b, x);
    return _result == SPARSE_STATUS_SUCCESS;
  }

  // mkl_sparse_?_ routines templates finished

  bool _analysed{false};

  sparse_matrix_t _csrA;

  sparse_status_t _result{SPARSE_STATUS_SUCCESS};

  int _cols{0};
  int _rows{0};
  int _capacity{0};

  std::vector<RealType> _values;
  std::vector<RealType> _rhs_values;
  std::vector<RealType> _sol_values;

  RealType _scalar_lhs;
};

std::unique_ptr<SparseCholesky> internal::MklSparseCholesky::Create(
    OrderingType ordering_type) {
  return CreateImpl<double>(ordering_type);
}

std::unique_ptr<SparseCholesky> FloatMklSparseCholesky::Create(
    OrderingType ordering_type) {
  return CreateImpl<float>(ordering_type);
}

}

namespace {

template<typename RealType>
std::unique_ptr<ceres::internal::SparseCholesky> CreateImpl(ceres::internal::OrderingType ordering_type) {
  switch (ordering_type) {
    case ceres::internal::OrderingType::NATURAL:
      return std::make_unique<ceres::internal::MklSparseCholeskyBase<
          RealType,
          ceres::internal::OrderingType::NATURAL>>();
    case ceres::internal::OrderingType::AMD:
      return std::make_unique<ceres::internal::MklSparseCholeskyBase<
          RealType,
          ceres::internal::OrderingType::AMD>>();
    case ceres::internal::OrderingType::NESDIS:
      return std::make_unique<ceres::internal::MklSparseCholeskyBase<
          RealType,
          ceres::internal::OrderingType::NESDIS>>();
    default: {
      LOG(FATAL) << "New ordering type is not supported by solver";
      return nullptr;
    }
  }
}

}

#else
std::unique_ptr<SparseCholesky> ceres::internal::MklSparseCholesky::Create(
    OrderingType ordering_type) {
  LOG(FATAL) << "Ceres built without MKL extensions";
  return nullptr;
}

std::unique_ptr<SparseCholesky> ceres::internal::FloatMklSparseCholesky::Create(
    OrderingType ordering_type) {
  LOG(FATAL) << "Ceres built without MKL extensions";
  return nullptr;
}
#endif
