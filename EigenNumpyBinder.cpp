
#include <iostream>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

class MatrixWrapper{

public:
    MatrixWrapper(const Eigen::MatrixXd& mat);

    Eigen::VectorXcd extractFirstEigenvector();
private:
    Eigen::MatrixXd matrix;

};

MatrixWrapper::MatrixWrapper(const Eigen::MatrixXd& mat) : matrix(mat) {}

Eigen::VectorXcd MatrixWrapper::extractFirstEigenvector() {
    Eigen::EigenSolver<Eigen::MatrixXd> solver(matrix);
    return solver.eigenvectors().col(0);
}


PYBIND11_MODULE(EigenNumpyBinder, m){
    m.doc() = "Eigen/NumPy bindings";
    py::class_<MatrixWrapper>(m, "MatrixWrapper")
    .def(py::init<Eigen::MatrixXd>())
    .def("extract_first_eigenvector", &MatrixWrapper::extractFirstEigenvector);
}