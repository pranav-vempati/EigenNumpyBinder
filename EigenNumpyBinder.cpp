
#include <iostream>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/complex.h>
#include <Eigen/LU>
#include <cassert>
#include <Eigen/QR>


namespace py = pybind11;

class MatrixWrapperEigenvector{

public:
    MatrixWrapperEigenvector(const Eigen::MatrixXd& mat);

    Eigen::VectorXcd extractFirstEigenvector();
private:
    Eigen::MatrixXd matrix;

};

MatrixWrapperEigenvector::MatrixWrapperEigenvector(const Eigen::MatrixXd& mat) : matrix(mat) {}

Eigen::VectorXcd MatrixWrapperEigenvector::extractFirstEigenvector() {
    Eigen::EigenSolver<Eigen::MatrixXd> solver(matrix);
    return solver.eigenvectors().col(0);
}


class MatrixWrapperLU{

public:
    MatrixWrapperLU(const Eigen::MatrixXd& matA, const Eigen::MatrixXd& matB );

    Eigen::MatrixXd LUSolver();
private:
    Eigen::MatrixXd matrixA;
    Eigen::MatrixXd matrixB;

};

MatrixWrapperLU::MatrixWrapperLU(const Eigen::MatrixXd &matA, const Eigen::MatrixXd &matB) : matrixA(matA), matrixB(matB) {}


Eigen::MatrixXd MatrixWrapperLU::LUSolver() {
    Eigen::MatrixXd sol = matrixA.lu().solve(matrixB);
    assert(sol.determinant() != 0); // Meaningful for small matrices
    return sol;
}

class MatrixWrapperQR{

public:
    MatrixWrapperQR(const Eigen::MatrixXf& matA);

    Eigen::MatrixXf calculateHouseHolder();
private:
    Eigen::MatrixXf matrixA;

};

MatrixWrapperQR::MatrixWrapperQR(const Eigen::MatrixXf &matA) : matrixA(matA) {}


Eigen::MatrixXf MatrixWrapperQR::calculateHouseHolder() {
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(matrixA);
    Eigen::MatrixXf sol = qr.householderQ();
    return sol;
}

PYBIND11_MODULE(EigenNumpyBinder, m){
    m.doc() = "Python bindings for Eigen routines";
    py::class_<MatrixWrapperEigenvector>(m, "EigenVectorWrapper")
            .def(py::init<Eigen::MatrixXd>())
            .def("extract_first_eigenvector", &MatrixWrapperEigenvector::extractFirstEigenvector);
    py::class_<MatrixWrapperLU>(m, "LUWrapper")
            .def(py::init<Eigen::MatrixXd, Eigen::MatrixXd>())
            .def("lu_solver", &MatrixWrapperLU::LUSolver);
    py::class_<MatrixWrapperQR>(m, "QRWrapper")
            .def(py::init<Eigen::MatrixXf>())
            .def("calculate_householder", &MatrixWrapperQR::calculateHouseHolder);
}
