#include "sgmpi/superglue_mpi.hpp"

#include "sg/option/instr_trace.hpp"
#include "sgmpi/option/mpiinstr.hpp"

#include <iostream>
#include <string>
#include <cmath>
#include <mkl.h>

using namespace std;

using sgmpi::MPIHandle;

//===========================================================================
// Task Library Options
//===========================================================================
struct Options : public sgmpi::DefaultOptions<Options> {
    typedef Enable TaskName;
    typedef Trace<Options> Instrumentation;
    typedef sgmpi::MPITrace<Options> MPIInstrumentation;
};

//===========================================================================
// BlockedMatrix
//===========================================================================
struct BlockedMatrix {
    sgmpi::MPIHandle<Options> *handles;
    size_t dim;
    
    BlockedMatrix() : handles(0) {}
    
    void allocate(size_t m, size_t n, sgmpi::MPISuperGlue<Options> &mpisg) {
        handles = new MPIHandle<Options>[m*n];
        dim = n; 
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                operator()(i,j).set_rank( (i*dim+j) % mpisg.sg.tman->get_num_ranks() );
    }
    ~BlockedMatrix() {
        delete [] handles;
    }
    sgmpi::MPIHandle<Options> &operator()(size_t i, size_t j) { return handles[i*dim+j]; }
    sgmpi::MPIHandle<Options> &operator()(size_t i) { return handles[i]; }
};

//===========================================================================
// Tasks
//===========================================================================
struct gemm : public sgmpi::MPITask<Options> {
    gemm(MPIHandle<Options> &a, MPIHandle<Options> &b, MPIHandle<Options> &c) {
        register_access(ReadWriteAdd::read, a);
        register_access(ReadWriteAdd::read, b);
        register_access(ReadWriteAdd::write, c);
    }
    void run(TaskExecutor<Options> &) {
        double *a(get_access(0).get_handle()->data);
        double *b(get_access(1).get_handle()->data);
        double *c(get_access(2).get_handle()->data);
        int nb=sqrt(get_access(0).get_handle()->size);
        double DONE=1.0, DMONE=-1.0;
        dgemm("N", "T", &nb, &nb, &nb, &DMONE, a, &nb, b, &nb, &DONE, c, &nb);
    }
    std::string get_name() { return "gemm"; }
};
struct syrk : public sgmpi::MPITask<Options> {
    syrk(MPIHandle<Options> &a, MPIHandle<Options> &b) {
        register_access(ReadWriteAdd::read, a);
        register_access(ReadWriteAdd::write, b);
    }
    void run(TaskExecutor<Options> &) {
        double *a(get_access(0).get_handle()->data);
        double *c(get_access(1).get_handle()->data);

        double DONE=1.0, DMONE=-1.0;
        int nb=sqrt(get_access(0).get_handle()->size);
        dsyrk("L", "N", &nb, &nb, &DMONE, a, &nb, &DONE, c, &nb);
    }
    std::string get_name() { return "syrk"; }
};
struct potrf : public sgmpi::MPITask<Options> {
    potrf(MPIHandle<Options> &a) {
        register_access(ReadWriteAdd::write, a);
    }
    void run(TaskExecutor<Options> &) {
        double *a(get_access(0).get_handle()->data);
        int info = 0;
        int nb=sqrt(get_access(0).get_handle()->size);
        dpotrf("L", &nb, a, &nb, &info);
    }
    std::string get_name() { return "potrf"; }
};
struct trsm : public sgmpi::MPITask<Options> {
    trsm(MPIHandle<Options> &a, MPIHandle<Options> &b) {
        register_access(ReadWriteAdd::read, a);
        register_access(ReadWriteAdd::write, b);
    }
    void run(TaskExecutor<Options> &) {
        double *a(get_access(0).get_handle()->data);
        double *b(get_access(1).get_handle()->data);
        double DONE=1.0;
        int nb=sqrt(get_access(0).get_handle()->size);
        dtrsm("R", "L", "T", "N", &nb, &nb, &DONE, a, &nb, b, &nb);
    }
    std::string get_name() { return "trsm"; }
};

//===========================================================================
// Create matrix
//===========================================================================
void create_pd_matrix(int rank, size_t dim, size_t block, BlockedMatrix &out) {

    int seed = 1234;
    size_t N = dim*block;
    double scale = 1.0/65536.0;

    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            if (out(i, j).get_rank() == rank) {
                double *data( out(i,j).data );
                // fill block with random
                for (size_t ii = 0; ii < block; ++ii) {
                    for (size_t jj = 0; jj < block; ++jj) {
                        seed = seed * 1664525 + 1013904223;
                        data[ii*block+jj] = scale * (seed % 65536);
                    }
                }
                // make it positive definite
                if (i == j) {
                    for (size_t ii = 0; ii < block; ++ii) {
                        data[ii*block+ii] += N;
                    }
                }
            }
        }
    }
}

//===========================================================================
// Cholesky
//===========================================================================
void chol(sgmpi::MPISuperGlue<Options> &nm, size_t DIM, BlockedMatrix &A) {
    // cholesky
    for (size_t j = 0; j < DIM; j++) {
        for (size_t k = 0; k < j; k++) {
            for (size_t i = j+1; i < DIM; i++) {
                // A[i,j] = A[i,j] - A[i,k] * (A[j,k])^t
                nm.submit(new gemm(A(i, k), A(j, k), A(i, j)));
            }
        }
        for (size_t i = 0; i < j; i++) {
            // A[j,j] = A[j,j] - A[j,i] * (A[j,i])^t
            nm.submit(new syrk(A(j, i), A(j, j)));
        }

        // Cholesky Factorization of A[j,j]
        nm.submit(new potrf(A(j, j)));

        for (size_t i = j+1; i < DIM; i++) {
            // A[i,j] <- A[i,j] = X * (A[j,j])^t
            nm.submit(new trsm(A(j, j), A(i, j)));
        }
    }
}

//===========================================================================
// Main
//===========================================================================
void cholmain(sgmpi::MPISuperGlue<Options> &nm) {

    const size_t DIM = 5;
    const size_t BLOCK = 900;

    const int rank = nm.get_rank();

    BlockedMatrix A;
    A.allocate(DIM, DIM, nm);

    double *data = new double[DIM*DIM*BLOCK*BLOCK];
    size_t index = 0;

    for (size_t i = 0; i < DIM; ++i) {
        for (size_t j = 0; j < DIM; ++j) {
            A(i, j).data = &data[index * BLOCK*BLOCK];
            A(i, j).size = BLOCK * BLOCK;
            index++;
        }
    }

    create_pd_matrix(rank, DIM, BLOCK, A);

    chol(nm, DIM, A);

    nm.wait(A(DIM-1, DIM-1));

    delete [] data;

    std::stringstream ss;
    ss << "trace-" << rank << ".log";
    Log<Options>::dump(ss.str().c_str(), rank);
}

int main() {
    mpisuperglue(cholmain);
    return 0;
}
