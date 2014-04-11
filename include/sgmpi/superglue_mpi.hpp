#ifndef SGMPI_SUPERGLUE_MPI_HPP_INCLUDED
#define SGMPI_SUPERGLUE_MPI_HPP_INCLUDED

#include <mpi.h>
#include "sgmpi/core/mpidefaults.hpp"
#include "sgmpi/core/mpihandle.hpp"
#include "sgmpi/core/mpitask.hpp"
#include "sgmpi/core/mpisuperglue.hpp"
#include "sgmpi/core/mpithreadingmanager.hpp"

namespace sgmpi {

template<typename Options>
void mpisuperglue( void (*mainfn)(sgmpi::MPISuperGlue<Options> &) ) {
    ThreadingManagerMPI<Options> tman(mainfn);
}

} // namespace sgmpi

#endif // SGMPI_SUPERGLUE_MPI_HPP_INCLUDED
