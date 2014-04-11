#ifndef SGMPI_MPIDEBUGDEFAULTS_HPP_INCLUDED
#define SGMPI_MPIDEBUGDEFAULTS_HPP_INCLUDED

#include "sgmpi/core/mpidefaults.hpp"
#include "sgmpi/option/mpiinstr.hpp"

namespace sgmpi {

template <typename Options>
struct DebugOptions : public sgmpi::DefaultOptions<Options> {
    typedef Trace<Options> Instrumentation;
    typedef MPITrace<Options> MPIInstrumentation;
};

} // namespace sgmpi

#endif // SGMPI_MPIDEBUGDEFAULTS_HPP_INCLUDED
