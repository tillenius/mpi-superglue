#ifndef SGMPI_MPIHANDLE_HPP_INCLUDED
#define SGMPI_MPIHANDLE_HPP_INCLUDED

#include "sg/core/handle.hpp"

namespace sgmpi {

template <typename Options>
class MPIHandleBase : public sg::Handle<Options> {
    typedef typename Options::version_t version_t;
private:
    int owner_rank;

public:
    unsigned int size;
    double *data;

    int last_written_rank;
    std::vector<bool> copies;

    MPIHandleBase() : owner_rank(0), last_written_rank(0) {}

    void set_rank(int new_owner_rank) {
        owner_rank = new_owner_rank;
        last_written_rank = owner_rank;
    }
    int get_rank() { return owner_rank; }

    unsigned int get_size() { return size; }
    double *get_data() { return data; }
};

template<typename Options> class MPIHandle : public Options::MPIHandleType {};

} // namespace sgmpi

#endif // SGMPI_MPIHANDLE_HPP_INCLUDED
