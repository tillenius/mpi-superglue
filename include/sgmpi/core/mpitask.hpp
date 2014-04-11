#ifndef SGMPI_MPITASK_HPP_INCLUDED
#define SGMPI_MPITASK_HPP_INCLUDED

#include "sg/core/task.hpp"
#include "sgmpi/core/mpihandle.hpp"

namespace sgmpi {

// This injects types so that we have
// MPITask<N> 
//  : Options::TaskType<N>::type = MPITaskDefault<N>
//  : TaskDefault<N>
//  : MPITaskBase
//  : Options::TaskBaseType = MPITaskBaseDefault
//  : TaskBaseDefault

// TaskBaseType is by default TaskBaseDefault<Options>

template<typename Options>
struct MPIAccess : public sg::Access<Options> {
    MPIHandle<Options> *get_handle() const {
        return (MPIHandle<Options> *) this->handle;
    }
};

template<typename Options>
class MPITaskBaseDefault : public sg::TaskBase<Options> {
    typedef typename Options::AccessInfoType AccessInfo;
    typedef typename AccessInfo::Type AccessType;
public:
    int owner_rank;
    std::vector< std::pair<AccessType, MPIHandle<Options> *> > accesses;

    MPITaskBaseDefault() : owner_rank(-1) {}

    int get_rank() { return owner_rank; }

    void store_access(AccessType type, MPIHandle<Options> &handle) {
        accesses.push_back(std::make_pair(type, &handle));
    }
    MPIAccess<Options> *get_access() const { return (MPIAccess<Options>*) this->access_ptr; }
    MPIAccess<Options> &get_access(size_t i) const { return *(MPIAccess<Options>*) &this->access_ptr[i]; }
};

// export Options::MPITaskBaseType as MPITaskBase (default: MPITaskBaseDefault<Options>)
template<typename Options> class MPITaskBase : public Options::MPITaskBaseType {};

template<typename Options, int N>
class MPITaskDefault : public sg::TaskAccessMixin<Options, MPITaskBase<Options>, N>  {
    typedef typename Options::AccessInfoType AccessInfo;
    typedef typename AccessInfo::Type AccessType;

public:
    void register_access(AccessType type, MPIHandle<Options> &handle) {
        MPITaskBase<Options>::store_access(type, handle);
    }
};

// export "Options::TaskType<>::type" (default: TaskDefault) as type Task
template<typename Options, int N = -1> class MPITask
 : public Options::template MPITaskType<N>::type {};

} // namespace sgmpi

#endif // SGMPI_MPITASK_HPP_INCLUDED
