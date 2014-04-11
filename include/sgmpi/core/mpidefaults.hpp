#ifndef SGMPI_MPIDEFAULTS_HPP_INCLUDED
#define SGMPI_MPIDEFAULTS_HPP_INCLUDED

#include "sg/core/defaults.hpp"
#include "sgmpi/core/mpitask.hpp"
#include "sgmpi/core/mpihandle.hpp"
#include "sgmpi/core/mpithreadingmanager.hpp"
#include "sg/option/taskqueue_prio.hpp"
#include "sg/platform/gettime.hpp"

#include <unistd.h>
#include <dirent.h>
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <cassert>

namespace sgmpi {

template<typename Options> class MPISuperGlue;

template<typename Options>
struct RankDecider {
    int determine_rank(MPITaskBase<Options> &task) {
        MPISuperGlue<Options> *this_( static_cast<MPISuperGlue<Options>*>(this));

        // first write access is stored in task, if any
        for (size_t i = 0; i < task.accesses.size(); ++i)
            if (!sg::AccessUtil<Options>::readonly(task.accesses[i].first))
                return task.accesses[i].second->get_rank();

        // no write accesses, use first read access
        if (task.accesses.size() > 0)
            return task.accesses[0].second->get_rank();

        // use current rank
        return this_->sg.get_rank();
    }

    int determine_task_rank(MPITaskBase<Options> *task) {
        return determine_rank(*task);
    }
};

// Thread affinity
template<typename Options>
struct DefaultThreadAffinity {
    static int &get_rank() {
        static int rank;
        return rank;
    }
    static int &get_cores() {
        static int cores;
        return cores;
    }
    static void init() {}
    static void init(int rank, int cores) {
        get_rank() = rank;
        get_cores() = cores;
        //fprintf(stderr, "threadaffinity init: rank=%d, cores=%d\n", rank, cores+1);
    }
    static void pin_main_thread() {
#ifdef SINGLE_NODE
        int cpuid = get_cores()*get_rank();
#else
        int cpuid = 0;
#endif
        sg::affinity_cpu_set cpu_set;
        cpu_set.set(cpuid);
        sg::ThreadAffinity::set_affinity(cpu_set);
    }
    static void pin_workerthread(int id) {
#ifdef SINGLE_NODE
        int cpuid = get_cores()*get_rank()+id;
#else
        int cpuid = id;
#endif
        sg::affinity_cpu_set cpu_set;
        cpu_set.set(cpuid);
        sg::ThreadAffinity::set_affinity(cpu_set);
    }
    static void pin_mpi_thread() {
#ifdef SINGLE_NODE
        int cpuid = get_cores()*(get_rank()+1);
#else
        int cpuid = get_cores();
#endif
        sg::affinity_cpu_set cpu_set;
        cpu_set.set(cpuid-1);
        sg::ThreadAffinity::set_affinity(cpu_set);
        { // todo: hard-coded for linux
            // some MPI implementations create additional threads.
            // this code finds all threads, and set their affinity.
        DIR *dp;
        assert((dp = opendir("/proc/self/task")) != NULL);
        struct dirent *dirp;
        while ((dirp = readdir(dp)) != NULL) {
            if (dirp->d_name[0] == '.')
                continue;
            assert(sched_setaffinity(atoi(dirp->d_name), sizeof(cpu_set.cpu_set), &cpu_set.cpu_set) == 0);
        }
        closedir(dp);
    }
    }
};

template<typename Options>
class NoMPIInstrumentation {
public:
    class MPIInstrData {};
    static void start() {}
    template<typename Req> static void start(Req &) {}
    template<typename Req> static void recv(Req &, Time::TimeUnit stop) {}
    template<typename Req> static void send(Req &, Time::TimeUnit stop) {}
};

template <typename Options>
struct DefaultOptions : public sg::DefaultOptions<Options> {
    typedef DefaultOptions<Options> parent;
    typedef typename parent::Enable Enable;

    enum { limit_recv_per_node = 4 };
    enum { limit_send_per_node = 4 };

    // Use MPI handles, with owner information
    typedef MPIHandleBase<Options> MPIHandleType;

    // Use MPI tasks, with owner information
    typedef MPITaskBaseDefault<Options> MPITaskBaseType;
    template<int N> struct MPITaskType {
        typedef MPITaskDefault<Options, N> type;
    };

    // Use priorities
    typedef sg::TaskQueuePrio<Options> WaitListType;
    typedef sg::TaskQueuePrio<Options> ReadyListType;

    // Handles IDs must be enabled -- used to identify handles between nodes
    typedef Enable HandleId;

    // use integers for versions, as we want to send them in MPI messages as MPI_UNSIGNEDs
    typedef unsigned int version_t;

    // we always want the TaskExecutor in tasks
    typedef Enable PassTaskExecutor;

    typedef sg::NoInstrumentation<Options> Instrumentation;
    typedef NoMPIInstrumentation<Options> MPIInstrumentation;

    typedef RankDecider<Options> TaskRankDecider;

    typedef DefaultThreadAffinity<Options> ThreadAffinity;

    typedef ThreadingManagerMPI<Options> ThreadingManagerType;
};

} // namespace sgmpi

#endif // SGMPI_MPIDEFAULTS_HPP_INCLUDED
