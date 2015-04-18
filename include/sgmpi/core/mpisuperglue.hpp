#ifndef SGMPI_MPISUPERGLUE_HPP_INCLUDED
#define SGMPI_MPISUPERGLUE_HPP_INCLUDED

#include <string>
#include <sstream>
#include "sg/superglue.hpp"
#include "sgmpi/core/mpitask.hpp"
#include "sgmpi/core/communicator.hpp"

namespace sg {
template <typename Options> class TaskExecutor;
}

namespace sgmpi {

template <typename Options> class MPISuperGlue;
template <typename Options> class MPIHandle;
template <typename Options> class Communicator;

template <typename Options>
struct SendDataTask : public MPITask<Options> {
    typedef typename Options::version_type version_type;

    Communicator<Options> &t;
    MPIHandle<Options> &handle;
    int transfer_id;
    int dest_rank;

    SendDataTask(Communicator<Options> &t_, MPIHandle<Options> &handle_, version_type required_version, int transfer_id_, int dest_rank_)
    : t(t_), handle(handle_), transfer_id(transfer_id_), dest_rank(dest_rank_)
    {
        this->is_prioritized = true;

        version_type transfer_version = t.send_handle.schedule(Options::AccessInfoType::write);
        this->fulfill(Options::AccessInfoType::write, t.send_handle, transfer_version);

        // TODO (HACK): wait for handle v[required_version]
        // We will not increase the version number after we are finished, since
        // we are actually not finished reading the data.
        // Instead, a FinishedSendTask will be submitted that will increase the
        // version when we are finished reading the data.
        this->fulfill(Options::AccessInfoType::read, handle, required_version);
    }

    void run(sg::TaskExecutor<Options> &te) {
        t.send(handle, transfer_id, dest_rank);

        // TODO (HACK): Prevent handle version from being increased.
        this->num_access = 1;
    }

    std::string get_name() {
        std::stringstream ss;
        ss << "SendDataTask to " << dest_rank << " id: " << transfer_id;
        return ss.str(); 
    }
};

template <typename Options>
class MPISuperGlue : public Options::TaskRankDecider {
    template<typename> friend class Communicator;

    typedef typename Options::AccessInfoType AccessInfo;
    typedef typename AccessInfo::Type AccessType;
    typedef typename Options::version_type version_type;

public:
    sg::SuperGlue<Options> sg;

public:
    int transfer_id;

private:
    bool has_local_copy(MPIHandle<Options> &handle, int rank) {
        if (handle.copies.empty())
            return false;
        return handle.copies[rank];
    }

    bool transfer_needed(MPIHandle<Options> &handle, int task_rank) {
        // if data is already local, there is no need to send
        if (handle.last_written_rank == task_rank)
            return false;
        if (has_local_copy(handle, task_rank))
            return false;
        return true;
    }

    void register_accesses(MPITask<Options> &task) {
        const int task_rank = task.get_rank();
        const int my_rank = get_rank();

        for (size_t i = 0; i < task.accesses.size(); ++i) {
            AccessType type(task.accesses[i].first);
            MPIHandle<Options> &handle(*task.accesses[i].second);

            if (transfer_needed(handle, task_rank)) {
                ++transfer_id;
                //fprintf(stderr, "%d: transfer_id %d is handle %d from %d to %d\n",
                //    rank, transfer_id, handle.get_global_id(), handle.last_written_rank, task_rank);

                if (my_rank == handle.last_written_rank) {
                    // need to send
                    const version_type required_version = handle.schedule(Options::AccessInfoType::read);
                    //fprintf(stderr, "%d: need-to-send handle %d to %d before '%s'\n", rank, handle.get_global_id(), task_rank, task.get_name().c_str());
                    sg.submit( new SendDataTask<Options>(sg.tman->mpicomm, handle, required_version, transfer_id, task_rank) );
                }
                else if (my_rank == task_rank) {
                    // need to receive
                    const version_type required_version = handle.schedule(Options::AccessInfoType::write);
                    const int sender_rank = handle.last_written_rank;
                    //fprintf(stderr, "%d: register_access(%s): need to receive handle %d (req v%d)\n", 
                    //    rank, task.get_name().c_str(), handle.get_global_id(), required_version);
                    sg.tman->mpicomm.recv(handle, required_version, transfer_id, sender_rank);
                }
                // update cache
                if (AccessUtil<Options>::readonly(type)) {
                    if (handle.copies.empty())
                        handle.copies.resize(get_num_ranks());
                    handle.copies[task_rank] = true;
                }
                else {
                    handle.last_written_rank = task_rank;
                    std::fill(handle.copies.begin(), handle.copies.end(), false);
                }
            }
            else {
                // update cache
                if (!AccessUtil<Options>::readonly(type)) {
                    handle.last_written_rank = task_rank;
                    std::fill(handle.copies.begin(), handle.copies.end(), false);
                }
            }

            if (task_rank == my_rank) {
                // local task

                // register locally and require local version
                version_type required_version = handle.schedule(type);

                // store access in the task
                task.fulfill(type, handle, required_version);
            }
        }
    }

private:
    MPISuperGlue(const MPISuperGlue &);
    MPISuperGlue &operator=(const MPISuperGlue &);

public:

    MPISuperGlue(ThreadingManagerMPI<Options> &tman_) : sg(tman_), transfer_id(2) {}
    ~MPISuperGlue() {
        sg.tman->mpicomm.terminate();
    }

    void submit(MPITask<Options> *task) {
        submit_to_rank(task, Options::TaskRankDecider::determine_task_rank(task));
    }
    void submit(MPITask<Options> *task, int core) {
        submit_to_rank(task, Options::TaskRankDecider::determine_task_rank(task), core);
    }

    void submit(TaskBase<Options> *task) {
        sg.submit(task);
    }
    void submit(TaskBase<Options> *task, int core) {
        sg.submit(task, core);
    }

    void submit_to_rank(MPITask<Options> *task, int task_rank, int core = -1) {
        assert(task_rank < get_num_ranks());
        task->owner_rank = task_rank;

        register_accesses(*task);

        if (task->owner_rank == get_rank()) {
            if (core == -1)
                sg.submit(task);
            else
                sg.submit(task, core);
        }
        else
            delete task;
    }

    void mpi_barrier() {
        sg.tman->mpicomm.mpi_barrier();
    }

    // wait until all tasks have been executed and the mpi queue is empty
    void barrier() {
        // on main thread 
        // only tasks and mpi communication can add new tasks

        TaskExecutor<Options> &main_task_executor(*sg.tman->get_worker(Options::ThreadingManagerType::MAIN_THREAD_ID));

        int &status(sg.tman->mpicomm.no_work_agreement);
        for (;;) {

            int local_status;

            // execute all tasks
            sg.tman->barrier_protocol.barrier(main_task_executor);

            // initialize agreement
            status = 1;
            Atomic::memory_fence_producer();

            // wait on MPI thread
            do {
                Atomic::rep_nop();
                local_status = status;
            } while (local_status == 1);

            if (local_status == 0) {
                // mpi events occurred -- must repeat the process
                continue;
            }

            // execute tasks that might have been created
            // by mpi thread before it saw that we initialized
            // an agreement.
            sg.tman->barrier_protocol.barrier(main_task_executor);

            // second pass -- to ensure no new mpi events have 
            // been created by those tasks
            status = 3;
            Atomic::memory_fence_producer();

            do {
                Atomic::rep_nop();
                local_status = status;
            } while (local_status == 3);

            if (local_status == 0) {
                // mpi events occurred -- must repeat the process
                continue;
            }

            return;
        }
    }

    /// Wait for MPI handles not supported!
private:
    void wait(MPIHandle<Options> &handle);
public:

    void wait(Handle<Options> &handle) {
        sg.wait(handle);
    }

    int get_rank() { 
        static const int rank( sg.tman->get_rank() );
        return rank;
    }
    int get_num_ranks() {
        static const int num_ranks( sg.tman->get_num_ranks() );
        return num_ranks;
    }
    int get_num_cpus() {
        static const int num_cpus( sg.tman->get_num_cpus() );
        return num_cpus;
    }
};

} // namespace sgmpi

#endif // SGMPI_MPISUPERGLUE_HPP_INCLUDED
