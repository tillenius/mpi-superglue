#ifndef SGMPI_COMMUNICATOR_HPP_INCLUDED
#define SGMPI_COMMUNICATOR_HPP_INCLUDED

#include "sg/platform/threads.hpp"
#include "sg/platform/gettime.hpp"

#include <sstream>
#include <algorithm>
#include <mpi.h>

namespace sgmpi {

template<typename Options>
class Communicator : public TaskExecutor<Options> {
    typedef typename Options::ReadyListType TaskQueue;
    typedef typename TaskQueue::unsafe_t TaskQueueUnsafe;
    typedef typename Options::version_type version_type;
    typedef typename Options::ThreadingManagerType ThreadingManager;

    struct FinishedSendTask : public sg::Task<Options> {

        FinishedSendTask(Handle<Options> &handle) {
            this->fulfill(Options::AccessInfoType::read, handle, 0);
            this->is_prioritized = true;
        }

        void run(TaskExecutor<Options> &te) {}

        std::string get_name() { return "FinishedSendTask"; }
    };

    struct PublishDataTask : public sg::Task<Options> {
        typedef typename Options::version_type version_type;

        MPIHandle<Options> &handle;
        double *buffer;

        PublishDataTask(MPIHandle<Options> &handle_, version_type required_version, double *data)
        : handle(handle_), buffer(data) {
            this->fulfill(Options::AccessInfoType::write, handle, required_version);
            this->is_prioritized = true;
        }
        void run(TaskExecutor<Options> &) {
            memcpy(handle.get_data(), buffer, sizeof(double) * handle.get_size());
            delete [] buffer;
        }
        std::string get_name() { return "PublishDataTask"; }
    };


    struct request_t : public Options::MPIInstrumentation::MPIInstrData {
        enum reqtype { invalid = 0, send, recv, event };

        reqtype type;
        MPIHandle<Options> *handle;
        version_type required_version;
        void *buffer;
        int transfer_id;
        int rank;

        request_t() : type(invalid) {}
        request_t(reqtype type_) : type(type_) {}

        request_t(MPIHandle<Options> *handle_, int transfer_id_, int dest_rank_)
        : type(send), handle(handle_), transfer_id(transfer_id_), rank(dest_rank_)
        {}

        request_t(MPIHandle<Options> *handle_, version_type rv, int transfer_id_, int sender_rank_)
        : type(recv), handle(handle_), required_version(rv), transfer_id(transfer_id_), rank(sender_rank_)
        {}

    };

    ThreadingManager &tman;

public:
    Handle<Options> send_handle;
    int rank;
    int num_ranks;

private:
    SpinLock recv_request_lock;
    std::deque<request_t> *recv_requests;
    SpinLock send_request_lock;
    std::deque<request_t> *send_requests;

    std::vector<MPI_Request> mpirequests;
    std::vector<request_t> requestinfo;
    std::vector<int> ongoing_recv;
    std::vector<int> ongoing_send;

    SpinLock signal_lock;
    bool signal_flag;
    bool shut_down;

public:
    char padding[64];
    int no_work_agreement;

private:
    void signal_event() {
        SpinLockScoped siglock(signal_lock);
        if (signal_flag)
            return;

        // no current signal, start a new one
        signal_flag = true;
    }

    void init() {
        rank = tman.get_rank();
        num_ranks = tman.get_num_ranks();

        recv_requests = new std::deque<request_t>[num_ranks];
        send_requests = new std::deque<request_t>[num_ranks];

        ongoing_recv.resize(num_ranks, 0);
        ongoing_send.resize(num_ranks, 0);

        assert( MPI_Barrier( MPI_COMM_WORLD ) == MPI_SUCCESS );

        Options::MPIInstrumentation::start();

        // unlock and allow wait_for_startup() to continue
        signal_lock.unlock();
    }

public:
    Communicator(ThreadingManager &tman_)
      : TaskExecutor<Options>(ThreadingManager::MPI_THREAD_ID, tman_),
       tman(tman_),
       recv_requests(0), send_requests(0),
       signal_flag(true), shut_down(false), no_work_agreement(0)
    {
        // take the signal lock here, and unlock first when init() is finished.
        signal_lock.lock();
    }
    ~Communicator() {
        delete [] recv_requests;
        delete [] send_requests;
    }

    // called by main thread
    void wait_for_startup() {
        // wait until signal_lock is released. It is taken in constructor and released when init() is finished.
        SpinLockScoped siglock(signal_lock);
    }

    // called by main thread
    void send(MPIHandle<Options> &handle, int transfer_id, int send_to) {
        bool need_to_signal;
        {
            SpinLockScoped scoped(send_request_lock);
            need_to_signal = ongoing_send[send_to] < Options::limit_send_per_node;
            send_requests[send_to].push_back(request_t(&handle, transfer_id, send_to));
        }
        if (need_to_signal)
            signal_event();
    }

    // called by main thread
    void notify(MPIHandle<Options> &handle, int transfer_id, int send_to) {
        bool need_to_signal;
        {
            SpinLockScoped scoped(send_request_lock);
            need_to_signal = ongoing_send[send_to] < Options::limit_send_per_node;
            send_requests[send_to].push_back(request_t::create_notify_request(&handle, transfer_id, send_to));
        }
        if (need_to_signal)
            signal_event();
    }

    // called by main thread
    void recv(MPIHandle<Options> &handle, version_type required_version, int transfer_id, int recv_from) {
        bool need_to_signal;
        {
            SpinLockScoped scoped(recv_request_lock);
            need_to_signal = ongoing_recv[recv_from] < Options::limit_send_per_node;
            recv_requests[recv_from].push_back(request_t(&handle, required_version, transfer_id, recv_from));
        }
        if (need_to_signal)
            signal_event();
    }

    void terminate() {
        // called by main thread
        SpinLockScoped siglock(signal_lock);
        shut_down = true;
        if (!signal_flag)
            signal_flag = true;
    }

    void mpi_barrier() {
        // can only be used when its known that mpi-superglue performs no mpi calls...
        assert( MPI_Barrier( MPI_COMM_WORLD ) == MPI_SUCCESS );
    }

    // called by mpi thread
    void operator()(sg::SuperGlue<Options> &sg) {
        init();

        std::vector<int> indices;

        for (;;) {

            // lock and check the requests queues

            bool local_signal_flag = signal_flag;
            if (local_signal_flag) {
                {
                    SpinLockScoped siglock(signal_lock);
                    signal_flag = false;
                }

                {
                    SpinLockScoped scoped(recv_request_lock);
                    for (int i = 0; i < num_ranks; ++i) {
                        if (ongoing_recv[i] >= Options::limit_recv_per_node)
                            continue;

                        if (recv_requests[i].empty())
                            continue;

                        requestinfo.push_back(recv_requests[i].front());
                        request_t &req(requestinfo.back());
                        recv_requests[i].pop_front();

                        const size_t size = req.handle->get_size();
                        req.buffer = (void *) new double[size];

                        MPI_Request mpireq;
                        Options::MPIInstrumentation::start(req);
                        assert( MPI_Irecv(req.buffer, size, MPI_DOUBLE, req.rank, 
                                          req.transfer_id, MPI_COMM_WORLD, &mpireq)
                                == MPI_SUCCESS);
                        mpirequests.push_back(mpireq);
                        ++ongoing_recv[i];
                    }
                }

                {
                    SpinLockScoped scoped(send_request_lock);
                    for (int i = 0; i < num_ranks; ++i) {
                        if (ongoing_send[i] >= Options::limit_send_per_node)
                            continue;

                        // finished request -- fill with new
                        if (send_requests[i].empty())
                            continue;

                        requestinfo.push_back(send_requests[i].front());
                        request_t &req(requestinfo[requestinfo.size()-1]);
                        send_requests[i].pop_front();

                        MPI_Request mpireq;
                        Options::MPIInstrumentation::start(req);
                        assert( MPI_Isend(req.handle->get_data(), req.handle->get_size(), MPI_DOUBLE,
                                          req.rank, req.transfer_id,  MPI_COMM_WORLD, &mpireq)
                                == MPI_SUCCESS );
                        mpirequests.push_back(mpireq);
                        ++ongoing_send[i];
                    }
                }
            }

            if (mpirequests.empty()) {

                // execute tasks
                {
                    TaskExecutor<Options> *this_(static_cast<TaskExecutor<Options> *>(this));
                    if (tman.barrier_protocol.update_barrier_state(*this_)) {
                        TaskQueueUnsafe woken;
                        if (TaskExecutor<Options>::execute_tasks(woken)) {
                            if (!woken.empty())
                                TaskExecutor<Options>::push_front_list(woken);
                            // tasks executed => restart barrier.
                            if (no_work_agreement != 0)
                                no_work_agreement = 0;
                        }
                    }
                }

                if (signal_flag)
                    continue;

                Atomic::compiler_fence(); // re-read no_work_agreement
                switch (no_work_agreement) {
                case 0:
                    // no barrier and no mpi requests. just loop.
                    continue;
                case 1:
                    // initialized, and no work seen.
                    no_work_agreement = 2;
                    continue;
                case 2:
                    // wait for other thread to respond
                    continue;
                case 3:
                    // barrier finalized.
                    no_work_agreement = 4;
                    break;
                }

                // no work seen since last barrier finished.

                // if not shutting down, just loop and wait for or another barrier.
                if (!shut_down)
                    continue;

                // shut down.
                assert(MPI_Finalize() == MPI_SUCCESS);
                return;
            }

            // mpi events in queue, restart barrier if initialized
            if (no_work_agreement != 0)
                no_work_agreement = 0;

            indices.resize(mpirequests.size());

            int outcount = 0;
            assert( MPI_Testsome(mpirequests.size(), &mpirequests[0], &outcount, &indices[0], MPI_STATUSES_IGNORE) == MPI_SUCCESS );
            Time::TimeUnit stop = Time::getTime();

            if (outcount > 0) {
                {
                    SpinLockScoped siglock(signal_lock);
                    if (!signal_flag)
                        signal_flag = true;
                }

                for (int i = 0; i < outcount; ++i) {
                    const int idx = indices[i];
                    const request_t req(requestinfo[idx]);

                    mpirequests[idx] = MPI_REQUEST_NULL;

                    switch (req.type) {
                        case request_t::recv: {
                            Options::MPIInstrumentation::recv(req, stop);
                            sg.submit( new PublishDataTask(*req.handle, req.required_version, (double*) req.buffer) );
                        }
                        continue;

                        case request_t::send: {
                            Options::MPIInstrumentation::send(req, stop);
                            sg.submit( new FinishedSendTask(*req.handle));
                        }
                        continue;

                        case request_t::invalid: // "should never happen"
                        default:
                            continue;
                    }
                }

                // erase finished transfers from mpirequests and requestinfo
                {
                    std::vector<request_t> new_requestinfo;
                    std::vector<MPI_Request> new_mpirequests;
                    new_requestinfo.reserve(requestinfo.size());
                    new_mpirequests.reserve(mpirequests.size());

                    for (size_t i = 0; i < mpirequests.size(); ++i) {
                        if (mpirequests[i] == MPI_REQUEST_NULL) {
                            if (requestinfo[i].type == request_t::recv)
                                --ongoing_recv[requestinfo[i].rank];
                            if (requestinfo[i].type == request_t::send)
                                --ongoing_send[requestinfo[i].rank];
                            continue;
                        }
                        new_requestinfo.push_back(requestinfo[i]);
                        new_mpirequests.push_back(mpirequests[i]);
                    }
                    std::swap(new_requestinfo, requestinfo);
                    std::swap(new_mpirequests, mpirequests);
                }
            }
            else {
                TaskExecutor<Options> *this_(static_cast<TaskExecutor<Options> *>(this));
                if (tman.barrier_protocol.update_barrier_state(*this_)) {
                    TaskQueueUnsafe woken;
                    TaskExecutor<Options>::execute_tasks(woken);
                    if (!woken.empty())
                        TaskExecutor<Options>::push_front_list(woken);
                }
            }
        }
    }
};

} // namespace sgmpi

#endif // SGMPI_COMMUNICATOR_HPP_INCLUDED
