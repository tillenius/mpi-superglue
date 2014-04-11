#ifndef SGMPI_COMMUNICATOR_HPP_INCLUDED
#define SGMPI_COMMUNICATOR_HPP_INCLUDED

#include "sg/platform/threads.hpp"
#include "sg/platform/gettime.hpp"

#include <sstream>
#include <algorithm>
#include <mpi.h>

//#define SGMPI_BUSYLOOP

namespace sgmpi {

template<typename Options>
class Communicator : public Options::Instrumentation {
    typedef typename Options::version_t version_t;
    typedef typename Options::ThreadingManagerType ThreadingManager;

    // FinishedSendTask
    struct FinishedSendTask : public sg::Task<Options> {

        FinishedSendTask(Handle<Options> &handle) {
            this->fulfill(Options::AccessInfoType::read, handle, 0);
            this->is_prioritized = true;
        }

        void run(TaskExecutor<Options> &te) {}

        std::string get_name() { return "FinishedSendTask"; }
    };

    // PublishDataTask
    struct PublishDataTask : public sg::Task<Options> {
        typedef typename Options::version_t version_t;

        MPIHandle<Options> &handle;
        double *buffer;

        PublishDataTask(MPIHandle<Options> &handle_, version_t required_version, double *data)
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
        enum reqtype { invalid = 0, send, recv, terminate, event };

        reqtype type;
        MPIHandle<Options> *handle;
        version_t required_version;
        void *buffer;
        int transfer_id;
        int rank;

        request_t() : type(invalid) {}
        request_t(reqtype type_) : type(type_) {}

        request_t(MPIHandle<Options> *handle_, int transfer_id_, int dest_rank_)
        : type(send), handle(handle_), transfer_id(transfer_id_), rank(dest_rank_)
        {}

        request_t(MPIHandle<Options> *handle_, version_t rv, int transfer_id_, int sender_rank_)
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

    MPI_Request send_req;

    SpinLock signal_lock;
    bool signal_flag;
    volatile bool ok_to_quit;

public:
    bool terminate_flag;

private:
    void signal_event() {

        SpinLockScoped siglock(signal_lock);
        if (signal_flag)
            return;

        // no current signal, start a new one
        signal_flag = true;

#ifndef SGMPI_BUSYLOOP
        // wait for old send, just to be careful. the receive is already successful
        assert( MPI_Wait( &send_req, MPI_STATUS_IGNORE ) == MPI_SUCCESS );
        assert( MPI_Start( &send_req ) == MPI_SUCCESS );
#endif // SGMPI_BUSYLOOP
    }

    void init() {
        rank = tman.get_rank();
        num_ranks = tman.get_num_ranks();

        recv_requests = new std::deque<request_t>[num_ranks];
        send_requests = new std::deque<request_t>[num_ranks];

        if (rank != 0) {
            // wait for termination
            MPI_Request term_req;
            assert( MPI_Irecv(0, 0, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &term_req) == MPI_SUCCESS);
            mpirequests.push_back(term_req);
            requestinfo.push_back(request_t(request_t::terminate));
        }

#ifndef SGMPI_BUSYLOOP
        // setup persistant request for waking MPI thread when desired.
        // start with an existing signal, to make it correct to always do a wait() on the last send.
        MPI_Request event_req;
        assert( MPI_Send_init(0, 0, MPI_INT, rank, 1, MPI_COMM_WORLD, &send_req) == MPI_SUCCESS );
        assert( MPI_Recv_init(0, 0, MPI_INT, rank, 1, MPI_COMM_WORLD, &event_req) == MPI_SUCCESS );
        assert( MPI_Start( &event_req ) == MPI_SUCCESS );
        assert( MPI_Start( &send_req ) == MPI_SUCCESS );
        mpirequests.push_back(event_req);
        requestinfo.push_back(request_t(request_t::event));
#endif // SGMPI_BUSYLOOP

        ongoing_recv.resize(num_ranks, 0);
        ongoing_send.resize(num_ranks, 0);

        assert( MPI_Barrier( MPI_COMM_WORLD ) == MPI_SUCCESS );

        Options::MPIInstrumentation::start();

        // unlock and allow wait_for_startup() to continue
        signal_lock.unlock();
    }

    void shut_down() {
        if (rank == 0) {
            // send termination messages
            std::vector<MPI_Request> request(num_ranks-1);
            for (int i = 1; i < num_ranks; ++i)
                assert( MPI_Isend(0, 0, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD, &request[i-1]) == MPI_SUCCESS );
            assert( MPI_Waitall(num_ranks-1, &request[0], MPI_STATUSES_IGNORE) == MPI_SUCCESS );
        }
        assert( MPI_Finalize() == MPI_SUCCESS );
    }

public:
    Communicator(ThreadingManager &tman_)
    : Options::Instrumentation(-1), tman(tman_),
       recv_requests(0), send_requests(0),
       signal_flag(true), ok_to_quit(false), terminate_flag(false)
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
    void recv(MPIHandle<Options> &handle, version_t required_version, int transfer_id, int recv_from) {
        bool need_to_signal;
        {
            SpinLockScoped scoped(recv_request_lock);
            need_to_signal = ongoing_recv[recv_from] < Options::limit_send_per_node;
            recv_requests[recv_from].push_back(request_t(&handle, required_version, transfer_id, recv_from));
        }
        if (need_to_signal)
            signal_event();
    }

    // called by main thread
    void terminate() {
        ok_to_quit = true;
        signal_event();
    }

    void barrier() {
        assert( MPI_Barrier( MPI_COMM_WORLD ) == MPI_SUCCESS );
    }

    // called by mpi thread
    void operator()(sg::SuperGlue<Options> &sg) {
        init();

        std::vector<int> indices;

        for (;;) {

            // lock and check the requests queues

#ifdef SGMPI_BUSYLOOP
            bool local_signal_flag = signal_flag;
            if (local_signal_flag) {
                SpinLockScoped siglock(signal_lock);
                signal_flag = false;
            }
            if (local_signal_flag)
#endif // SGMPI_BUSYLOOP
            {
                {
                    SpinLockScoped scoped(recv_request_lock);
                    for (int i = 0; i < num_ranks; ++i) {
                        if (ongoing_recv[i] >= Options::limit_recv_per_node)
                            continue;

                        if (recv_requests[i].empty())
                            continue;

                        requestinfo.push_back(recv_requests[i].front());
                        request_t &req(requestinfo[requestinfo.size()-1]);
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

#ifndef SGMPI_BUSYLOOP
                if (ok_to_quit && (mpirequests.size() == 1)) { // event request
                    shut_down();
                    return;
                }
#else
                if (ok_to_quit && mpirequests.empty()) {
                    shut_down();
                    return;
                }
#endif
            }

            if (mpirequests.empty())
                continue;

            indices.resize(mpirequests.size());

            int outcount = 0;
#ifndef SGMPI_BUSYLOOP
            assert( MPI_Waitsome(mpirequests.size(), &mpirequests[0], &outcount, &indices[0], MPI_STATUSES_IGNORE) == MPI_SUCCESS );
#else // SGMPI_BUSYLOOP
            assert( MPI_Testsome(mpirequests.size(), &mpirequests[0], &outcount, &indices[0], MPI_STATUSES_IGNORE) == MPI_SUCCESS );
#endif // SGMPI_BUSYLOOP
            Time::TimeUnit stop = Time::getTime();

            if (outcount > 0) {
#ifdef SGMPI_BUSYLOOP
                {
                    SpinLockScoped siglock(signal_lock);
                    signal_flag = true;
                }
#endif // SGMPI_BUSYLOOP

                for (int i = 0; i < outcount; ++i) {
                    const int idx = indices[i];
                    const request_t req(requestinfo[idx]);

#ifndef SGMPI_BUSYLOOP
                    if (req.type == request_t::event) {
                        {
                            SpinLockScoped siglock(signal_lock);
                            assert(signal_flag);
                            signal_flag = false;
                            assert( MPI_Start( &mpirequests[idx] ) == MPI_SUCCESS );
                        }
                        continue;
                    }
#endif // SGMPI_BUSYLOOP

                    mpirequests[idx] = MPI_REQUEST_NULL;

                    switch (req.type) {
                        case request_t::terminate:
                            terminate_flag = true;
                        continue;

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

                        case request_t::event: // already handled above, or not used
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
        }
    }
};

} // namespace sgmpi

#endif // SGMPI_COMMUNICATOR_HPP_INCLUDED
