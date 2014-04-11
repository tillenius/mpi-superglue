#ifndef SGMPI_MPIINSTR_HPP_INCLUDED
#define SGMPI_MPIINSTR_HPP_INCLUDED

//
// WARNING: Requires Log<Options>::register_thread() to be called from some other
// instrumentation, or it will crash.
//

namespace sgmpi {

template<typename Options>
class MPITrace {
public:
    struct MPIInstrData {
        Time::TimeUnit mpiinstr_start;
    };

    static void start() {
        double mpi_time = MPI_Wtime();
        Time::TimeUnit sg_time = Time::getTime();
        char event[80];
        sprintf(event, "MPI_Wtime %f", mpi_time);
        Log<Options>::log(event, sg_time, sg_time);
    }
    template<typename Req>
    static void start(Req &req) {
        req.mpiinstr_start = Time::getTime();
    }
    template<typename Req>
    static void recv(Req &req, Time::TimeUnit stop) {
        char msg[80];
        sprintf(msg, "recv %d from %d", req.handle->get_global_id(), req.rank);
        Log<Options>::log(msg, /*req.start*/ stop, stop);
    }
    template<typename Req>
    static void send(Req &req, Time::TimeUnit stop) {
        char msg[80];
        sprintf(msg, "send %d to %d", req.handle->get_global_id(), req.rank);
        Log<Options>::log(msg, req.mpiinstr_start, stop);
    }
};

} // namespace sgmpi

#endif // SGMPI_MPIINSTR_HPP_INCLUDED