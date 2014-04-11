#ifndef SGMPI_MPITHREADINGMANAGER_HPP_INCLUDED
#define SGMPI_MPITHREADINGMANAGER_HPP_INCLUDED

namespace sg {
template <typename Options> class Log;
template <typename Options> class TaskExecutor;
class ThreadUtil;
}

namespace sgmpi {

template <typename Options> class MPISuperGlue;
template <typename Options> class Communicator;

template<typename Options>
struct MPIMain { virtual void run(sgmpi::MPISuperGlue<Options> &) = 0; };

template <typename Options>
class ThreadingManagerMPI {
    typedef typename Options::ReadyListType TaskQueue;
    typedef typename Options::ThreadingManagerType ThreadingManager;

private:

    // ===========================================================================
    // MainThread
    // ===========================================================================
    struct MainThread : public sg::Thread {
        ThreadingManagerMPI<Options> &tman;

        MainThread(sgmpi::ThreadingManagerMPI<Options> &tman_) : tman(tman_) {}

        void run() {
            Options::ThreadAffinity::pin_main_thread();

            tman.threads[0] = new sg::TaskExecutor<Options>(0, tman);
            tman.task_queues[0] = &tman.threads[0]->get_task_queue();

            {
                MPISuperGlue<Options> mpisg(tman);
                tman.mpisg = &mpisg;
                tman.main_thread_initialized.unlock();

                // wait for mpicomm to initialize
                tman.mpicomm.wait_for_startup();

                // run main
                tman.main_function(mpisg);

                // MPISuperGlue and sg::SuperGlue destructed here
            }

            tman.mpicomm.terminate();

            // wait for mpicomm to shut down
            tman.lock_mpicomm_running.lock();
            tman.lock_mpicomm_running.unlock();
        }
    };

    // ===========================================================================
    // WorkerThread: Thread to run worker
    // ===========================================================================
    class WorkerThread : public sg::Thread {
    private:
        const int id;
        ThreadingManagerMPI &tman;

    public:
        WorkerThread(int id_, ThreadingManagerMPI &tman_)
        : id(id_), tman(tman_) {}

        void run() {
            Options::ThreadAffinity::pin_workerthread(id);
            // allocate Worker on thread
            sg::TaskExecutor<Options> *te = new sg::TaskExecutor<Options>(id, tman);

            tman.threads[id] = te;
            tman.task_queues[id] = &te->get_task_queue();
            sg::Atomic::memory_fence_producer();
            sg::Atomic::increase(&tman.start_counter);

            // wait to start
            tman.lock_workers_initialized.lock();
            tman.lock_workers_initialized.unlock();

            // wait for mpicomm to initialize
            tman.mpicomm.wait_for_startup();

            te->work_loop();
        }
    };

    sg::SpinLock main_thread_initialized;
    sg::SpinLock lock_workers_initialized;
    sg::SpinLock lock_mpicomm_running;
    MPISuperGlue<Options> *mpisg;
    size_t start_counter;
    char padding1[Options::CACHE_LINE_SIZE];
    int rank;
    int num_ranks;
    int num_cpus;
    void (*main_function)(sgmpi::MPISuperGlue<Options> &);
    std::vector<WorkerThread *> workerthreads;

public:
    Communicator<Options> mpicomm;
    sg::BarrierProtocol<Options> barrier_protocol;
    sg::TaskExecutor<Options> **threads;
    TaskQueue **task_queues;
    char padding2[Options::CACHE_LINE_SIZE];
    MainThread *main_thread;

private:
    static bool workers_start_paused(typename Options::Disable) { return false; }
    static bool workers_start_paused(typename Options::Enable) { return true; }
    static bool workers_start_paused() { return workers_start_paused(typename Options::PauseExecution()); }

    int get_start_signal(typename Options::Disable) { return num_cpus; }
    int get_start_signal(typename Options::Enable) { return 0; }

    int decide_num_cpus_inner(int requested) {
        assert(requested == -1 || requested > 0);
        const char *var = getenv("OMP_NUM_THREADS");
        if (var != NULL) {
            const int OMP_NUM_THREADS(atoi(var));
            assert(OMP_NUM_THREADS >= 0);
            if (OMP_NUM_THREADS != 0)
                return OMP_NUM_THREADS;
        }
        if (requested == -1 || requested == 0)
            return sg::ThreadUtil::get_num_cpus();
        return requested;
    }

    int decide_num_cpus(int requested) {
        const int num_cpus = decide_num_cpus_inner(requested)-1;
        assert(num_cpus > 0);
        return num_cpus;
    }

public:
    ThreadingManagerMPI(int requested_num_cpus = -1)
    : barrier_protocol(*static_cast<ThreadingManager*>(this))
    {
        // required by interface, but not allowed.
        assert(false);
    }

    ThreadingManagerMPI(void (*mainfn)(sgmpi::MPISuperGlue<Options> &))
    : start_counter(0),
      num_cpus(decide_num_cpus(-1)),
      main_function(mainfn),
      mpicomm(*static_cast<ThreadingManager *>(this)),
      barrier_protocol(*static_cast<ThreadingManager *>(this))
    {

    #ifdef SGMPI_BUSYLOOP
        assert( MPI_Init(NULL, NULL) == MPI_SUCCESS );
    #else
        int provided;
        assert( MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided) == MPI_SUCCESS );
        assert( provided == MPI_THREAD_MULTIPLE );
    #endif

        assert( MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS );
        assert( MPI_Comm_size(MPI_COMM_WORLD, &num_ranks) == MPI_SUCCESS );

        //const int num_cpus = Options::ThreadBackend::decide_num_cpus(-1);
        Options::ThreadAffinity::init();
        Options::ThreadAffinity::init(rank, num_cpus+1);
        Options::ThreadAffinity::pin_mpi_thread();

        main_thread_initialized.lock();
        lock_workers_initialized.lock();
        lock_mpicomm_running.lock();
        threads = new sg::TaskExecutor<Options> *[num_cpus];
        task_queues = new TaskQueue*[num_cpus];

        // start main thread.
        // main thread unlocks main_thread_initialized when inited
        // but then waits for mpicomm.wait_for_startup()
        main_thread = new MainThread(*this);
        main_thread->start();

        // start worker threads.
        // workers will increase start_counter, then wait for lock_workers_initialized
        const int num_workers(num_cpus-1);
        workerthreads.resize(num_workers);
        for (int i = 0; i < num_workers; ++i) {
            workerthreads[i] = new WorkerThread(i+1, *this);
            workerthreads[i]->start();
        }

        // wait for all workers to register
        while (start_counter != num_cpus-1)
            sg::Atomic::rep_nop();
        sg::Atomic::memory_fence_consumer();

        // wait for main thread to be initialized (especially mpisg->sg)
        main_thread_initialized.lock();
        main_thread_initialized.unlock();

        // release workers
        if (!workers_start_paused())
            lock_workers_initialized.unlock();

        // init mpicomm (using mpisg->sg)
        mpicomm.operator()(mpisg->sg);
        lock_mpicomm_running.unlock();
    }

    ~ThreadingManagerMPI() {
        // shut down main thread
        main_thread->join();
        delete main_thread;

        // shut down workers
        for (int i = 1; i < get_num_cpus(); ++i)
            threads[i]->terminate();

        const int num_workers(num_cpus-1);
        for (int i = 0; i < num_workers; ++i)
            workerthreads[i]->join();

        // cleanup memory
        for (int i = 0; i < num_workers; ++i)
            delete workerthreads[i];
        for (int i = 0; i < num_cpus; ++i)
            delete threads[i];

        delete [] threads;
        delete [] task_queues;
    }

    void init() {
        // All initialization done in constructor
    }

    void stop() {
        start_executing(); // make sure threads have been started, or we will wait forever in barrier
        barrier_protocol.barrier(*threads[0]);
    }

    void start_executing() {
        if (workers_start_paused() && lock_workers_initialized.is_locked())
            lock_workers_initialized.unlock();
    }

    TaskQueue **get_task_queues() const { return const_cast<TaskQueue**>(&task_queues[0]); }
    sg::TaskExecutor<Options> *get_worker(int i) { return threads[i]; }
    int get_num_cpus() { return num_cpus; }

    // specific for mpi

    int get_rank() { return rank; }
    int get_num_ranks() { return num_ranks; }
};

} // namespace sgmpi

#endif // SGMPI_MPITHREADINGMANAGER_HPP_INCLUDED
