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
public:
    enum { MPI_THREAD_ID = 0, MAIN_THREAD_ID = 1, WORKER_THREAD_ID_BASE = 2 };

private:

    // ===========================================================================
    // MainThread
    // ===========================================================================
    struct MainThread : public sg::Thread {
        ThreadingManagerMPI<Options> &tman;
        sg::TaskExecutor<Options> *task_executor;

        MainThread(sgmpi::ThreadingManagerMPI<Options> &tman_) : tman(tman_) {}

        void run() {
            // MAIN THREAD [id 1]
            Options::ThreadAffinity::pin_main_thread();
            task_executor = new sg::TaskExecutor<Options>(MAIN_THREAD_ID, tman);

            tman.task_executors[MAIN_THREAD_ID] = task_executor;
            tman.task_queues[MAIN_THREAD_ID] = &tman.task_executors[MAIN_THREAD_ID]->get_task_queue();

            {
                MPISuperGlue<Options> mpisg(tman);
                tman.mpisg = &mpisg;
                tman.main_thread_initialized.unlock();

                // wait for mpicomm to initialize
                tman.mpicomm.wait_for_startup();

                // run main
                tman.main_function(mpisg);

                // MPISuperGlue (and sg::SuperGlue) destructed here => implicit barrier.
            }

            // wait for mpicomm to shut down
            tman.lock_mpicomm_running.lock();
            tman.lock_mpicomm_running.unlock();
        }
        ~MainThread() {
            delete task_executor;
        }
    };

    // ===========================================================================
    // WorkerThread: Thread to run worker
    // ===========================================================================
    class WorkerThread : public sg::Thread {
    public:
        const int id;
    private:
        ThreadingManagerMPI &tman;
        sg::TaskExecutor<Options> *task_executor;

    public:
        WorkerThread(int id_, ThreadingManagerMPI &tman_)
        : id(id_), tman(tman_) {}

        void run() {
            // WORKER THREAD [id 2 -- id N]
            Options::ThreadAffinity::pin_workerthread(id);
            // allocate Worker on thread
            task_executor = new sg::TaskExecutor<Options>(id, tman);

            tman.task_executors[id] = task_executor;
            tman.task_queues[id] = &task_executor->get_task_queue();
            sg::Atomic::memory_fence_producer();
            sg::Atomic::increase(&tman.start_counter);

            // wait to start
            tman.lock_workers_initialized.lock();
            tman.lock_workers_initialized.unlock();

            // wait for mpicomm to initialize
            tman.mpicomm.wait_for_startup();

            task_executor->work_loop();
        }
        ~WorkerThread() {
            delete task_executor;
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
    std::vector<sg::TaskExecutor<Options> *> task_executors;
    std::vector<TaskQueue *> task_queues;
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
        std::string var = sg_getenv("OMP_NUM_THREADS");
        if (!var.empty()) {
            const int OMP_NUM_THREADS(atoi(var.c_str()));
            assert(OMP_NUM_THREADS >= 0);
            if (OMP_NUM_THREADS != 0)
                return OMP_NUM_THREADS;
        }
        if (requested == -1 || requested == 0)
            return sg::ThreadUtil::get_num_cpus();
        return requested;
    }

    int decide_num_cpus(int requested) {
        const int num_cpus = decide_num_cpus_inner(requested);
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
        // MPI THREAD [id 0]

        assert( MPI_Init(NULL, NULL) == MPI_SUCCESS );
        assert( MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS );
        assert( MPI_Comm_size(MPI_COMM_WORLD, &num_ranks) == MPI_SUCCESS );

        Options::ThreadAffinity::init();
        Options::ThreadAffinity::init(rank, num_cpus);
        Options::ThreadAffinity::pin_mpi_thread();

        main_thread_initialized.lock();
        lock_workers_initialized.lock();
        lock_mpicomm_running.lock();

        task_executors.resize(num_cpus);
        task_queues.resize(num_cpus);

        task_executors[MPI_THREAD_ID] = &mpicomm;
        task_queues[MPI_THREAD_ID] = &mpicomm.get_task_queue();

        // start main thread.
        // main thread unlocks main_thread_initialized when inited
        // but then waits for mpicomm.wait_for_startup()
        main_thread = new MainThread(*this);
        main_thread->start();

        // start worker threads.
        // workers will increase start_counter, then wait for lock_workers_initialized
        const size_t num_workers(num_cpus - WORKER_THREAD_ID_BASE);

        workerthreads.resize(num_workers);
        for (size_t i = 0; i < num_workers; ++i) {
            workerthreads[i] = new WorkerThread(i + WORKER_THREAD_ID_BASE, *this);
            workerthreads[i]->start();
        }

        // wait for all workers to register
        while (start_counter != num_workers)
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
        // MPI THREAD [id 0]

        // wait for main thread to shut down
        main_thread->join();

        // shut down workers
        for (size_t i = 0; i < workerthreads.size(); ++i)
            task_executors[workerthreads[i]->id]->terminate();

        // wait for workers
        for (size_t i = 0; i < workerthreads.size(); ++i)
            workerthreads[i]->join();

        // cleanup memory
        delete main_thread;
        for (size_t i = 0; i < workerthreads.size(); ++i)
            delete workerthreads[i];
    }

    void init() {
        // All initialization done in constructor
    }

    void stop() {
        // called from main thread, when superglue is destructed.
        // no need to do a superglue barrier here, since that is
        // included in shutting down mpi-superglue.
    }

    void start_executing() {
        if (workers_start_paused() && lock_workers_initialized.is_locked())
            lock_workers_initialized.unlock();
    }

    TaskQueue **get_task_queues() const { return const_cast<TaskQueue**>(&task_queues[0]); }
    sg::TaskExecutor<Options> *get_worker(int i) { return task_executors[i]; }
    int get_num_cpus() { return num_cpus; }

    // specific for mpi

    int get_rank() { return rank; }
    int get_num_ranks() { return num_ranks; }
};

} // namespace sgmpi

#endif // SGMPI_MPITHREADINGMANAGER_HPP_INCLUDED
