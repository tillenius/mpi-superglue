#ifndef SGMPI_INSTR_DEBUGMPI_HPP_INCLUDED
#define SGMPI_INSTR_DEBUGMPI_HPP_INCLUDED

#include <sstream>

template<typename Options>
struct TaskRunDebug {
    TaskRunDebug(int threadid) {}
    static void run_task_before(TaskBase<Options> *) {}
    static void run_task_after(TaskBase<Options> *task) {
        int rank = task->owner_rank;
        std::stringstream ss;
        ss << rank << ": run    '" << task << "' => ";

        for (size_t i = 0; i < task->getNumAccess(); ++i) {
            ss << " handle " 
               << task->get_access(i).handle->getGlobalId() 
               << " v" << task->get_access(i).requiredVersion
               << "-> v" << task->get_access(i).handle->version+1;
        }
        ss << std::endl;
        fprintf(stderr, "%s", ss.str().c_str());
    }
};

#endif // SGMPI_INSTR_DEBUGMPI_HPP_INCLUDED
