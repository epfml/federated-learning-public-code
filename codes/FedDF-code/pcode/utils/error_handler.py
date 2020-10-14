# -*- coding: utf-8 -*-
import sys


def global_except_hook(exctype, value, traceback):
    import sys

    try:
        import mpi4py.MPI

        sys.stderr.write("\n*****************************************************\n")
        sys.stderr.write(
            "Uncaught exception was detected on rank {}. \n".format(
                mpi4py.MPI.COMM_WORLD.Get_rank()
            )
        )
        from traceback import print_exception

        print_exception(exctype, value, traceback)
        sys.stderr.write("*****************************************************\n\n\n")
        sys.stderr.write("\n")
        sys.stderr.write("Calling MPI_Abort() to shut down MPI processes...\n")
        sys.stderr.flush()
    finally:
        try:
            import mpi4py.MPI

            mpi4py.MPI.COMM_WORLD.Abort(1)
        except Exception as e:
            sys.stderr.write("*****************************************************\n")
            sys.stderr.write("Sorry, we failed to stop MPI, this process will hang.\n")
            sys.stderr.write("*****************************************************\n")
            sys.stderr.flush()
            raise e


def error_abort():
    try:
        import mpi4py.MPI

        mpi4py.MPI.COMM_WORLD.Abort(1)
    except Exception as e:
        sys.stderr.write("*****************************************************\n")
        sys.stderr.write("Sorry, we failed to stop MPI, this process will hang.\n")
        sys.stderr.write("*****************************************************\n")
        sys.stderr.flush()
        raise e


def abort():
    error_abort()
