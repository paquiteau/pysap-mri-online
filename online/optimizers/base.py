import time

def online_algorithm(opt, kspace_generator, estimate_call_period=None,
                     nb_run=1, verbose=0):
    opt.idx = 0
    cost_op = opt._cost_func
    # Perform the first reconstruction
    if verbose > 0:
        print("Starting optimization...")

    start = time.perf_counter()
    estimates = list()
    for run in range(nb_run):
        estimates += kspace_generator.opt_iterate(opt, estimate_call_period=estimate_call_period)

    end = time.perf_counter()
    # Goodbye
    if verbose > 0:
        if hasattr(cost_op, "cost"):
            print(" - final iteration number: ", cost_op._iteration)
            print(" - final cost value: ", cost_op.cost)
        print(" - converged: ", opt.converge)
        print("Done.")
        print("Execution time: ", end - start, " seconds")
        print("-" * 40)
    # Get the final solution
    observer_kwargs = opt.get_notify_observers_kwargs()

    ret_dict = dict()
    ret_dict['x_final'] = observer_kwargs['x_new']
    ret_dict['metrics'] = opt.metrics
    if hasattr(opt, '_y_new'):
        ret_dict['y_final'] = observer_kwargs['y_new']
    if hasattr(cost_op, "cost"):
        ret_dict['costs'] =  cost_op._cost_list
    if estimates:
        ret_dict['x_estimates'] = estimates
    return ret_dict
