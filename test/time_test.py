import time

duration = 2.0
start_time = time.time()
run_time = 0.0
last_iteration = time.time()
all_iterations = []

while run_time < duration:
    current_time = time.time()
    run_time = current_time - start_time
    iteration_time = current_time - last_iteration
    all_iterations.append(iteration_time)
    last_iteration = time.perf_counter()
    print(last_iteration - time.perf_counter())
    # time.sleep(0.001)
    # if iteration_time > 0.001:
    #     # print(iteration_time)
    #     all_iterations.append(iteration_time)
    #     last_iteration = time.time()

print(all_iterations)
