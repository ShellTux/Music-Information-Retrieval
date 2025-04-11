from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def task(n):
    time.sleep(1)
    return n * n

with ProcessPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(task, i): i for i in range(10)}
    for future in as_completed(futures):
        n = futures[future]
        try:
            result = future.result()
        except Exception as e:
            print(f'Task {n} generated an exception: {e}')
        else:
            print(f'Task {n} returned {result}')
