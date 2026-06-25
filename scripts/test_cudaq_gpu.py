import cudaq

# Verify CUDA-Q GPU target works
cudaq.set_target('nvidia')
print('CUDA-Q target set to nvidia')

@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])

result = cudaq.sample(bell, shots_count=100)
print('Bell state sample:', dict(result))
