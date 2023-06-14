# distutils: sources = ../external/c-algorithms-master/src/queue.c
# distutils: include_dirs = ../external/c-algorithms-master/src/

cimport cqueue


cdef class Queue:
    cdef cqueue.Queue* _c_queue

    def __cinit__(self):
        self._c_queue = cqueue.queue_new()
        if self._c_queue is NULL:
            raise MemoryError()


    def __dealloc__(self):
        if self._c_queue is not NULL:
            cqueue.queue_free(self._c_queue)


    def __bool__(self):
        return not cqueue.queue_is_empty(self._c_queue)


    cdef append(self, int value):
    if not cqueue.queue_push_tail(self._c_queue, <void*>value):
        raise MemoryError()


    cdef extend(self, int* values, size_t count):
    """Append all ints to the queue.
    """
    cdef int value
    for value in values[:count]:  # Slicing pointer to limit the iteration boundaries.
        self.append(value)


   cdef int peek(self):
        if cqueue.queue_is_empty(self._c_queue):
            raise IndexError("Queue is empty")
        return <Py_ssize_t>cqueue.queue_peek_head(self._c_queue)

   cdef int pop(self):
        if cqueue.queue_is_empty(self._c_queue):
                raise IndexError("Queue is empty")
        return <Py_ssize_t>cqueue.queue_pop_head(self._c_queue)
   