---
title: 异步 I/O Asyncio
date: 2023-03-27
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- Other
---



[](https://docs.python.org/zh-cn/3/library/asyncio.html)

### eventloop

Eventloop 是 Asyncio 的核心功能之一。他能够用于储存多个 task，并指定 task 进行运行。同时，在某个 task A 进行等待时，eventloop 可以将执行权分发给下一个等候的 task。

### corroutine 

在执行 corroutine 过程中，可以将执行权交给其他协程。

```python
# corroutine function
async def main():
    await asyncio.sleep(1)
    
coro = main()  # # corroutine object
```

运行 corroutine function， 返回的结果是 corrouting object。corrouting funciton 内的内容不会被执行。

执行 corroutine function 内容，需要使用：

```python
asyncio.run(coro)  # 传入 corroutine object
```

或者使用 eventloop :

```python
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
```



### Future

Future 是协程的封装，提供了对协程任务的管理，回调，查看结果等。

```python
def __await_(self):
    if not self.done():
        self._asyncio_future_blocking = True
        yield self
    if not self.done():
        raise RuntimeError("await wasn't used with future")
    return self.result()
```

### task 和 await

task 是 future 的子类，实际开发中使用 task 更多。当 await 一个 task 时候，await 能够同时执行所有 `await task`。如下：

```python
async def main():
    task1 = asyncio.create_task(
         say_after(2, 'hello')
    )
    task2 = asyncio.create_task(
         say_after(1, 'world')
    )
    print(f"started at {time.strftime( '%X')}")
    result1 = await task1
    print(result1)
    result2 = await task2

    print(result2)
    print(f"finished at {time.strftime( '%X')}")
"""输出
started at 20:01:45
world
hello
hello-2
world-1
finished at 20:01:47
"""
```

以上代码的执行顺序较难分辨，建议使用 `.gather()` 等其他操作来处理：

```python
results = await asyncio.gather(
        say_after(2, 'hello'), say_after(1, 'world')
    )
# 输出结果保持顺序。
```

::: warning

不要直接 `await` 一个协程，否则他执行的结果会和同步执行的结果一样。

也不要 `await asyncio.create_task(...) `

:::

### wait

`asyncio.wait()`

在 asyncio 中，wait() 方法是一个非常重要的方法，它允许我们等待多个协程完成执行。wait() 方法将一组协程对象作为参数传递，并等待它们全部完成。一旦所有协程完成，wait() 方法将返回一个由 `(done, pending)` 两个集合组成的元组，其中 `done` 包含已完成的协程，而 `pending` 包含仍在等待执行的协程。

下面是一个简单的示例，说明如何使用 wait() 方法等待多个协程的完成：

```python
done, pending = await asyncio.wait(
        [say_after(2, 'hello'), say_after(1, 'world')]
    )
for result in done:
    print(result.result())
```

请注意，`wait()` 方法将一直等待，直到所有协程完成。如果您需要在一定时间内等待协程完成，可以使用 `asyncio.wait_for()` 方法。这个方法与 `wait()` 方法类似，但它会等待一段特定的时间，如果在这段时间内某些协程没有完成，它将引发一个 `asyncio.TimeoutError` 异常。

### 异步资源管理 async with

当我们在 Python 的异步编程中使用 `async with` 语句时，我们可以使用异步上下文管理器来设置和撤销上下文，这与普通的 `with` 语句类似。异步上下文管理器是一种对象，它定义了 `__aenter__()` 和 `__aexit__()` 方法，而不是 `__enter__()` 和 `__exit__()` 方法。

异步上下文管理器可以被用于异步的 `with` 语句中，以设置和撤销异步上下文。异步上下文管理器的作用是在异步代码块中为执行环境提供必要的支持，以确保上下文的正确性和完整性。

让我们来看一个例子来理解 `async with` 语句的用法：

```python
import asyncio

class MyAsyncContextManager:
    async def __aenter__(self):
        print("Entering asynchronous context")
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        print("Exiting asynchronous context")

async def my_coroutine():
    async with MyAsyncContextManager() as context:
        print("Inside the asynchronous context")

asyncio.run(my_coroutine())
```

正如您所看到的，`async with` 语句可以方便、安全地进入和离开异步上下文。`async with` 语句保证异步上下文管理器的 `__aexit__()` 方法始终会被调用，即使在 `async with` 代码块中出现异常。