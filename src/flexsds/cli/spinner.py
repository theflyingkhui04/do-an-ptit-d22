"""Small ASCII status spinner that leaves redirected output readable."""
from contextlib import contextmanager
import sys
import threading


@contextmanager
def status_spinner(message):
    stream = sys.stdout
    if not stream.isatty():
        stream.write(message + '...\n')
        stream.flush()
        yield
        return

    stop = threading.Event()
    width = len(message) + 2

    def draw(frame):
        stream.write('\r' + frame + ' ' + message)
        stream.flush()

    def animate():
        frames = ('-', '\\', '|', '/')
        index = 0
        while not stop.wait(0.1):
            draw(frames[index % len(frames)])
            index += 1

    draw('/')
    thread = threading.Thread(target=animate, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        stream.write('\r' + ' ' * width + '\r')
        stream.flush()
