"""Compact subprocess progress; Ctrl+O replays and streams complete saved logs."""
from __future__ import annotations
import os
from pathlib import Path
import re
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from flexsds.cli.output import ascii_text

LOG_DIR = Path.home() / '.flexsds' / 'logs'
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]|\x1b\].*?(?:\x07|\x1b\\)')


class ProcessProgress:
    def __init__(self, kind: str, total_cells: int = 0):
        self.kind = kind
        self.total_cells = total_cells
        self.cells = set()
        self.percent = 0
        self.message = 'Preparing Modal ' + kind
        self.app_url = None
        self.details = False
        self.closed = False
        self.interactive = sys.stdout.isatty() and sys.stdin.isatty()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.done = threading.Event()
        self.listener = None
        self.terminal = None
        self.started = time.monotonic()
        self.last_plain = None
        self.last_drawn = None
        self.last_error = None
        self.read_error = None
        LOG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.log = tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', prefix=f'modal-{kind}-',
                                             suffix='.log', dir=LOG_DIR, delete=False)
        self.log_path = Path(self.log.name)

    def _write(self, text):
        try:
            sys.stdout.write(text)
        except UnicodeEncodeError:
            sys.stdout.write(ascii_text(text))

    def _draw(self):
        if self.closed or self.details:
            return
        elapsed = int(time.monotonic()-self.started)
        hint = 'Ctrl+O for details' if self.interactive else f'log: {self.log_path}'
        line = f'[{self.percent:3d}%] {self.message} ({elapsed}s; {hint})'
        if self.interactive:
            width = max(1, shutil.get_terminal_size((100,24)).columns-1)
            # Preserve the shortcut even on a narrow terminal.
            suffix = ' (Ctrl+O for details)'
            available = max(0, width-len(suffix))
            body = f'[{self.percent}%] {self.message} ({elapsed}s)'
            if len(body) > available:
                body = body[:max(0,available-3)] + '...'
            visible = (body+suffix)[:width]
            if visible == self.last_drawn:
                return
            self.last_drawn = visible
            self._write('\r\x1b[K' + visible)
        elif (self.percent,self.message) != self.last_plain:
            self._write(line+'\n')
            self.last_plain = (self.percent,self.message)
        sys.stdout.flush()

    def notice(self, text):
        with self.lock:
            if self.interactive and not self.details:
                self._write('\r\x1b[K')
                self.last_drawn = None
            self._write(text+'\n')
            self._draw()
            sys.stdout.flush()

    def update(self, message, percent=None):
        with self.lock:
            self.message = message
            if percent is not None:
                self.percent = max(self.percent, min(100,percent))
            self._draw()

    def feed(self, line):
        clean = ANSI.sub('',line).rstrip('\r\n')
        with self.lock:
            if self.closed:
                return
            self.log.write(line)
            self.log.flush()
            if self.details:
                self._write(clean+'\n')
                sys.stdout.flush()
            app = re.search(r'https://modal.com/apps/[^\s\x1b]+',clean)
            if app and app.group() != self.app_url:
                self.app_url = app.group()
                self.notice('View app: '+self.app_url)
            error = re.search(r'(?:RuntimeError|ValueError|ImportError|ModuleNotFoundError|OSError): (.+)',clean)
            if error and self.last_error is None:
                self.last_error = error.group(1)
            cell = re.search(r'Calibration measured K=(\d+), B=(\d+)',clean)
            if cell and self.total_cells:
                self.cells.add(tuple(map(int,cell.groups())))
                self.update(f'Measured {len(self.cells)}/{self.total_cells} GPU cells',
                            10+int(85*len(self.cells)/self.total_cells))
            elif 'Building image' in clean:
                self.update('Preparing CUDA image',5)
            elif 'Created Function' in clean:
                self.update('Waiting for GPU container',10)
            elif 'Starting to load model' in clean:
                self.update('Loading target and EAGLE weights',30 if self.kind=='serving' else 10)
            elif 'Model loading took' in clean:
                self.update('Preparing KV cache',50 if self.kind=='serving' else 10)
            elif 'JIT kernel warmup' in clean:
                match = re.search(r'(\d+)%',clean)
                self.update('GPU kernel warmup'+(f' ({match[1]}%)' if match else ''),
                            70 if self.kind=='serving' else 10)
            elif 'Application startup complete' in clean:
                self.update('Checking endpoint readiness',85)

    def toggle_details(self):
        with self.lock:
            if self.closed:
                return
            self.details = not self.details
            if self.details:
                self._write('\r\x1b[K\n--- Modal logs (Ctrl+O to collapse) ---\n')
                self.log.flush()
                with self.log_path.open(encoding='utf-8') as saved:
                    for line in saved:
                        self._write(ANSI.sub('',line))
                sys.stdout.flush()
            else:
                # Full logs stay in scrollback and on disk; reset the visible view.
                self._write('\x1b[2J\x1b[H')
                self.last_drawn = None
                if self.app_url:
                    self._write('View app: '+self.app_url+'\n')
                self._draw()

    def start(self):
        if not self.interactive:
            self.notice('Logs: '+str(self.log_path))
            return
        try:
            if os.name != 'nt':
                import termios
                import tty
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                self.terminal = (fd,old)
                tty.setcbreak(fd)
                attrs = termios.tcgetattr(fd)
                attrs[3] &= ~termios.IEXTEN  # Ctrl+O must not be eaten as VDISCARD.
                termios.tcsetattr(fd,termios.TCSANOW,attrs)
            def listen():
                while not self.stop.wait(0.1):
                    try:
                        key = ''
                        if os.name == 'nt':
                            import msvcrt
                            if msvcrt.kbhit(): key = msvcrt.getwch()
                        elif select.select([sys.stdin.fileno()],[],[],0)[0]:
                            key = os.read(sys.stdin.fileno(),1).decode(errors='replace')
                        if key == '\x0f': self.toggle_details()
                        with self.lock: self._draw()
                    except (OSError,ValueError):
                        return
            self.listener = threading.Thread(target=listen,daemon=True)
            self.listener.start()
            self.notice('Logs: '+str(self.log_path))
        except (OSError,ValueError,ImportError):
            self.close()
            raise

    def capture(self, proc, on_line=None, close_on_exit=False):
        def read():
            try:
                for line in proc.stdout:
                    self.feed(line)
                    if on_line: on_line(ANSI.sub('',line))
            except Exception as exc:
                self.read_error = exc
                if proc.poll() is None:
                    proc.terminate()
            finally:
                self.done.set()
                if close_on_exit:
                    code = proc.wait()
                    self.update('Serving stopped' if code in (0,130,-2,-15) else
                                'Serving failed: '+(self.last_error or f'exit {code}'))
                    self.close()
        thread = threading.Thread(target=read,daemon=True)
        thread.start()
        return thread

    def close(self):
        self.stop.set()
        if self.listener and self.listener is not threading.current_thread():
            self.listener.join(timeout=0.3)
        with self.lock:
            if self.closed:
                return
            if self.terminal:
                import termios
                fd,old = self.terminal
                try:
                    termios.tcsetattr(fd,termios.TCSADRAIN,old)
                except OSError:
                    pass  # A disconnected terminal cannot be restored.
                self.terminal = None
            self.closed = True
            self.log.close()
            if self.interactive:
                self._write('\n')
                sys.stdout.flush()


def run_logged_process(command, *, env, timeout, total_cells):
    """Run a calibration CLI with bounded lifetime and complete private logs."""
    progress = ProcessProgress('calibration',total_cells)
    proc = None
    reader = None
    try:
        proc = subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1)
        progress.start()
        reader = progress.capture(proc)
        code = proc.wait(timeout=timeout)
        reader.join(timeout=5)
        if progress.read_error:
            raise RuntimeError(f'Cannot read Modal logs: {progress.read_error}') from progress.read_error
        if code:
            raise subprocess.CalledProcessError(code,command)
        progress.update('Calibration job completed',100)
        return subprocess.CompletedProcess(command,code)
    except BaseException as exc:
        if proc and proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
        if reader: reader.join(timeout=5)
        progress.update('Calibration stopped' if isinstance(exc,KeyboardInterrupt) else 'Calibration failed'+(': '+progress.last_error if progress.last_error else ''))
        progress.notice('Full details: '+str(progress.log_path))
        raise
    finally:
        progress.close()
