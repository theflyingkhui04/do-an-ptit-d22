"""Small cross-platform terminal paste dialog: Ctrl+X saves, Esc cancels."""
from collections import deque
import os
import select
import shutil
import sys
import time


def _read_key():
    if os.name == 'nt':
        import msvcrt
        key = msvcrt.getwch()
        if key in ('\x00', '\xe0'):
            msvcrt.getwch()
            return ''
        return key
    raw = os.read(sys.stdin.fileno(), 1)
    if not raw:
        raise EOFError('Input closed before Ctrl+X.')
    if raw == b'\x1b':
        # Recognize bracketed-paste boundaries without treating them as Escape.
        sequence = raw.decode()
        while len(sequence) < 6 and select.select([sys.stdin.fileno()], [], [], 0.03)[0]:
            sequence += os.read(sys.stdin.fileno(), 1).decode(errors="replace")
        return '' if sequence in ('\x1b[200~', '\x1b[201~') else '\x1b'
    import codecs
    decoder = codecs.getincrementaldecoder(sys.stdin.encoding or 'utf-8')(errors='replace')
    while True:
        decoded = decoder.decode(raw)
        if decoded:
            return decoded
        raw = os.read(sys.stdin.fileno(), 1)
        if not raw:
            raise EOFError('Input closed before Ctrl+X.')


def _collect(read_key, draw):
    chars = []
    tail = deque(maxlen=4000)
    last_draw = 0
    while True:
        key = read_key()
        if key == '\x18':
            return ''.join(chars)
        if key == '\x1b':
            return None
        if key == '\x03':
            raise KeyboardInterrupt
        if key in ('\x08', '\x7f'):
            if chars:
                chars.pop()
                tail = deque(chars[-4000:], maxlen=4000)
        elif key:
            if key == '\r':
                key = '\n'
            chars.append(key)
            tail.append(key)
        if len(chars) > 16 * 1024 * 1024:
            raise ValueError('Profile exceeds 16 MiB.')
        now = time.monotonic()
        if now-last_draw >= 0.1:
            draw(''.join(tail), len(chars))
            last_draw = now


def paste_profile():
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return sys.stdin.read().split('\x18', 1)[0]
    original = None
    def draw(tail, count):
        width, height = shutil.get_terminal_size((100, 24))
        lines = tail.replace('\r', '').split('\n')[-max(1,height-5):]
        header = 'Import hardware profile | Paste JSON | Ctrl+X: save | Esc: cancel'
        body = '\n'.join(line[-max(1,width-1):] for line in lines)
        sys.stdout.write('\x1b[H\x1b[2J'+header[:max(1,width-1)]+'\n'+body+f'\nCharacters: {count}')
        sys.stdout.flush()
    try:
        if os.name != 'nt':
            import termios
            import tty
            original = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        sys.stdout.write('\x1b[?1049h\x1b[?2004h')
        draw('', 0)
        return _collect(_read_key, draw)
    finally:
        sys.stdout.write('\x1b[?2004l\x1b[?1049l')
        sys.stdout.flush()
        if original is not None:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original)
