"""Selection help follows focus; --help must not inspect hardware or launch."""
import io
import os
import pytest
from flexsds.cli import main as cli, wizard


class TerminalInput(io.StringIO):
    def isatty(self): return True


def test_footer_follows_arrow_selection_and_bool_context(monkeypatch, capsys):
    monkeypatch.setattr(wizard.sys,'stdin',TerminalInput())
    monkeypatch.setattr(wizard,'HAS_TERMIOS',True)
    keys=iter(['down','enter'])
    monkeypatch.setattr(wizard,'_get_key_stroke',lambda:next(keys))
    assert wizard.prompt_choice('Select action:',['Launch serving','Detect hardware']) == 'Detect hardware'
    output=capsys.readouterr().out
    assert 'Configure models, check compatibility' in output
    assert 'Show local GPU, CUDA availability' in output
    assert wizard.choice_description('Enable HardwareAdaptive engine?','Yes') != wizard.choice_description('Launch server now?','Yes')


def test_footer_stays_one_terminal_row(monkeypatch):
    monkeypatch.setattr(wizard.shutil,'get_terminal_size',lambda fallback:os.terminal_size((35,24)))
    line=wizard._menu_line('  Info: '+wizard.choice_description('Select action:','Launch serving'))
    assert len(line) <= 34 and '\n' not in line
    assert line.endswith('...')


@pytest.mark.parametrize('command',[['detect'],['profile'],['serve'],['stop'],['chat'],['monitor'],
                                     ['auth'],['auth','login'],['auth','status'],['auth','logout']])
def test_each_command_help_has_usage_and_example_without_runtime(command,monkeypatch,capsys):
    def forbidden(*args,**kwargs): pytest.fail('Help must not inspect hardware')
    monkeypatch.setattr(cli,'detect_hardware',forbidden)
    with pytest.raises(SystemExit) as result:
        cli.main([*command,'--help'])
    assert result.value.code == 0
    output=capsys.readouterr().out
    assert 'usage: flexsds' in output and 'Example: flexsds' in output
