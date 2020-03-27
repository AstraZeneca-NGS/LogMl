
import inspect
import logging
import os
import sys

from IPython import display


def _create_logml_logger():
    ''' Create and set-up a logml_logger for this module '''
    logml_logger = logging.getLogger("LogMl")
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logml_logger.addHandler(handler)
    logml_logger.setLevel(logging.WARNING)
    return logml_logger


logml_logger = _create_logml_logger()


class Tee:
    '''
    Tee functionality in a class
    Reference: http://code.activestate.com/recipes/580767-unix-tee-like-functionality-via-a-python-class/
    '''
    def __init__(self, tee_filename, is_stderr=False):
        self.is_stderr = is_stderr
        try:
            self.tee_file = open(tee_filename, "w")
        except IOError as ioe:
            error_exit("Caught IOError: {}".format(repr(ioe)))
        except Exception as e:
            error_exit("Caught Exception: {}".format(repr(e)))
        # Save and replace
        if is_stderr:
            self.std = sys.stderr
            sys.stderr = self
        else:
            self.std = sys.stdout
            sys.stdout = self

    def close(self):
        try:
            self.restore()
            self.tee_file.close()
        except IOError as ioe:
            error_exit("Caught IOError: {}".format(repr(ioe)))
        except Exception as e:
            error_exit("Caught Exception: {}".format(repr(e)))

    def __del__(self):
        ''' Restore stdout/stderr '''
        self.restore()

    def restore(self):
        ''' Restore stdout/stderr '''
        if self.std is not None:
            if self.is_stderr:
                sys.stderr = self.std
            else:
                sys.stdout = self.std
            self.std = None

    def write(self, s):
        self.std.write(s)
        self.tee_file.write(s)

    def writeln(self, s):
        self.write(s + '\n')

    def flush(self):
        self.tee_file.flush()
        self.std.flush()


class MlLogMessages:
    ''' ML Log: Log events and redirect STDOUT/STDERR to files '''
    def __init__(self):
        pass

    def _context(self, frame_number=2):
        ''' Return a string representing the caller function context (used for log messages) '''
        frame = inspect.stack()[frame_number]
        fbase = os.path.basename(frame.filename)
        return f"{type(self).__name__}.{frame.function} ({fbase}:{frame.lineno})"

    def _debug(self, msg):
        ''' Show a debug message '''
        logml_logger.debug(f"{self._context()}: {msg}")
        sys.stderr.flush()

    def _error(self, msg):
        ''' Show an error message '''
        logml_logger.error(f"{self._context()}: {msg}")
        sys.stderr.flush()

    def _fatal_error(self, msg):
        ''' Show an error message and exit '''
        logml_logger.error(f"{self._context()}: {msg}")
        sys.stderr.flush()
        if self.config.exit_on_fatal_error:
            sys.exit(1)

    def _info(self, msg):
        ''' Show an INFO message '''
        logml_logger.info(f"{msg}")
        sys.stderr.flush()

    def set_log_level(self, level):
        ''' A shortcut to setting log level '''
        logml_logger.setLevel(level)

    def _warning(self, msg):
        ''' Show a warning message '''
        logml_logger.warning(f"{self._context()}: {msg}")
        sys.stderr.flush()


class MlLog(MlLogMessages):
    '''
    ML Log: Log events and redirect STDOUT/STDERR to files
    '''
    def __init__(self, config=None, config_section=None):
        self.config = config
        self.config_section = config_section
        self.parameters = dict()
        self.enable = False
        self.file_stdout = None
        self.file_stderr = None
        self.tee_stdout = None
        self.tee_stderr = None

    def _config_sanity_check(self):
        '''
        Check parameters from config.
        Return True on success, False if there are errors
        '''
        pass

    def _html(self, msg):
        display.HTML(msg)

    def _set_from_config(self):
        '''
        Set object variables from 'self.config'
        '''
        if ('config' not in self.__dict__) or (self.config is None):
            self._debug("No config object found, skipping")
            return False
        if not self.config_section:
            self._debug(f"No config_section, skipping")
            return False
        if self.config_section not in self.config.parameters:
            self._debug(f"Config section '{self.config_section}' not in config.parameters, skipping")
            return False
        self._debug(f"Setting parameters from 'self.config', section '{self.config_section}'")
        ret = self._set_from_dict(self.config.get_parameters(self.config_section))
        return self._config_sanity_check() and ret

    def _set_from_dict(self, parameters):
        ''' Set object parameters from 'parameters' dictionary '''
        ok = False
        if parameters is None:
            return False
        self.parameters = parameters
        for field_name in parameters.keys():
            if field_name in self.__dict__:
                val = parameters.get(field_name)
                if val is not None:
                    self._debug(f"Setting field '{field_name}' to '{val}'")
                    self.__dict__[field_name] = val
                    ok = True
        return ok

    def _subtitle(self, msg):
        display.HTML(f"<h3>{msg}</h3>")

    def tee(self, close=False):
        ''' Copy STDOUT / STDERR to file '''
        if close:
            # Close tees
            if self.tee_stdout:
                self.tee_stdout.close()
                self.tee_stdout = None
            if self.tee_stderr:
                self.tee_stderr.close()
                self.tee_stderr = None
        else:
            # Open tees
            if self.file_stdout:
                self.tee_stdout = Tee(self.file_stdout)
            if self.file_stderr:
                self.tee_stderr = Tee(self.file_stderr, True)

    def _title(self, msg):
        display.HTML(f"<h1>{msg}</h1>")
