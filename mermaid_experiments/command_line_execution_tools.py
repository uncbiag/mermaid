import subprocess
import os

def get_string_argument_from_list( arguments_as_list ):
  arguments_as_one_string = ''
  for el in arguments_as_list:
    arguments_as_one_string += el
    arguments_as_one_string += ' '
  return arguments_as_one_string

def execute_command( command, arguments_as_list, verbose=True ):
  # all commands that require substantial computation
  # will be executed through this function.
  # This will allow a uniform interface also for various clusters
  arguments_as_one_string = get_string_argument_from_list( arguments_as_list )

  if verbose:
    print('\nExecuting command:')
    print('     ' + command + ' ' + arguments_as_one_string )
  # create list for subprocess
  command_list = [ command ]
  for el in arguments_as_list:
    command_list.append( el )
  subprocess.call( command_list )

def execute_python_script_via_bash( python_script, arguments_as_list, pre_command=None, log_file=None ):

  str = get_string_argument_from_list(['python'] + [python_script] + arguments_as_list)
  if pre_command is not None and log_file is not None:
    bash_command = 'bash -c "{:s} {:s} > >(tee {:s}) 2>&1"'.format(pre_command,str,log_file)
  elif pre_command is not None and log_file is None:
    bash_command = 'bash -c "{:s} {:s}"'.format(pre_command, str)
  elif pre_command is None and log_file is not None:
    bash_command = 'bash -c "{:s} > >(tee {:s}) 2>&1"'.format(str, log_file)
  else:
    bash_command = 'bash -c "{:s}"'.format(str)

  print('\nExecuting command:')
  print('         ' + bash_command)
  os.system(bash_command)

def call_python_script( cmd ):
  cmd_full = 'python ' + cmd
  p = subprocess.Popen(cmd_full, stdout=subprocess.PIPE, shell=True)
  out, err = p.communicate()
  result = out.split('\n')
  for lin in result:
    if not lin.startswith('#'):
        print(lin)
