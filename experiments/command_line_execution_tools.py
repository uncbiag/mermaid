import subprocess

def getStringArgumentFromList( argumentsAsList ):
  argumentsAsOneString = ''
  for el in argumentsAsList:
    argumentsAsOneString += el
    argumentsAsOneString += ' '
  return argumentsAsOneString

def executeCommand( commandWithPath, argumentsAsList, verbose=True ):
  # all commands that require substantial computation
  # will be executed through this function.
  # This will allow a uniform interface also for various clusters
  argumentsAsOneString = getStringArgumentFromList( argumentsAsList )

  if verbose:
    print('Executing command:')
    print('     ' + commandWithPath + ' ' + argumentsAsOneString )
  # create list for subprocess
  commandList = [ commandWithPath ]
  for el in argumentsAsList:
    commandList.append( el )
  subprocess.call( commandList )


def callPythonScript( cmd ):
  cmdFull = 'python ' + cmd
  p = subprocess.Popen(cmdFull, stdout=subprocess.PIPE, shell=True)
  out, err = p.communicate()
  result = out.split('\n')
  for lin in result:
    if not lin.startswith('#'):
        print(lin)


#    callPythonScript( settings['svmDir'] + '/tools/subset.py' + ' ' + trainDataFile1 + ' 66667 ' + trainDataFile1_small )
