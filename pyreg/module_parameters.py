import json

class ModuleParameters(object):
    def __init__(self):
        self.root_ext = {}
        self.root_intClean = {}
        self.root_intComments = {}

    def loadJSON(self,fileName):
        with open(fileName) as data_file:
            self.root_ext = json.load(data_file)

    def writeJSON(self,fileName):
        with open(fileName, 'w') as outfile:
            json.dump(self.root_intClean, outfile, indent=4, sort_keys=True)

    def writeJSONComments(self,fileNameComments):
        with open(fileNameComments, 'w') as outfile:
            json.dump(self.root_intComments, outfile, indent=4, sort_keys=True)

    def getRoot(self):
        r = {
            'ext': self.root_ext,
            'int': self.root_intClean,
            'com': self.root_intComments,
            'currentCategoryName': 'root'
        }
        return r

def setCurrentKey(par_dicts,keyname,value,comment=None):
    return getCurrentKey(par_dicts,keyname,value,comment,False)

def getCurrentKey(par_dicts,keyname,defaultValue,comment=None,reportDefaultValues=True):
    currentCategoryName = par_dicts['currentCategoryName'] + '.' + keyname
    c_ext = par_dicts['ext']
    c_int = par_dicts['int']
    c_com = par_dicts['com']

    if not c_ext.has_key(keyname):
        c_ext[keyname]=defaultValue
        if reportDefaultValues:
            print('Using default value: ' + str(defaultValue) +
                  '; key = ' + keyname + '; category = ' + currentCategoryName )

    c_int[keyname] = c_ext[keyname]
    c_com[keyname] = comment

    return c_int[keyname]

def setCurrentCategory(par_dicts,keyname,comment=''):

    cc = par_dicts
    for key in keyname.split('.'):
        cc = getCurrentCategory(cc,key,comment,False)
    return cc

def getCurrentCategory(par_dicts,keyname,comment=None,reportDefaultValues=True):
    currentCategoryName = par_dicts['currentCategoryName'] + '.' + keyname
    c_ext = par_dicts['ext']
    c_int = par_dicts['int']
    c_com = par_dicts['com']

    if not c_ext.has_key(keyname):
        c_ext[keyname] = {}
        if reportDefaultValues:
            print('Creating default category: ' + currentCategoryName)

    if not c_int.has_key(keyname):
        c_int[keyname] = {}

    if not c_com.has_key(keyname):
        c_com[keyname] = {}
        if comment is not None:
            if len(comment)>0:
                c_com[keyname]['__doc__'] = comment

    r = {
        'ext': c_ext[keyname],
        'int': c_int[keyname],
        'com': c_com[keyname],
        'currentCategoryName': currentCategoryName
    }

    return r

def testMe():
    p = ModuleParameters()

    d = p.getRoot()
    dc = getCurrentCategory(d,'similarity_measures','defines settings for image similarity measure')
    getCurrentKey(dc,'sigma',0.1,'weight')
    getCurrentKey(dc,'exp',2,'exponent')
    dc = getCurrentCategory(d,'regularizer','settings for the regularizer')
    getCurrentKey(dc,'alpha',0.1,'regalpha')
    getCurrentKey(dc,'beta',0.2,'regbeta')
    dc2 = getCurrentCategory(dc,'subreg','subsettings')
    getCurrentKey(dc2,'test','bla','testvalue')

    p.writeJSON('testOut.json')
    p.writeJSONComments('testOutComments.json')
