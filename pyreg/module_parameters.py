"""
This package implements a simple way of dealing with parameters and providing
default parameters and comments.
"""

import json

class ParameterDict(object):
    def __init__(self):
        self.ext = {}
        self.int = {}
        self.com = {}
        self.currentCategoryName = 'root'
        self.printSettings = True

    def __str__(self):

        return 'ext = ' + self.ext.__str__() + "\n" + \
            'int = ' + self.int.__str__() + "\n" + \
            'com = ' + self.com.__str__() + "\n" + \
            'currentCategoryName = ' + str( self.currentCategoryName) +"\n"

    def loadJSON(self, fileName):
        with open(fileName) as data_file:
            if self.printSettings:
                print('Loading parameter file = ' + fileName )
            self.ext = json.load(data_file)

    def writeJSON(self, fileName):
        with open(fileName, 'w') as outfile:
            if self.printSettings:
                print('Writing parameter file = ' + fileName )
            json.dump(self.int, outfile, indent=4, sort_keys=True)

    def writeJSONComments(self, fileNameComments):
        with open(fileNameComments, 'w') as outfile:
            if self.printSettings:
                print('Writing parameter file = ' + fileNameComments )
            json.dump(self.com, outfile, indent=4, sort_keys=True)

    def printSettingsOn(self):
        self.printSettings = True

    def printSettingsOff(self):
        self.printSettings = False

    def getPrintSettings(self):
        return self.printSettings

    def _setValueOfInstance(self,ext,int,com,currentCategoryName):
        self.ext = ext
        self.int = int
        self.com = com
        self.currentCategoryName = currentCategoryName

    def __missing__(self, key):
        # if key cannot be found
        raise ValueError('Could not find key = ' + str( key ) )

    def __getitem__(self, key_or_keyTuple):
        # getting an item based on key
        # here the key can be three different things
        # 1) simply a text key (then returns the current value)
        # 2) A 2-tuple (keyname,defaultvalue)
        # 3) A 3-tuple (keyname,defaultvalue,comment)

        # returns a ParDicts object if we are accessing a category (i.e., a dictionary)
        # returns just the value if it is a regular value

        if type(key_or_keyTuple)==tuple:
            # here, we need to distinguish cases 2) and 3)
            lT = len(key_or_keyTuple)
            if lT==1:
                # treat this as if it would only be the keyword
                return self._getCurrentKey( key_or_keyTuple[0] )
            elif lT==2:
                # treat this as keyword + default value
                return self._getCurrentKey( key_or_keyTuple[0],
                                             key_or_keyTuple[1] )
            elif lT==3:
                # treat this as keyword + default value + comment
                return self._getCurrentKey( key_or_keyTuple[0],
                                             key_or_keyTuple[1],
                                             key_or_keyTuple[2] )
            else:
                raise ValueError('Tuple of incorrect size')
        else:
            # now we just want to return it (there is no default value or comment)
            return self._getCurrentKey( key_or_keyTuple )


    def __setitem__(self, key, valueTuple):
        # to set an item
        # valueTuple is either a 2-tuple (actual value, comment)
        # or it is simply a comment, then this key becomes a category
        if type(valueTuple)==tuple:
            if len(valueTuple)==2:
                value = valueTuple[0]
                comment = valueTuple[1]
            elif len(valueTuple)==1:
                value = {}
                comment = valueTuple[0]
            else:
                raise ValueError('Expected a 2-tuple as input')
        else: # not a tuple
            value = valueTuple
            comment = None

        if type(value)==dict:
            # only add if this is an empty dictionary
            if len(value)==0:
                self._setCurrentCategory(key,comment)
            else:
                raise ValueError('Can only add empty dictionaries')
            # we are assigning a category
        else:
            # now we have to set an actual value (not a category)
            self._setCurrentKey(key,value,comment)

    def _setCurrentCategory(self, key, comment):
        currentCategoryName = self.currentCategoryName + '.' + str(key)

        if not self.ext.has_key(key) or (self.ext.has_key(key) and type(self.ext[key])!=dict):
            # we do not want to over-write any settings here
            if self.printSettings:
                print('Creating new category: ' + currentCategoryName)
                self.ext[key] = {}

        self.int[key] = {}
        self.com[key] = {}

        if comment is not None:
            if len(comment) > 0:
                self.com[key]['__doc__'] = comment

    def _setCurrentKey(self, key, value, comment=None):

        if self.printSettings:
            if self.ext.has_key(key):
                print('Overwriting key = ' + str(key) + '; category = ' + self.currentCategoryName + '; value =  ' +
                      str( self.ext[key] ) + ' -> ' + str(value) )
            else:
                print('Creating key = ' + str(key) + '; category = ' + self.currentCategoryName + '; value = ' + str(value))

        self.ext[key] = value
        self.int[key] = value
        if comment is not None:
            if len(comment)>0:
                self.com[key] = comment


    def _getCurrentKey(self, key, defaultValue=None, comment=None):

        # returns a ParDicts object if we are accessing a category (i.e., a dictionary)
        # returns just the value if it is a regular value

        if self.ext.has_key(key):
            value = self.ext[key]
            if type(value)==dict:
                # this is a category, need to create a ParDicts object to return
                # if the key already exists in int and com keep it otherwise initialize it to empty
                if not self.int.has_key(key):
                    self.int[key]={}
                if not self.com.has_key(key):
                    self.com[key]={}
                    if comment is not None:
                        if len(comment)>0:
                            self.com[key]['__doc__'] = comment

                newpar = ParameterDict()
                currentCategoryName = self.currentCategoryName + '.' + str(key)
                newpar._setValueOfInstance(self.ext[key],self.int[key],self.com[key],currentCategoryName)

                return newpar
            else:
                # just a regular value which we can return
                self.int[key] = value
                if comment is not None:
                    if len(comment)>0:
                        self.com[key] = comment

                return value
        else:
            # does not have the key, create it via the default value
            if defaultValue is not None:
                if type(defaultValue)==dict:
                    # make sure it is empty and if it is create a category
                    if len(defaultValue)==0:
                        self._setCurrentCategory(key,comment)
                        # and now we need to return it
                        newpar = ParameterDict()
                        currentCategoryName = self.currentCategoryName + '.' + str(key)
                        newpar._setValueOfInstance(self.ext[key], self.int[key], self.com[key], currentCategoryName)

                        return newpar
                    else:
                        raise ValueError('Cannot create a default key of type dict()')
                else:
                    # now we can create it and return it
                    self.ext[key]=defaultValue
                    self.int[key]=defaultValue
                    if comment is not None:
                        if len(comment)>0:
                            self.com[key]=comment
                    if self.printSettings:
                        print('Using default value = ' + str(defaultValue) + ' for key = ' + str(key) + ' of category = ' + self.currentCategoryName  )

                    return defaultValue
            else:
                raise ValueError('Cannot create key = ' + str(key) + ' without a default value')


# test it
def testParameterDict():
    p = ParameterDict()

    # we can directly assign
    p['registration_model'] = ({},'general settings for registration models')
    p['registration_model']['similarity_measure'] = ({},'settings for the similarity measures')
    p['registration_model']['similarity_measure']['type']=('ssd','similarity measure type')
    # we can also ask for a parameter and use a default parameter if it does not exist
    p['registration_model'][('nrOfIterations',10,'number of iterations')]

    # we can also create a new category with default values if it does not exist yet
    p[('new_category',{},'this is a new category')]
    p[('registration_model',{},'this category already existed')]

    # and we can print everything of course
    print(p)

    # lastly we can write it all out as json
    p.writeJSON('test_pars.json')
    p.writeJSONComments('test_pars_comments.json')



