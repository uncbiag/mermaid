# start with the setup

import os
import sys

sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../mermaid'))
sys.path.insert(0,os.path.abspath('../mermaid/libraries'))

import unittest
import imp

try:
    imp.find_module('HtmlTestRunner')
    foundHTMLTestRunner = True
    import HtmlTestRunner
except ImportError:
    foundHTMLTestRunner = False

# done with all the setup

# testing code starts here

import module_parameters as MP

# test it

class Test_module_parameters(unittest.TestCase):

    def setUp(self):
        self.p = MP.ParameterDict()

    def tearDown(self):
        pass

    def test_set_category(self):
        self.p['registration_model'] = ({},'settings')
        assert(self.p.int=={'registration_model':{}})
        assert(self.p.ext=={'registration_model':{}})
        assert(self.p.com=={'registration_model': {'__doc__':'settings'}})

    def test_set_two_categories_two_step(self):
        self.p['r'] = ({},'rs')
        self.p['r']['s']=({},'settings')
        assert(self.p.int=={'r': {'s': {}}})
        assert(self.p.ext=={'r': {'s': {}}})
        assert(self.p.com=={'r': {'__doc__': 'rs', 's': {'__doc__': 'settings'}}})

    def test_set_two_categories(self):
        self.p['r']['s']=({},'settings')
        assert(self.p.int=={'r': {'s': {}}})
        assert(self.p.ext=={'r': {'s': {}}})
        assert(self.p.com=={'r': {'s': {'__doc__': 'settings'}}})

    def test_set_two_categories_and_one_entry_three_step(self):
        self.p['r'] = ({},'rs')
        self.p['r']['s']=({},'settings')
        self.p['r']['s']['t'] = ('ssd', 'setting')
        assert(self.p.int=={'r': {'s': {'t': 'ssd'}}})
        assert(self.p.ext=={'r': {'s': {'t': 'ssd'}}})
        assert(self.p.com=={'r': {'__doc__': 'rs', 's': {'__doc__': 'settings', 't':'setting'}}})

    def test_set_two_categories_and_one_entry(self):
        self.p['r']['s']['t'] = ('ssd', 'setting')
        assert(self.p.int=={'r': {'s': {'t': 'ssd'}}})
        assert(self.p.ext=={'r': {'s': {'t': 'ssd'}}})
        assert(self.p.com=={'r': {'s': {'t': 'setting'}}})

    def test_set_default_parameter(self):
        assert( self.p['registration_model'][('nrOfIterations', 10, 'number of iterations')] == 10 )

if __name__ == '__main__':
    if foundHTMLTestRunner:
        unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner(output='test_output'))
    else:
       unittest.main()

