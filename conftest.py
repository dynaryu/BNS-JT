# Import ALL fixtures from 'examples.py'
#from BNS_JT.tests.test_gen_bnb import info_bridge
from BNS_JT.tests.test_trans import data_bridge, setup_bridge, expected_probs
from BNS_JT.tests.test_gen_bnb import main_sys, setup_inference, setup_brs
from BNS_JT.tests.test_operations import setup_sys_rbd
from BNS_JT.tests.test_cpm import setup_condition, setup_hybrid

#, setup_bridge_alt
# Import fixture_a and fixture_b from 'examples.py'
#from root.module.tests.fixtures.examples import fixture_a, fixture_b  
# Import all fixtures from list of plugins
#pytest_plugins = [
#            "root.module.tests.fixtures.examples",
#                "root.module.tests.fixtures.other_examples",
#                ]
