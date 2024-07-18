import unittest
from modules.module_registry import create_module_registry

MODULE_REGISTRY = create_module_registry()


class TestAllModules(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.registry = MODULE_REGISTRY

    def _test_module_instantiation(self, module_name):
        module_class = self.registry[module_name]
        try:
            sample_config = module_class.sample_config()
            module = module_class(sample_config, {})
            self.assertIsInstance(module, module_class)
        except Exception as e:
            self.fail(f"{module_name} instantiation failed: {str(e)}")


def create_test_method(module_name):
    def test_method(self):
        self._test_module_instantiation(module_name)
    return test_method


for module_name in MODULE_REGISTRY.keys():
    test_method = create_test_method(module_name)
    test_method_name = f'test_{module_name}'
    setattr(TestAllModules, test_method_name, test_method)

if __name__ == '__main__':
    unittest.main()