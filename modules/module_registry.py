import os
import importlib
import inspect
from utils.modules import TTSModule


def discover_modules(package_name):
    """Discover all modules in a package, including subfolders."""
    package = importlib.import_module(package_name)
    package_dir = os.path.dirname(package.__file__)
    modules = {}

    for root, _, files in os.walk(package_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                file_path = os.path.join(root, file)
                module_path = os.path.relpath(file_path, package_dir)
                module_name = f"{package_name}.{os.path.splitext(module_path)[0].replace(os.path.sep, '.')}"
                try:
                    modules[module_name] = importlib.import_module(module_name)
                except ImportError as e:
                    print(f"Error importing {module_name}: {e}")

    return modules


def get_tts_subclasses(module):
    """Get all TTSModule subclasses in a module."""
    return {
        name: cls
        for name, cls in inspect.getmembers(module, inspect.isclass)
        if issubclass(cls, TTSModule) and cls is not TTSModule
    }


def create_module_registry():
    """Create a registry of all TTSModule subclasses in the modules package."""
    modules = discover_modules('modules')  # Adjust 'modules' to your package name if different
    registry = {}

    for module_name, module in modules.items():
        registry.update(get_tts_subclasses(module))

    return registry


if __name__ == '__main__':
    MODULE_REGISTRY = create_module_registry()