import os
import sys

def get_version_from_tag():
    ref = os.getenv('GITHUB_REF', '')
    if ref.startswith('refs/tags/'):
        return ref.split('/')[-1]
    return None

def main():
    version = get_version_from_tag()
    if version is None:
        print("Could not determine version from tag")
        sys.exit(1)

    # Write the version to __init__.py
    init_filepath = 'src/pythonradex/__init__.py'
    with open(init_filepath, 'r') as f:
        lines = f.readlines()

    with open(init_filepath, 'w') as f:
        for line in lines:
            if line.startswith('__version__'):
                f.write(f"__version__ = '{version}'\n")
            else:
                f.write(line)

if __name__ == "__main__":
    main()
