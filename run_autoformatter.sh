#!/bin/bash


# Exclude files in third_party folder for isort
isort . --skip third_party

# Exclude files in third_party folder for Black
python -m black . --exclude third_party/

# Exclude files in third_party folder for docformatter
docformatter -i -r . --exclude venv --exclude third_party
