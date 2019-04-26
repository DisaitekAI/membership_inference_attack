#! /bin/sh

sphinx-apidoc -f -o . ../src/utils
sphinx-apidoc -f -o . ../src/tests
sphinx-apidoc -f -o . ../src/data
sphinx-apidoc -f -o . ../src/models

mfile=`cat <<EOF
modules
=======

.. toctree::
   :maxdepth: 4
   
   models
   data
   utils
   tests
EOF`

echo "$mfile" > modules.rst

