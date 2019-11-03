$PYTHON setup.py install --single-version-externally-managed --record=record.txt

echo
echo
echo 'Copying settings and data files to the install directory'
echo "cp -r $RECIPE_DIR/mermaid_settings $SP_DIR/mermaid_settings"
echo "cp -r $RECIPE_DIR/mermaid_test_data $SP_DIR/mermaid_test_data"
echo
echo

cp -r $RECIPE_DIR/mermaid_settings $SP_DIR/mermaid_settings
cp -r $RECIPE_DIR/mermaid_test_data $SP_DIR/mermaid_test_data

# Add more build steps here, if they are necessary.

# See
# http://docs.continuum.io/conda/build.html
# for a list of environment variables that are set during the build process.



