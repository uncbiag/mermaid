"%PYTHON%" setup.py install --single-version-externally-managed --record=record.txt

xcopy "%RECIPE_DIR%"\mermaid_settings "%SP_DIR%"\mermaid_settings /O /X /E /H /K
xcopy "%RECIPE_DIR%"\mermaid_test_data "%SP_DIR%"\mermaid_test_data /O /X /E /H /K

if errorlevel 1 exit 1
