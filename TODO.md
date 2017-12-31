- There seems to be an issue with the configuration file. load_default_settings reads the model from variable 'name', but it should be model.registration_model.type

- Fix absolute path in data_manager.py (support data location via a variable).

- Warp image functionality assumes [-1,1] maps. Add support to also be able to use physical coordinates.