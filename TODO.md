- Write tests for the adaptive smoothers. Test them more and check that the CUDA version in fact works.

- URGENT TODO: There appears to be an inconsistency between the spacing and the map, as the map operates in [-1,1]^d and entirely disregards all the spacing information, whereas typically the input spacing is defined in [0,1]^d

- There seems to be an issue with the configuration file. load_default_settings reads the model from variable 'name', but it should be model.registration_model.type

- Fix absolute path in data_manager.py (support data location via a variable).

- Warp image functionality assumes [-1,1] maps. Add support to also be able to use physical coordinates.