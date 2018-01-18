- Multi-scaling currently does not support propagating additional parameters (for example from smoothers).

- For the learned smoother multi-scale support would require supporting storing and instantiating different smoothers at the different multi-scale levels.
    (i.e., level-specific learned deep networks)

- Write tests for the adaptive smoothers. Test them more and check that the CUDA version in fact works.

- Fix absolute path in data_manager.py (support data location via a variable).

