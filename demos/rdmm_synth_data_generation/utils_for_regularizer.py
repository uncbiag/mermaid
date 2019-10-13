import mermaid.module_parameters as pars
import mermaid.smoother_factory as sf

def get_single_gaussian_smoother(gaussian_std,sz,spacing):
    s_m_params = pars.ParameterDict()
    s_m_params['smoother']['type'] = 'gaussian'
    s_m_params['smoother']['gaussian_std'] = gaussian_std
    s_m = sf.SmootherFactory(sz, spacing).create_smoother(s_m_params)
    return s_m