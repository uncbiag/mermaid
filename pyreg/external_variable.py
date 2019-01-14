use_mermaid_net = False
try:
    from model_pool.global_variable import use_mermaid_iter
    use_mermaid_net = not use_mermaid_iter
except:
    pass
