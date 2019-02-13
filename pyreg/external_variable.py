use_mermaid_net = False
reg_factor_in_mermaid = 1.
try:
    from model_pool.global_variable import use_mermaid_iter, reg_factor_in_mermaid
    use_mermaid_net = not use_mermaid_iter
except:
    pass
print("use_mermaid_net:{}".format(use_mermaid_net))
print("reg_factor_in_mermaid:{}".format(reg_factor_in_mermaid))