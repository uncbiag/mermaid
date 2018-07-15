import warnings

def formatwarning(message, category, filename, lineno, line=None):
    """Function to format a warning the standard way."""
    msg = warnings.WarningMessage(message, category, filename, lineno, None, line)
    return warnings._formatwarnmsg_impl(msg)

warnings.formatwarning = formatwarning