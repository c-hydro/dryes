from datetime import datetime

def substitute_values(structure, tag_dict, rec = False):
    """
    replace the {tags} in the structure with the values in the tag_dict
    """

    if isinstance(structure, dict):
        return {key: substitute_values(value, tag_dict, rec) for key, value in structure.items()}
    elif isinstance(structure, list):
        return [substitute_values(value, tag_dict, rec) for value in structure]
    elif isinstance(structure, int):
        return substitute_string(str(structure), tag_dict, rec)
    elif isinstance(structure, str):
        return substitute_string(structure, tag_dict, rec)
    else:
        return structure
    
def substitute_string(string, tag_dict, rec = False):
    """
    replace the {tags} in the string with the values in the tag_dict
    """
    while "{" in string and "}" in string:
        for key, value in tag_dict.items():
            #if key == "history_start" : breakpoint()
            # check if the value is a datetime object and the string contains a format specifier for the key
            if isinstance(value, datetime) and '{' + key + ':' in string:
                # extract the format specifier from the string
                fmt = string.split('{' + key + ':')[1].split('}')[0]
                # format the value using the format specifier
                value = value.strftime(fmt)
                key = key + ':' + fmt
            # replace the bracketed part with the value
            string = string.replace('{' +   key + '}', str(value))
        if not rec:
            break
    return string