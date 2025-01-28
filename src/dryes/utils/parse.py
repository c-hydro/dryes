import itertools
import copy

def make_case_hierarchy(cases, opt_groups):
    options = cases_to_options(cases)
    option_hierarchy = make_option_hierarchy(options, opt_groups)

    cases = []
    for options in option_hierarchy:
        cases.append(options_to_cases(options))

    return cases

def make_option_hierarchy(options, opt_groups):

    opt_permutations = []
    for i in range(len(opt_groups)):
        opts = opt_groups[i]
        #for j in range(i+1): opts += opt_groups[j]
        #breakpoint()
        group_options = {k: v for k, v in options.items() if k in opts}
        fixed_options = {k: v for k, v in group_options.items() if not isinstance(v, dict)}
        to_permutate = {k: [{kv:vv} for kv,vv in v.items()] 
                                    for k, v  in group_options.items() if isinstance(v, dict)}
        values_to_permutate = [v for v in to_permutate.values()]
        keys = list(to_permutate.keys())
        permutations = [dict(zip(keys, p)) for p in itertools.product(*values_to_permutate)]

        for p in permutations:
            p.update(fixed_options)
        opt_permutations.append(permutations)
        #breakpoint()
    
    opt_hierarchy = []
    for i in range(len(opt_groups)):
        these_options = {k: v for k, v in options.items() if k in opt_groups[i]}
        if i == 0:
            opt_hierarchy.append(these_options)
        else:
            this_level_options = []
            for old_options in  opt_permutations[i-1]: #.copy()
            #breakpoint()
            #new_options = combine_options(old_options, these_options)
            #this_level_options.append(new_options)
                #new_options = combine_options(old_options, these_options)
                new_options = copy.deepcopy(these_options)
                new_options.update(old_options)
                this_level_options.append(new_options)
            opt_hierarchy.append(this_level_options)

    return opt_hierarchy

def combine_options(old_options, new_options):
    levels = len(old_options)
    if levels == 1:
        combined = []
        for options in old_options[0]:
            #if isinstance(options, str): breakpoint()
            these_options = options.copy()
            if not isinstance(new_options, list): new_options = [new_options]
            for new_option_set in new_options:
                these_options_copy = these_options.copy()
                these_options_copy.update(new_option_set)
                combined.append(these_options_copy)

        return combined

    while len(old_options) > 1:
        these_old_options = [old_options.pop(-1)]
        new_options = combine_options(these_old_options, new_options)

    return combine_options(old_options, new_options)
      



    # if isinstance(old_options, list):
    #     these_options = []
    #     for case_options in old_options:
    #         these_options.append(combine_options(case_options, new_options))
    #     return these_options
    
    # new_options = copy.deepcopy(new_options)
    # new_options.update(old_options)
    # return new_options
    
    # opt_hierarchy = []
    # for i in range(len(opt_groups)):
    #     if i == 0:
    #         opt_hierarchy.append(opt_permutations[0])
    #     else:
    #         opt_hierarchy.append([])
    #         for parent in opt_hierarchy[i-1]:
    #             these_options = []
    #             for child in opt_permutations[i]:
    #                 this_parent = copy.deepcopy(parent)
    #                 this_parent.update(child)
    #                 these_options.append(this_parent)
    #             opt_hierarchy[i].append(these_options)
          
def permutate_options(options):

    # get the options that need to be permutated and the ones that are fixed
    fixed_options = {k: v for k, v in options.items() if not isinstance(v, dict)}
    to_permutate = {k: list(v.keys()) for k, v in options.items() if isinstance(v, dict)}
    values_to_permutate = [v for v in to_permutate.values()]
    keys = list(to_permutate.keys())

    permutations = [dict(zip(keys, p)) for p in itertools.product(*values_to_permutate)]
    identifiers = copy.deepcopy(permutations)
    for permutation in permutations:
        permutation.update(fixed_options)

    return permutations, identifiers

def cases_to_options(cases):
    """
    convert the cases to options -> this is the reverse of options_to_cases
    """
    options = {}
    for case in cases:
        for key, value in case['options'].items():
            if case['tags'][key] == "":
                options[key] = value
            else:
                if key not in options: options[key] = {}
                options[key].update({case['tags'][key]: value})
    return options

def options_to_cases(options):
    """
    convert the options to cases -> this is the reverse of cases_to_options
    """
    # this allows us to to this recursively
    if isinstance(options, list):
        cases = []
        for these_options in options:
            cases.append(options_to_cases(these_options))
        return cases
    
    # get the options that need to be permutated and the ones that are fixed
    permutations, identifiers = permutate_options(options)

    cases_opts = []
    for permutation in permutations:
        this_case_opts = {}
        for k, v in permutation.items():
            # if this is one of the options that we permutated
            if isinstance(options[k], dict):
                this_case_opts[k] = options[k][v]
            # if not, this is fixed
            else:
                this_case_opts[k] = v
                permutation[k] = ""
        cases_opts.append(this_case_opts)

    opt_cases = []
    for case, permutation, i in zip(cases_opts, permutations, range(len(identifiers))):
        this_case = dict()
        this_case['id']   = i
        this_case['name'] = ','.join(v for v in permutation.values() if v != "")
        this_case['tags'] = {pk:pv for pk,pv in permutation.items()}
        this_case['options'] = case
        opt_cases.append(this_case)

    return opt_cases
